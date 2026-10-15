from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from types import MappingProxyType
from typing import Any, cast

import numpy as np

from ..errors import CapabilityError, RuntimeContractError
from ..runtime import OnnxSession
from ..types import InferenceResult, Installation, RuntimeDiagnostic
from .base import SystemAdapter
from .qwen3_audio import (
    QWEN3_BASE_MAX_REFERENCE_SAMPLES,
    QWEN3_BASE_MAX_REFERENCE_SECONDS,
    QWEN3_CODEC_SAMPLES_PER_FRAME,
    Qwen3VoiceState,
    compute_log_mel,
    validate_reference_audio,
    validate_reference_token_ids,
)
from .qwen3_contract import (
    LAYOUT,
    OPTIONAL_EMBEDDINGS,
    REQUIRED_EMBEDDINGS,
    Qwen3Bundle,
    validate_graph_abi,
)
from .qwen3_generation import voice_clone_result, voice_design_result
from .qwen3_sampling import Qwen3GenerationConfig

_GENERATION_COMPONENTS = {"talker_prefill", "talker_decode", "code_predictor", "vocoder"}
_ENCODER_COMPONENTS = {"speaker_encoder", "tokenizer_encoder"}
_TOKENIZER_FILES = {
    "tokenizer_json": "tokenizer.json",
    "vocab": "vocab.json",
    "merges": "merges.txt",
    "tokenizer_config": "tokenizer_config.json",
    "special_tokens": "special_tokens_map.json",
    "added_tokens": "added_tokens.json",
}


class Qwen3Adapter(SystemAdapter):
    """Managed or unmanaged WaveKat Qwen3 bundle with strict file inventory checks."""

    system = "qwen3"

    def __init__(self, installation: Installation, **kwargs: Any) -> None:
        super().__init__(installation, **kwargs)
        runtime = installation.metadata.get("runtime")
        if not isinstance(runtime, Mapping):
            raise RuntimeContractError("Qwen3 installation runtime metadata must be an object")
        if runtime.get("layout") != LAYOUT:
            raise RuntimeContractError(f"Qwen3 runtime layout must be {LAYOUT!r}")
        model_type = runtime.get("model_type")
        if not isinstance(model_type, str) or model_type not in {"base", "voice_design"}:
            raise RuntimeContractError("Qwen3 runtime model_type must be 'base' or 'voice_design'")
        self.model_type: str = model_type
        quality = runtime.get("quality")
        if not isinstance(quality, str) or quality not in {"fp32", "int4"}:
            raise RuntimeContractError("Qwen3 runtime quality must be 'fp32' or 'int4'")
        self.quality: str = quality
        if installation.sample_rate != 24000 or runtime.get("sample_rate") != 24000:
            raise RuntimeContractError("Qwen3 WaveKat installations require sample_rate=24000")
        selected_quality = installation.metadata.get("selected_quality")
        if selected_quality is not None and selected_quality != self.quality:
            raise RuntimeContractError("Qwen3 runtime quality conflicts with the installed profile")
        self._sessions: dict[str, OnnxSession] = {}
        self._provider_policy: tuple[str, ...] | None = None
        self._validate_inventory()

    def _validate_inventory(self) -> None:
        expected_models = set(_GENERATION_COMPONENTS)
        if self.model_type == "base":
            expected_models.update(_ENCODER_COMPONENTS)

        expected_paths: dict[tuple[str, str | None, str | None], str] = {
            ("config", None, None): "config.json",
        }
        expected_paths.update(
            {
                ("tokenizer", component, None): f"tokenizer/{filename}"
                for component, filename in _TOKENIZER_FILES.items()
            }
        )
        expected_paths.update(
            {
                ("embedding", component, None): f"embeddings/{component}.npy"
                for component in REQUIRED_EMBEDDINGS
            }
        )
        for component in expected_models:
            quality = None if component in _ENCODER_COMPONENTS else self.quality
            graph_path = f"{component}.onnx" if quality is None else f"{quality}/{component}.onnx"
            expected_paths[("model", component, quality)] = graph_path
            expected_paths[("model_data", component, quality)] = graph_path + ".data"

        actual: dict[tuple[str, str | None, str | None], Any] = {}
        optional_embeddings: set[str] = set()
        for artifact in self.installation.artifacts:
            artifact_metadata: Any = cast(Any, artifact.metadata)
            if not isinstance(artifact_metadata, Mapping):
                raise RuntimeContractError("Qwen3 artifact metadata must be an object")
            artifact_component: Any = artifact.component
            if artifact_component is not None and not isinstance(artifact_component, str):
                raise RuntimeContractError("Qwen3 artifact component must be a string or null")
            key: tuple[str, str | None, str | None] = (
                artifact.role,
                artifact_component,
                artifact.quality,
            )
            if key in actual:
                raise RuntimeContractError(f"Duplicate Qwen3 artifact {key!r}")
            actual[key] = artifact
            if (
                artifact.role == "embedding"
                and isinstance(artifact_component, str)
                and artifact_component in OPTIONAL_EMBEDDINGS
            ):
                optional_embeddings.add(artifact_component)

        expected_keys = set(expected_paths)
        allowed_keys = expected_keys | {
            ("embedding", component, None) for component in OPTIONAL_EMBEDDINGS
        }
        missing = expected_keys - actual.keys()
        extra = actual.keys() - allowed_keys
        if missing or extra:
            raise CapabilityError(
                f"Qwen3 bundle inventory mismatch; missing={sorted(missing, key=repr)}, extra={sorted(extra, key=repr)}"
            )
        for key, expected_path in expected_paths.items():
            artifact = actual[key]
            if artifact.metadata.get("path") != expected_path:
                raise RuntimeContractError(
                    f"Qwen3 artifact {key!r} path metadata must be {expected_path!r}"
                )
        for component in optional_embeddings:
            artifact = actual[("embedding", component, None)]
            expected_path = f"embeddings/{component}.npy"
            if artifact.metadata.get("path") != expected_path:
                raise RuntimeContractError(
                    f"Qwen3 artifact ('embedding', {component!r}) path metadata is invalid"
                )

        for component in expected_models:
            quality = None if component in _ENCODER_COMPONENTS else self.quality
            graph = actual[("model", component, quality)]
            data = actual[("model_data", component, quality)]
            if graph.path.name != f"{component}.onnx" or data.path.name != f"{component}.onnx.data":
                raise RuntimeContractError(
                    f"Qwen3 graph and external-data filenames disagree for {component!r}"
                )
            if data.path.resolve() != Path(str(graph.path.resolve()) + ".data"):
                raise RuntimeContractError(
                    f"Qwen3 external data must be adjacent to the {component!r} graph"
                )

        config_path = actual[("config", None, None)].path
        embedding_paths = {
            component: artifact.path
            for (role, component, _quality), artifact in actual.items()
            if role == "embedding" and component is not None
        }
        model_paths: dict[str, Path] = {}
        external_data_paths: dict[str, Path] = {}
        for component in expected_models:
            quality = None if component in _ENCODER_COMPONENTS else self.quality
            model_paths[component] = actual[("model", component, quality)].path
            external_data_paths[component] = actual[("model_data", component, quality)].path
        self.bundle = Qwen3Bundle.load(
            config_path=config_path,
            embedding_paths=embedding_paths,
            model_paths=model_paths,
            external_data_paths=external_data_paths,
            expected_model_type=self.model_type,
            quality=self.quality,
        )
        self.config = self.bundle.config
        self.embeddings = self.bundle.embeddings
        if self.config.sample_rate != self.installation.sample_rate:
            raise RuntimeContractError(
                "Qwen3 config sample rate conflicts with installation metadata"
            )

    def _required_graph_components(self) -> tuple[str, ...]:
        required = set(_GENERATION_COMPONENTS)
        if self.model_type == "base":
            required.update(_ENCODER_COMPONENTS)
        order = (
            "speaker_encoder",
            "tokenizer_encoder",
            "talker_prefill",
            "talker_decode",
            "code_predictor",
            "vocoder",
        )
        return tuple(component for component in order if component in required)

    def _get_session(self, component: str) -> OnnxSession:
        if component not in self._required_graph_components():
            raise CapabilityError(f"Qwen3 {self.model_type} bundle has no {component!r} graph")
        if component in self._sessions:
            return self._sessions[component]
        try:
            model_path = self.bundle.model_paths[component]
        except KeyError as exc:
            raise RuntimeContractError(f"Qwen3 bundle is missing graph {component!r}") from exc
        session = OnnxSession(
            model_path,
            component=component,
            providers=self.providers,
            provider_options=self.provider_options,
            session_options=self.session_options,
            strict_providers=True,
        )
        try:
            validate_graph_abi(
                component,
                self.config,
                session.input_specs,
                session.output_specs,
            )
            requested = tuple(session.resolved_providers)
            active = tuple(getattr(session, "active_providers", requested))
            if not requested:
                raise RuntimeContractError("Qwen3 graph session resolved an empty provider policy")
            if not set(requested).intersection(active):
                raise RuntimeContractError(
                    f"Qwen3 {component} did not activate any requested provider: {active!r}"
                )
            unexpected = set(active) - set(requested) - {"CPUExecutionProvider"}
            if unexpected:
                raise RuntimeContractError(
                    f"Qwen3 {component} activated unexpected providers: {sorted(unexpected)}"
                )
            if self._provider_policy is not None and requested != self._provider_policy:
                raise RuntimeContractError(
                    "Qwen3 graph sessions resolved inconsistent provider policies"
                )
            self._provider_policy = requested
        except Exception:
            session.close()
            raise
        self._sessions[component] = session
        return session

    def _ensure_sessions(self) -> None:
        """Initialize and validate the complete model-specific graph set atomically."""
        existing = set(self._sessions)
        try:
            for component in self._required_graph_components():
                self._get_session(component)
        except Exception:
            for component in set(self._sessions) - existing:
                session = self._sessions.pop(component)
                session.close()
            if not self._sessions:
                self._provider_policy = None
            raise

    def close(self) -> None:
        super().close()
        self._provider_policy = None

    def diagnostics(self) -> RuntimeDiagnostic:
        base = super().diagnostics()
        providers = self.providers
        if providers is None:
            requested: tuple[str, ...] | None = None
        elif isinstance(providers, str):
            requested = tuple(part.strip() for part in providers.split(",") if part.strip())
        else:
            requested = tuple(providers)
        active = tuple(
            sorted({provider for session in base.sessions for provider in session.providers_active})
        )
        properties: dict[str, Any] = {
            "model_type": self.model_type,
            "quality": self.quality,
            "sample_rate": self.config.sample_rate,
            "talker_num_layers": self.config.talker_num_layers,
            "talker_hidden_size": self.config.talker_hidden_size,
            "talker_num_code_groups": self.config.talker_num_code_groups,
            "reference_max_seconds": QWEN3_BASE_MAX_REFERENCE_SECONDS
            if self.model_type == "base"
            else None,
            "provider": active or requested,
            "providers_requested": requested,
            "providers_active": active,
        }
        return RuntimeDiagnostic(
            system=base.system,
            ref=base.ref,
            layout=base.layout,
            sessions=base.sessions,
            properties=MappingProxyType(properties),
        )

    def prepare_voice(
        self,
        audio: np.ndarray,
        *,
        sample_rate: int,
        reference_token_ids: Sequence[int],
    ) -> Qwen3VoiceState:
        """Prepare immutable Base speaker and codec conditioning from finite mono 24-kHz audio."""
        if self.model_type != "base":
            raise CapabilityError("Qwen3 prepare_voice requires a base bundle")
        samples = validate_reference_audio(audio, sample_rate=sample_rate)
        token_ids = validate_reference_token_ids(
            reference_token_ids,
            text_embedding_rows=self.embeddings["text_embedding"].shape[0],
            im_start_token_id=self.config.im_start_token_id,
            im_end_token_id=self.config.im_end_token_id,
        )
        reference_frames = (
            samples.size + QWEN3_CODEC_SAMPLES_PER_FRAME - 1
        ) // QWEN3_CODEC_SAMPLES_PER_FRAME
        try:
            mel_outputs = self._get_session("speaker_encoder").run(
                {"mels": compute_log_mel(samples)}
            )
        except Exception as exc:
            raise RuntimeContractError(f"Qwen3 speaker_encoder inference failed: {exc}") from exc
        if not isinstance(mel_outputs, Sequence) or len(mel_outputs) != 1:
            raise RuntimeContractError("Qwen3 speaker_encoder must return one speaker embedding")
        try:
            speaker = np.asarray(mel_outputs[0], dtype=np.float32)
        except (TypeError, ValueError, OverflowError) as exc:
            raise RuntimeContractError(
                "Qwen3 speaker_encoder returned an invalid embedding"
            ) from exc
        if speaker.shape != (1, self.config.talker_hidden_size) or not np.isfinite(speaker).all():
            raise RuntimeContractError(
                "Qwen3 speaker_encoder returned an invalid embedding shape or values"
            )

        padded_waveform = np.zeros((1, QWEN3_BASE_MAX_REFERENCE_SAMPLES), dtype=np.float32)
        padded_waveform[0, : samples.size] = samples
        try:
            tokenizer_outputs = self._get_session("tokenizer_encoder").run(
                {"waveform": padded_waveform}
            )
        except Exception as exc:
            raise RuntimeContractError(f"Qwen3 tokenizer_encoder inference failed: {exc}") from exc
        if not isinstance(tokenizer_outputs, Sequence) or len(tokenizer_outputs) != 1:
            raise RuntimeContractError("Qwen3 tokenizer_encoder must return one codec tensor")
        codes = np.asarray(tokenizer_outputs[0])
        if (
            codes.ndim != 3
            or codes.shape[0] != 1
            or codes.shape[1] != self.config.talker_num_code_groups
            or codes.shape[2] < reference_frames
            or not np.issubdtype(codes.dtype, np.integer)
        ):
            raise RuntimeContractError(
                "Qwen3 tokenizer_encoder returned an invalid codec tensor shape or dtype"
            )
        codes = np.ascontiguousarray(codes[:, :, :reference_frames], dtype=np.int64)
        if np.any(codes < 0):
            raise RuntimeContractError("Qwen3 tokenizer_encoder returned negative codec IDs")
        if np.any(codes[0, 0] >= self.config.talker_vocab_size):
            raise RuntimeContractError(
                "Qwen3 tokenizer_encoder group-0 ID exceeds the talker vocabulary"
            )
        for group in range(1, self.config.talker_num_code_groups):
            if np.any(codes[0, group] >= self.config.cp_vocab_size):
                raise RuntimeContractError(
                    f"Qwen3 tokenizer_encoder group-{group} ID exceeds the predictor vocabulary"
                )
        model_id = self.config.model_id or self.installation.id
        return Qwen3VoiceState(
            speaker_embedding=speaker,
            reference_codes=codes,
            reference_token_ids=token_ids,
            sample_rate=self.config.sample_rate,
            reference_samples=int(samples.size),
            model_id=model_id,
        )

    def infer(
        self,
        token_ids: Sequence[int],
        *,
        instruction_token_ids: Sequence[int] | None = None,
        voice_state: Qwen3VoiceState | None = None,
        language: str = "english",
        temperature: float = 0.9,
        top_k: int = 50,
        repetition_penalty: float = 1.05,
        max_new_tokens: int = 2048,
        seed: int | None = None,
        **kwargs: Any,
    ) -> InferenceResult:
        if not isinstance(language, str) or not language:
            raise RuntimeContractError("Qwen3 language must be a non-empty string")
        if kwargs:
            raise RuntimeContractError(
                "Unsupported Qwen3 inference arguments: " + ", ".join(sorted(kwargs))
            )
        generation = Qwen3GenerationConfig(
            temperature=temperature,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            max_new_tokens=max_new_tokens,
            seed=seed,
        )
        if self.model_type == "voice_design":
            if instruction_token_ids is None:
                raise RuntimeContractError("Qwen3 VoiceDesign requires instruction_token_ids")
            if voice_state is not None:
                raise RuntimeContractError("Qwen3 VoiceDesign does not accept a Base voice_state")
        elif self.model_type == "base":
            if instruction_token_ids is not None:
                raise RuntimeContractError(
                    "Qwen3 Base inference does not accept instruction_token_ids"
                )
            if not isinstance(voice_state, Qwen3VoiceState):
                raise RuntimeContractError(
                    "Qwen3 Base inference requires a prepared Qwen3VoiceState"
                )
        else:
            raise CapabilityError(f"Unsupported Qwen3 model type {self.model_type!r}")

        self._ensure_sessions()
        providers_active = tuple(
            sorted(
                {
                    provider
                    for session in self._sessions.values()
                    for provider in getattr(session, "active_providers", ())
                }
            )
        )
        model_id = self.config.model_id or self.installation.id
        if self.model_type == "voice_design":
            assert instruction_token_ids is not None
            return voice_design_result(
                config=self.config,
                embeddings=self.embeddings,
                sessions=self._sessions,
                token_ids=token_ids,
                instruction_token_ids=instruction_token_ids,
                language=language,
                generation=generation,
                model_id=model_id,
                quality=self.quality,
                providers_active=providers_active,
            )
        assert voice_state is not None
        return voice_clone_result(
            config=self.config,
            embeddings=self.embeddings,
            sessions=self._sessions,
            token_ids=token_ids,
            voice=voice_state,
            language=language,
            generation=generation,
            model_id=model_id,
            quality=self.quality,
            providers_active=providers_active,
        )
