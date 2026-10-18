from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from types import MappingProxyType
from typing import Any

import numpy as np

from ..errors import CapabilityError, RuntimeContractError
from ..runtime import OnnxSession, resolve_providers
from ..types import InferenceResult, Installation
from .base import SystemAdapter
from .kitten2_contract import load_kitten2_bundle
from .kitten2_decoder import Kitten2Decoder, _cpu_session_options
from .kitten2_frontend import Kitten2Frontend, PreparedVoice, build_generation_prompt
from .kitten2_generation import Kitten2LanguageModel
from .kitten2_sampling import sampling_settings_from_config
from .kitten2_tokens import TokenMap


class Kitten2Adapter(SystemAdapter):
    system = "kitten2"

    def __init__(
        self,
        installation: Installation,
        *,
        providers: str | Sequence[str] | None = None,
        provider_options: Sequence[dict[str, Any]] | Mapping[str, dict[str, Any]] | None = None,
        session_options: Any | None = None,
    ) -> None:
        super().__init__(
            installation,
            providers=providers,
            provider_options=provider_options,
            session_options=session_options,
        )
        bundle = load_kitten2_bundle(
            installation.path / "bundle.json",
            ignored_files=(
                frozenset({".onnxvoice-install.json"})
                if installation.kind == "bundle"
                else frozenset()
            ),
        )
        if installation.sample_rate != 24000:
            raise RuntimeContractError("Kitten2 installation sample_rate must be 24000")
        runtime = installation.metadata.get("runtime")
        if (
            not isinstance(runtime, Mapping)
            or runtime.get("layout") != "kitten2-onnx-bundle-v1"
            or runtime.get("sample_rate") != 24000
        ):
            raise RuntimeContractError(
                "Kitten2 runtime requires layout 'kitten2-onnx-bundle-v1' and sample_rate=24000"
            )
        self.bundle = bundle
        self.manifest = bundle.manifest
        self._decode_presets = MappingProxyType(
            {
                name: MappingProxyType(dict(preset))
                for name, preset in self.manifest["decode_presets"].items()
            }
        )
        self._sessions: dict[str, OnnxSession] = {}
        self._language_model: Kitten2LanguageModel | None = None
        self._decoder: Kitten2Decoder | None = None
        self._prepared_voices: dict[str, PreparedVoice] = {}
        self.token_map = TokenMap.from_config({"token_map": self.manifest["token_map"]})
        self.frontend = Kitten2Frontend(bundle.root, self.manifest, self.token_map)
        lm = self.manifest["lm"]
        speaker = self.manifest["speaker"]
        decoder = self.manifest["decoder"]
        self._component_paths = {
            "lm": lm["model"],
            "speaker": speaker["model"],
            "flow": decoder["flow"],
            "mel2wav": decoder["mel2wav"],
        }
        self._component_metadata = {
            "lm": self._read_metadata(lm["metadata"]),
            "speaker": self._read_metadata(speaker["metadata"]),
            "flow": self._read_metadata(decoder["flow_metadata"]),
            "mel2wav": self._read_metadata(decoder["mel2wav_metadata"]),
        }

    def _read_metadata(self, relative_path: str) -> dict[str, Any]:
        return json.loads((self.bundle.root / relative_path).read_text(encoding="utf-8"))

    @property
    def voices(self) -> tuple[str, ...]:
        return tuple(sorted(self.manifest["voices"]))

    @property
    def default_voice(self) -> str:
        return self.manifest["default_voice"]

    @property
    def default_preset(self) -> str:
        return self.manifest["default_preset"]

    @property
    def decode_presets(self) -> Mapping[str, Mapping[str, Any]]:
        return self._decode_presets

    @property
    def sample_rate(self) -> int:
        return int(self.manifest["sample_rate"])

    def infer(self, token_ids: Sequence[int], **kwargs: Any) -> InferenceResult:
        raise CapabilityError("Kitten2 is a text-level multi-stage runtime; use infer_text()")

    def infer_text(
        self,
        text: str,
        *,
        voice: str | None = None,
        preset: str | None = None,
        seed: int | None = None,
        max_new_tokens: int = 512,
    ) -> InferenceResult:
        if max_new_tokens < 1:
            raise ValueError("max_new_tokens must be positive")
        resolved_voice = voice or self.default_voice
        resolved_preset = preset or self.default_preset
        if resolved_preset not in self.decode_presets:
            available = ", ".join(sorted(self.decode_presets))
            raise ValueError(f"Unknown decode preset {resolved_preset!r}; available presets: {available}")
        prepared = self._prepared_voices.get(resolved_voice)
        if prepared is None:
            prepared = self.frontend.prepare_voice(resolved_voice)
            self._prepared_voices[resolved_voice] = prepared
        _, text_ids = self.frontend.encode_text(text)
        reference_prefix = (
            prepared.reference_prefix
            if self.manifest["generation"].get("use_reference_prompt", True)
            else None
        )
        prompt_ids = build_generation_prompt(
            self.token_map, text_ids, reference_prefix=reference_prefix
        )
        generated = self._get_language_model().generate(
            prompt_ids,
            token_map=self.token_map,
            max_new_tokens=max_new_tokens,
            speaker_embedding=prepared.speaker_embedding,
            settings=sampling_settings_from_config(self.manifest, resolved_preset),
            seed=seed,
        )
        if not generated.s3_ids:
            raise RuntimeContractError("LM generation produced no valid S3 speech tokens")
        waveform = self._get_decoder().decode(generated.s3_ids, prepared.conditioning, seed=seed)
        if waveform is None:
            raise RuntimeContractError("S3 decoder produced no waveform")
        audio = np.asarray(waveform, dtype=np.float32)
        if audio.ndim == 2 and audio.shape[0] == 1:
            audio = audio[0]
        if audio.ndim != 1:
            raise RuntimeContractError(f"decoder waveform must be mono, found shape {audio.shape}")
        if not np.all(np.isfinite(audio)):
            raise RuntimeContractError("decoder waveform contains NaN or infinite samples")
        return InferenceResult(
            audio=audio,
            sample_rate=self.sample_rate,
            metadata={
                "system": self.system,
                "voice": resolved_voice,
                "preset": resolved_preset,
                "seed": seed,
                "max_new_tokens": max_new_tokens,
                "layout": "kitten2-onnx-bundle-v1",
            },
        )

    def _get_language_model(self) -> Kitten2LanguageModel:
        if self._language_model is None:
            self._validate_lm_runtime_provider()
            self._language_model = Kitten2LanguageModel(
                self._get_session("lm"),
                self._get_session("speaker"),
                self._component_metadata["lm"],
                self.bundle.root / self.manifest["lm"]["embeddings"],
            )
        return self._language_model

    def _validate_lm_runtime_provider(self) -> None:
        runtime_provider = self._component_metadata["lm"].get("runtime_provider")
        if runtime_provider not in {"CPUExecutionProvider", "CUDAExecutionProvider"}:
            raise RuntimeContractError(
                f"Unsupported runtime provider in LM metadata: {runtime_provider!r}"
            )
        if (
            runtime_provider == "CUDAExecutionProvider"
            and runtime_provider not in resolve_providers(self.providers)
        ):
            raise CapabilityError("Kitten2 LM profile requires CUDAExecutionProvider")

    def _get_decoder(self) -> Kitten2Decoder:
        if self._decoder is None:
            self._decoder = Kitten2Decoder(
                self._get_session("flow"),
                self._get_session("mel2wav"),
                self._component_metadata["flow"],
                self._component_metadata["mel2wav"],
            )
        return self._decoder

    def _get_session(self, component: str) -> OnnxSession:
        session = self._sessions.get(component)
        if session is None:
            session_options = self.session_options
            if component == "lm":
                session_options = self._lm_session_options()
            session = OnnxSession(
                self.bundle.root / self._component_paths[component],
                component=component,
                providers=self.providers,
                provider_options=self.provider_options,
                session_options=session_options,
            )
            self._sessions[component] = session
        return session

    def _lm_session_options(self) -> Any | None:
        metadata = self._component_metadata["lm"]
        if metadata.get("runtime_memory_profile", "default") != "low-memory-exact-ternary":
            return self.session_options
        if metadata.get("runtime_provider") != "CPUExecutionProvider":
            raise CapabilityError("Low-memory exact-ternary Kitten2 LM requires CPUExecutionProvider")
        selected = resolve_providers(self.providers)
        if selected != ("CPUExecutionProvider",):
            raise CapabilityError("Low-memory exact-ternary Kitten2 LM requires CPU only")
        if self.session_options is None:
            return _cpu_session_options("low-memory-exact-ternary")
        import onnxruntime as ort

        expected = {
            "intra_op_num_threads": 1,
            "inter_op_num_threads": 1,
            "execution_mode": ort.ExecutionMode.ORT_SEQUENTIAL,
            "enable_mem_pattern": False,
            "enable_cpu_mem_arena": False,
            "graph_optimization_level": ort.GraphOptimizationLevel.ORT_DISABLE_ALL,
        }
        for name, value in expected.items():
            if getattr(self.session_options, name, None) != value:
                raise RuntimeContractError(
                    f"session_options.{name} conflicts with the low-memory exact-ternary profile"
                )
        return self.session_options

    def close(self) -> None:
        self._language_model = None
        self._decoder = None
        self._prepared_voices.clear()
        super().close()
