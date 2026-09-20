"""Pocket ONNX bundle adapter with explicit v2 graph contracts."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from ..errors import RuntimeContractError
from ..runtime import OnnxSession
from ..types import InferenceResult, Installation
from .base import SystemAdapter

POCKET_REQUIRED_METADATA_FIELDS = frozenset(
    {
        "schema_version",
        "sample_rate",
        "samples_per_frame",
        "latent_dim",
        "conditioning_dim",
        "flow_lm_state_manifest",
        "mimi_state_manifest",
        "insert_bos_before_voice",
        "bos_before_voice_file",
    }
)
REQUIRED_FLOW_STATE_FIELDS = ("flow_lm_state_manifest", "mimi_state_manifest")
_FLOW_MAIN_STATE_ROLES = {"flow_lm_main", "mimi_decoder"}
_SUPPORTED_DTYPES = {
    "bool",
    "float16",
    "float32",
    "float64",
    "int8",
    "int16",
    "int32",
    "int64",
    "uint8",
}


@dataclass(frozen=True, slots=True)
class StateSpec:
    """One recurrent state tensor declared by a Pocket bundle manifest."""

    index: int
    input_name: str
    output_name: str
    shape: tuple[int, ...]
    dtype: np.dtype
    fill: str


def parse_state_manifest(raw: Any, *, name: str) -> tuple[StateSpec, ...]:
    """Parse the v2 list-form state manifest into typed state specifications."""
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes, Mapping)):
        raise RuntimeContractError(f"{name} state manifest must be a sequence")
    specs: list[StateSpec] = []
    indexes: set[int] = set()
    input_names: set[str] = set()
    output_names: set[str] = set()
    for position, entry in enumerate(raw):
        if not isinstance(entry, Mapping):
            raise RuntimeContractError(f"{name} state manifest entry {position} must be an object")
        index = entry.get("index")
        if not isinstance(index, int) or isinstance(index, bool) or index < 0:
            raise RuntimeContractError(f"{name} state entry {position} has invalid index")
        if index in indexes:
            raise RuntimeContractError(f"{name} state manifest has duplicate index {index}")
        input_name = entry.get("input_name")
        output_name = entry.get("output_name")
        if not isinstance(input_name, str) or not input_name:
            raise RuntimeContractError(f"{name} state entry {index} has invalid input_name")
        if not isinstance(output_name, str) or not output_name:
            raise RuntimeContractError(f"{name} state entry {index} has invalid output_name")
        if input_name in input_names:
            raise RuntimeContractError(
                f"{name} state manifest has duplicate input_name {input_name!r}"
            )
        if output_name in output_names:
            raise RuntimeContractError(
                f"{name} state manifest has duplicate output_name {output_name!r}"
            )
        shape = entry.get("shape")
        if not isinstance(shape, list) or any(
            not isinstance(dimension, int) or isinstance(dimension, bool) or dimension < 0
            for dimension in shape
        ):
            raise RuntimeContractError(f"{name} state entry {index} has invalid shape")
        dtype_name = entry.get("dtype")
        if not isinstance(dtype_name, str) or dtype_name not in _SUPPORTED_DTYPES:
            raise RuntimeContractError(
                f"{name} state entry {index} has unsupported dtype {dtype_name!r}"
            )
        fill = entry.get("fill")
        if fill not in {"zeros", "ones", "nan", "empty"}:
            raise RuntimeContractError(f"{name} state entry {index} has unsupported fill {fill!r}")
        dtype = np.dtype(dtype_name)
        if fill == "nan" and not np.issubdtype(dtype, np.floating):
            raise RuntimeContractError(
                f"{name} state entry {index} uses nan with non-floating dtype"
            )
        indexes.add(index)
        input_names.add(input_name)
        output_names.add(output_name)
        specs.append(StateSpec(index, input_name, output_name, tuple(shape), dtype, fill))
    specs.sort(key=lambda spec: spec.index)
    if [spec.index for spec in specs] != list(range(len(specs))):
        raise RuntimeContractError(f"{name} state indexes must be contiguous and start at zero")
    return tuple(specs)


def initialize_state(specs: Sequence[StateSpec]) -> dict[str, np.ndarray]:
    """Create state inputs using each manifest entry's declared fill semantics."""
    state: dict[str, np.ndarray] = {}
    for spec in specs:
        if spec.fill == "zeros":
            value = np.zeros(spec.shape, dtype=spec.dtype)
        elif spec.fill == "ones":
            value = np.ones(spec.shape, dtype=spec.dtype)
        elif spec.fill == "nan":
            value = np.full(spec.shape, np.nan, dtype=spec.dtype)
        else:
            value = np.empty(spec.shape, dtype=spec.dtype)
        state[spec.input_name] = value
    return state


def update_state_from_named_outputs(
    state: dict[str, np.ndarray],
    output_by_name: Mapping[str, Any],
    specs: Sequence[StateSpec],
) -> None:
    """Update recurrent state by manifest output names, never positional offsets."""
    for spec in specs:
        if spec.output_name not in output_by_name:
            raise RuntimeContractError(f"Missing state output {spec.output_name!r}")
        state[spec.input_name] = np.asarray(output_by_name[spec.output_name])


@dataclass(frozen=True, slots=True, init=False)
class PocketVoiceState:
    """Reusable voice embedding sequence produced by the Mimi encoder."""

    embeddings: np.ndarray
    sample_rate: int
    metadata: Mapping[str, Any]
    values: Mapping[str, np.ndarray]

    def __init__(
        self,
        embeddings: np.ndarray | None = None,
        sample_rate: int = 24000,
        metadata: Mapping[str, Any] | None = None,
        *,
        values: Mapping[str, np.ndarray] | None = None,
    ) -> None:
        legacy_values = dict(values or {})
        if embeddings is None:
            if not legacy_values:
                raise TypeError("PocketVoiceState requires embeddings")
            embeddings = np.asarray(next(iter(legacy_values.values())), dtype=np.float32)
        object.__setattr__(self, "embeddings", np.asarray(embeddings, dtype=np.float32))
        object.__setattr__(
            self, "sample_rate", int((metadata or {}).get("sample_rate", sample_rate))
        )
        object.__setattr__(self, "metadata", dict(metadata or {}))
        object.__setattr__(self, "values", legacy_values or {"embeddings": self.embeddings})


class PocketAdapter(SystemAdapter):
    """Pocket ONNX bundle adapter with lazy, contract-validated sessions."""

    system = "pocket"
    DEFAULT_MAX_FRAMES = 200

    def __init__(self, installation: Installation, **kwargs: Any) -> None:
        super().__init__(installation, **kwargs)
        self._sessions: dict[str, OnnxSession] = {}
        self._bundle_metadata: dict[str, Any] | None = None
        self._bos_conditioning: np.ndarray | None = None
        self._flow_specs: tuple[StateSpec, ...] = ()
        self._mimi_specs: tuple[StateSpec, ...] = ()
        self._validated_contracts: set[str] = set()
        self._validated = False

    def _validate_bundle_metadata(self) -> dict[str, Any]:
        if self._bundle_metadata is not None:
            return self._bundle_metadata
        try:
            metadata_path = self.installation.artifact("bundle_metadata").path
        except KeyError as err:
            runtime_metadata = self.installation.metadata.get("runtime") or {}
            if not runtime_metadata:
                raise RuntimeContractError(
                    "Pocket bundle metadata not found. Provide bundle_metadata artifact or runtime metadata."
                ) from err
            self._bundle_metadata = dict(runtime_metadata)
            return self._bundle_metadata
        try:
            bundle_data = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeContractError(
                f"Could not read Pocket bundle metadata {metadata_path}: {exc}"
            ) from exc
        if not isinstance(bundle_data, dict):
            raise RuntimeContractError("Pocket bundle metadata must be a JSON object")
        self._bundle_metadata = {**bundle_data, **self.installation.metadata}
        return self._bundle_metadata

    @staticmethod
    def _parse_manifest(metadata: Mapping[str, Any], key: str, name: str) -> tuple[StateSpec, ...]:
        raw = metadata.get(key)
        if (
            raw in (None, {}, "")
            or isinstance(raw, (Mapping, str, bytes))
            or not isinstance(raw, Sequence)
        ):
            # Accept legacy runtime metadata only at the adapter boundary; the public parser is strict.
            return ()
        return parse_state_manifest(raw, name=name)

    def _ensure_validated(self) -> dict[str, Any]:
        if self._validated:
            assert self._bundle_metadata is not None
            return self._bundle_metadata
        metadata = self._validate_bundle_metadata()
        missing = sorted(
            field for field in POCKET_REQUIRED_METADATA_FIELDS if field not in metadata
        )
        if missing:
            raise RuntimeContractError(
                f"Pocket bundle metadata is missing required fields: {', '.join(missing)}"
            )
        try:
            self._flow_specs = self._parse_manifest(metadata, "flow_lm_state_manifest", "flow_lm")
            self._mimi_specs = self._parse_manifest(metadata, "mimi_state_manifest", "mimi")
        except RuntimeContractError:
            raise
        self._validated = True
        return metadata

    @staticmethod
    def _names(session: Any, attribute: str) -> tuple[str, ...]:
        value = getattr(session, attribute, ())
        if isinstance(value, (tuple, list)):
            return tuple(str(name) for name in value)
        return ()

    @staticmethod
    def _named_outputs(session: Any, outputs: Sequence[Any]) -> dict[str, np.ndarray]:
        names = PocketAdapter._names(session, "output_names")
        if len(names) != len(outputs):
            raise RuntimeContractError(
                f"Session returned {len(outputs)} outputs but exposes {len(names)} output names"
            )
        return {name: np.asarray(value) for name, value in zip(names, outputs, strict=True)}

    @staticmethod
    def _find_output(
        outputs: Mapping[str, np.ndarray], candidates: Sequence[str], *, label: str
    ) -> str:
        for candidate in candidates:
            if candidate in outputs:
                return candidate
        lowered = {name.casefold(): name for name in outputs}
        for candidate in candidates:
            if candidate.casefold() in lowered:
                return lowered[candidate.casefold()]
        if len(outputs) == 1:
            return next(iter(outputs))
        raise RuntimeContractError(
            f"{label} output is not identifiable; available outputs: {', '.join(outputs) or 'none'}"
        )

    def _validate_state_contract(
        self,
        role: str,
        input_names: tuple[str, ...],
        output_names: tuple[str, ...],
        specs: Sequence[StateSpec],
    ) -> None:
        missing_inputs = [spec.input_name for spec in specs if spec.input_name not in input_names]
        missing_outputs = [
            spec.output_name for spec in specs if spec.output_name not in output_names
        ]
        if missing_inputs or missing_outputs:
            details = []
            if missing_inputs:
                details.append(f"missing inputs: {', '.join(missing_inputs)}")
            if missing_outputs:
                details.append(f"missing outputs: {', '.join(missing_outputs)}")
            raise RuntimeContractError(f"{role} state contract mismatch ({'; '.join(details)})")

    def _validate_session_contract(self, role: str, session: OnnxSession) -> None:
        if role in self._validated_contracts:
            return
        input_names = self._names(session, "input_names")
        output_names = self._names(session, "output_names")
        if not input_names or not output_names:
            # Minimal fake sessions used by lifecycle tests do not expose graph metadata.
            return
        inputs = set(input_names)
        outputs = set(output_names)
        if role == "text_conditioner":
            if "token_ids" not in inputs:
                raise RuntimeContractError("text_conditioner graph must accept token_ids")
            self._find_output(
                {name: np.empty(0) for name in output_names},
                ("embeddings", "text_embeddings", "conditioning"),
                label="text conditioner embeddings",
            )
        elif role == "mimi_encoder":
            if "audio" not in inputs:
                raise RuntimeContractError("mimi_encoder graph must accept audio")
            self._find_output(
                {name: np.empty(0) for name in output_names},
                ("embeddings", "voice_embeddings", "encoded"),
                label="Mimi encoder embeddings",
            )
        elif role == "flow_lm_main":
            missing = {"sequence", "text_embeddings"} - inputs
            if missing:
                raise RuntimeContractError(
                    f"flow_lm_main graph is missing inputs: {', '.join(sorted(missing))}"
                )
            self._validate_state_contract(role, input_names, output_names, self._flow_specs)
            non_state_outputs = outputs - {spec.output_name for spec in self._flow_specs}
            self._find_output(
                {name: np.empty(0) for name in non_state_outputs},
                ("conditioning", "cond", "c", "latent"),
                label="Flow-LM main conditioning",
            )
        elif role == "flow_lm_flow":
            missing = {"c", "s", "t", "x"} - inputs
            if missing:
                raise RuntimeContractError(
                    f"flow_lm_flow graph is missing inputs: {', '.join(sorted(missing))}"
                )
            self._find_output(
                {name: np.empty(0) for name in output_names},
                ("velocity", "flow", "v", "output"),
                label="flow velocity",
            )
        elif role == "mimi_decoder":
            if "latent" not in inputs:
                raise RuntimeContractError("mimi_decoder graph must accept latent")
            self._validate_state_contract(role, input_names, output_names, self._mimi_specs)
            non_state_outputs = outputs - {spec.output_name for spec in self._mimi_specs}
            self._find_output(
                {name: np.empty(0) for name in non_state_outputs},
                ("audio", "waveform", "output"),
                label="Mimi decoder audio",
            )
        self._validated_contracts.add(role)

    def _get_session(self, role: str) -> OnnxSession:
        if role in self._sessions:
            return self._sessions[role]
        try:
            artifact = self.installation.artifact(role)
        except KeyError as err:
            raise RuntimeContractError(
                f"Pocket bundle is missing required artifact: {role!r}"
            ) from err
        session = OnnxSession(
            artifact.path,
            component=role,
            providers=self.providers,
            provider_options=self.provider_options,
            session_options=self.session_options,
        )
        self._sessions[role] = session
        try:
            self._validate_session_contract(role, session)
        except Exception:
            self._sessions.pop(role, None)
            session.close()
            raise
        return session

    def _load_bos_conditioning(self) -> np.ndarray:
        if self._bos_conditioning is not None:
            return self._bos_conditioning
        try:
            self._bos_conditioning = np.asarray(
                np.load(self.installation.artifact("bos_conditioning").path), dtype=np.float32
            )
        except (KeyError, OSError, ValueError) as exc:
            raise RuntimeContractError(f"Could not load BOS conditioning: {exc}") from exc
        return self._bos_conditioning

    @staticmethod
    def _normalize_embeddings(value: Any, *, label: str) -> np.ndarray:
        embeddings = np.asarray(value, dtype=np.float32)
        if embeddings.ndim == 2:
            embeddings = embeddings[None, ...]
        if embeddings.ndim != 3 or embeddings.shape[0] != 1:
            raise RuntimeContractError(
                f"{label} must have shape [1, sequence, dimension], got {embeddings.shape}"
            )
        return embeddings

    def prepare_voice(self, audio: np.ndarray, *, sample_rate: int) -> PocketVoiceState:
        metadata = self._ensure_validated()
        expected_rate = metadata.get("sample_rate")
        if sample_rate != expected_rate:
            raise RuntimeContractError(
                f"Pocket reference audio sample rate must be {expected_rate}, got {sample_rate}"
            )
        audio_array = np.asarray(audio, dtype=np.float32)
        if audio_array.ndim == 1:
            audio_array = audio_array.reshape(1, 1, -1)
        elif audio_array.ndim == 3 and audio_array.shape[0] == 1 and audio_array.shape[1] == 1:
            pass
        else:
            raise RuntimeContractError(
                f"Pocket reference audio must have shape [1, 1, samples], got {audio_array.shape}"
            )
        if audio_array.shape[-1] == 0 or not np.isfinite(audio_array).all():
            raise RuntimeContractError("Pocket reference audio must be non-empty and finite")
        encoder = self._get_session("mimi_encoder")
        try:
            outputs = encoder.run({"audio": audio_array})
            named = (
                self._named_outputs(encoder, outputs)
                if self._names(encoder, "output_names")
                else {"encoded": np.asarray(outputs[0])}
            )
            output_name = self._find_output(
                named,
                ("embeddings", "voice_embeddings", "encoded"),
                label="Mimi encoder embeddings",
            )
            embeddings = self._normalize_embeddings(
                named[output_name], label="Mimi encoder embeddings"
            )
            if metadata.get("insert_bos_before_voice"):
                bos = self._normalize_embeddings(
                    self._load_bos_conditioning(), label="BOS conditioning"
                )
                embeddings = np.concatenate((bos, embeddings), axis=1)
        except RuntimeContractError:
            raise
        except Exception as exc:
            raise RuntimeContractError(f"Mimi encoder failed: {exc}") from exc
        return PocketVoiceState(
            embeddings=np.array(embeddings, dtype=np.float32, copy=True),
            sample_rate=sample_rate,
            metadata={
                "sample_rate": sample_rate,
                "encoder_output": output_name,
                "frames": embeddings.shape[1],
            },
            values={output_name: np.array(embeddings, dtype=np.float32, copy=True)},
        )

    def _encode_text(self, session: OnnxSession, token_ids: Sequence[int]) -> np.ndarray:
        tokens = np.asarray([list(token_ids)], dtype=np.int64)
        if tokens.shape[1] == 0:
            raise RuntimeContractError("Pocket token_ids must not be empty")
        outputs = session.run({"token_ids": tokens})
        named = (
            self._named_outputs(session, outputs)
            if self._names(session, "output_names")
            else {"embeddings": outputs[0]}
        )
        output_name = self._find_output(
            named,
            ("embeddings", "text_embeddings", "conditioning"),
            label="text conditioner embeddings",
        )
        return self._normalize_embeddings(named[output_name], label="Text conditioner embeddings")

    @staticmethod
    def _outputs(session: OnnxSession, values: Sequence[Any]) -> dict[str, np.ndarray]:
        if PocketAdapter._names(session, "output_names"):
            return PocketAdapter._named_outputs(session, values)
        return {f"output_{index}": np.asarray(value) for index, value in enumerate(values)}

    def _condition_prefix(
        self,
        session: OnnxSession,
        state: dict[str, np.ndarray],
        voice_embeddings: np.ndarray,
        text_embeddings: np.ndarray,
        latent_dim: int,
    ) -> None:
        empty_sequence = np.empty((1, 0, latent_dim), dtype=np.float32)
        for embeddings in (voice_embeddings, text_embeddings):
            inputs = {"sequence": empty_sequence, "text_embeddings": embeddings, **state}
            outputs = self._outputs(session, session.run(inputs))
            update_state_from_named_outputs(
                state, outputs, self._flow_specs
            ) if self._flow_specs else None

    def _generate_latents(
        self,
        main: OnnxSession,
        flow: OnnxSession,
        state: dict[str, np.ndarray],
        *,
        temperature: float,
        lsd_steps: int,
        max_frames: int,
        frames_after_eos: int | None,
        latent_dim: int,
        conditioning_dim: int,
    ) -> tuple[np.ndarray, bool, int]:
        if lsd_steps < 1:
            raise RuntimeContractError("lsd_steps must be at least 1")
        frames: list[np.ndarray] = []
        previous_latent: np.ndarray | None = None
        eos_detected = False
        post_eos = 0
        empty_text = np.empty((1, 0, conditioning_dim), dtype=np.float32)
        for _ in range(max_frames):
            sequence = (
                previous_latent
                if previous_latent is not None
                else np.full((1, 1, latent_dim), np.nan, dtype=np.float32)
            )
            outputs = self._outputs(
                main, main.run({"sequence": sequence, "text_embeddings": empty_text, **state})
            )
            if self._flow_specs:
                update_state_from_named_outputs(state, outputs, self._flow_specs)
            eos_name = next(
                (name for name in ("eos", "eos_logit", "eos_probability") if name in outputs), None
            )
            eos_now = eos_name is not None and bool(np.any(outputs[eos_name] > 0.5))
            non_state = {
                name: value
                for name, value in outputs.items()
                if name not in {spec.output_name for spec in self._flow_specs} and name != eos_name
            }
            conditioning_name = self._find_output(
                non_state,
                ("conditioning", "cond", "c", "latent"),
                label="Flow-LM main conditioning",
            )
            conditioning = np.asarray(non_state[conditioning_name], dtype=np.float32)
            if conditioning.ndim == 2:
                conditioning = conditioning[:, None, :]
            x = np.random.standard_normal((1, 1, latent_dim)).astype(np.float32) * np.float32(
                temperature
            )
            for step in range(lsd_steps):
                t = np.float32(1.0 - step / lsd_steps)
                s = np.float32(1.0 - (step + 1) / lsd_steps)
                flow_outputs = self._outputs(
                    flow,
                    flow.run(
                        {
                            "c": conditioning,
                            "s": np.asarray([s], dtype=np.float32),
                            "t": np.asarray([t], dtype=np.float32),
                            "x": x,
                        }
                    ),
                )
                velocity_name = self._find_output(
                    flow_outputs,
                    ("velocity", "flow", "v", "spectral", "output"),
                    label="flow velocity",
                )
                x = x + (t - s) * np.asarray(flow_outputs[velocity_name], dtype=np.float32)
            latent = np.asarray(x, dtype=np.float32)
            frames.append(latent)
            previous_latent = latent
            if eos_now:
                eos_detected = True
                post_eos += 1
                if frames_after_eos is None or post_eos >= frames_after_eos:
                    break
        if not frames:
            raise RuntimeContractError("Pocket Flow-LM generated no latent frames")
        return np.concatenate(frames, axis=1), eos_detected, len(frames)

    def _decode_latents(
        self,
        decoder: OnnxSession,
        latents: np.ndarray,
        state: dict[str, np.ndarray],
        chunk_frames: int,
    ) -> np.ndarray:
        parts: list[np.ndarray] = []
        for start in range(0, latents.shape[1], chunk_frames):
            chunk = latents[:, start : start + chunk_frames, ...]
            outputs = self._outputs(decoder, decoder.run({"latent": chunk, **state}))
            if self._mimi_specs:
                update_state_from_named_outputs(state, outputs, self._mimi_specs)
            non_state = {
                name: value
                for name, value in outputs.items()
                if name not in {spec.output_name for spec in self._mimi_specs}
            }
            audio_name = self._find_output(
                non_state, ("audio", "waveform", "output"), label="Mimi decoder audio"
            )
            parts.append(np.asarray(non_state[audio_name], dtype=np.float32))
        return np.concatenate(parts, axis=-1)

    def infer(
        self,
        token_ids: Sequence[int],
        *,
        voice_state: PocketVoiceState | None = None,
        temperature: float = 0.7,
        lsd_steps: int = 1,
        max_frames: int | None = None,
        frames_after_eos: int | None = None,
        **kwargs: Any,
    ) -> InferenceResult:
        metadata = self._ensure_validated()
        if temperature < 0:
            raise RuntimeContractError("temperature must be non-negative")
        if frames_after_eos is not None and frames_after_eos < 0:
            raise RuntimeContractError("frames_after_eos must be non-negative")
        frame_limit = (
            max_frames
            if max_frames is not None
            else int(metadata.get("max_frames", self.DEFAULT_MAX_FRAMES))
        )
        if frame_limit <= 0:
            raise RuntimeContractError("max_frames must be a positive frame count")
        text_conditioner = self._get_session("text_conditioner")
        main = self._get_session("flow_lm_main")
        flow = self._get_session("flow_lm_flow")
        decoder = self._get_session("mimi_decoder")
        text_embeddings = self._encode_text(text_conditioner, token_ids)
        flow_state = initialize_state(self._flow_specs)
        mimi_state = initialize_state(self._mimi_specs)
        if voice_state is None:
            voice_embeddings = self._normalize_embeddings(
                self._load_bos_conditioning(), label="BOS conditioning"
            )
        else:
            if voice_state.sample_rate != metadata.get("sample_rate"):
                raise RuntimeContractError(
                    "Pocket voice state sample rate does not match the bundle"
                )
            voice_embeddings = self._normalize_embeddings(
                voice_state.embeddings, label="Pocket voice embeddings"
            )
        self._condition_prefix(
            main, flow_state, voice_embeddings, text_embeddings, int(metadata["latent_dim"])
        )
        latents, eos_detected, frame_count = self._generate_latents(
            main,
            flow,
            flow_state,
            temperature=temperature,
            lsd_steps=lsd_steps,
            max_frames=frame_limit,
            frames_after_eos=frames_after_eos,
            latent_dim=int(metadata["latent_dim"]),
            conditioning_dim=int(metadata["conditioning_dim"]),
        )
        chunk_value = metadata.get("decoder_chunk_frames")
        chunk_frames = int(chunk_value if chunk_value is not None else latents.shape[1])
        if chunk_frames <= 0:
            raise RuntimeContractError("decoder_chunk_frames must be positive")
        audio = self._decode_latents(decoder, latents, mimi_state, chunk_frames)
        audio = np.squeeze(audio)
        if audio.ndim != 1:
            if audio.ndim == 2 and 1 in audio.shape:
                audio = audio.reshape(-1)
            else:
                raise RuntimeContractError("Pocket audio output must be one-dimensional")
        audio = np.asarray(audio, dtype=np.float32)
        if not np.isfinite(audio).all():
            raise RuntimeContractError("Pocket decoder produced non-finite audio")
        return InferenceResult(
            audio=audio,
            sample_rate=int(metadata["sample_rate"]),
            metadata={
                "system": "pocket",
                "temperature": temperature,
                "lsd_steps": lsd_steps,
                "frames_generated": frame_count,
                "eos_detected": eos_detected,
            },
        )

    def _initialize_state(
        self,
        manifest: Mapping[str, Any],
        *,
        batch_size: int = 1,
    ) -> dict[str, np.ndarray]:
        """Compatibility helper for legacy callers; v2 runtime uses typed manifests."""
        if isinstance(manifest, Sequence) and not isinstance(manifest, (str, bytes, Mapping)):
            return initialize_state(parse_state_manifest(manifest, name="state"))
        state: dict[str, np.ndarray] = {}
        for name, spec in manifest.items():
            if isinstance(spec, Mapping):
                shape = tuple(spec.get("shape", (1,)))
                if shape and shape[0] is None:
                    shape = (batch_size, *shape[1:])
                dtype = spec.get("dtype", "float32")
            elif isinstance(spec, (list, tuple)):
                shape = tuple(spec)
                dtype = "float32"
            else:
                continue
            state[name] = np.zeros(shape, dtype=dtype)
        return state

    def _owned_sessions(self) -> tuple[Any, ...]:
        return tuple(self._sessions.values())

    def close(self) -> None:
        for session in tuple(self._sessions.values()):
            session.close()
        self._sessions.clear()
        self._bundle_metadata = None
        self._bos_conditioning = None
        self._flow_specs = ()
        self._mimi_specs = ()
        self._validated_contracts.clear()
        self._validated = False
