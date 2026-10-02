from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np

from ..errors import CapabilityError, RuntimeContractError
from ..types import InferenceResult, Installation
from .base import SystemAdapter
from .kokoro_split import SplitKokoroRuntime

CLONING_LAYOUT = "cloning-onnx-v1"
CLONING_MODEL_COMPONENTS = (
    "reference_wavlm",
    "reference_encoders",
    "reference_mapper",
    "prosody",
    "curves",
    "decoder",
)
FINGERPRINT_COMPONENTS = (*CLONING_MODEL_COMPONENTS, "source_params")
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")

N_MELS = 80
N_FFT = 2048
WIN_LENGTH = 1200
HOP_LENGTH = 300
MEL_EPSILON = 1.0e-5
MEL_MEAN = -4.0
MEL_STD = 4.0
MEL_SAMPLE_RATE = 16000
REFERENCE_PAD_SAMPLES = 5000
REFERENCE_MIN_SECONDS = 3.0
REFERENCE_MAX_SECONDS = 30.0
REFERENCE_SPEECH_RMS_MINIMUM = 1.0e-4
REFERENCE_CLIPPING_THRESHOLD = 0.999
REFERENCE_CLIPPING_FRACTION_MAXIMUM = 0.001
AUDIO_DURATION_TOLERANCE_SECONDS = 0.02


def _mel_filterbank() -> np.ndarray:
    frequencies = np.linspace(0.0, MEL_SAMPLE_RATE / 2, N_FFT // 2 + 1, dtype=np.float32)
    mel_min = 2595.0 * np.log10(1.0 + 0.0 / 700.0)
    mel_max = 2595.0 * np.log10(1.0 + (MEL_SAMPLE_RATE / 2) / 700.0)
    mel_points = np.linspace(mel_min, mel_max, N_MELS + 2, dtype=np.float32)
    hz_points = (700.0 * (10.0 ** (mel_points / 2595.0) - 1.0)).astype(np.float32)
    differences = hz_points[1:] - hz_points[:-1]
    slopes = hz_points[:, None] - frequencies[None, :]
    lower = -slopes[:-2] / differences[:-1, None]
    upper = slopes[2:] / differences[1:, None]
    return np.maximum(0.0, np.minimum(lower, upper)).T.astype(np.float32)


_MEL_FILTERBANK = _mel_filterbank()
_MEL_WINDOW = (
    0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(WIN_LENGTH, dtype=np.float64) / WIN_LENGTH)
).astype(np.float32)
_MEL_WINDOW = np.pad(_MEL_WINDOW, (N_FFT // 2 - WIN_LENGTH // 2, N_FFT // 2 - WIN_LENGTH // 2))


def _reference_mel(wave: np.ndarray) -> np.ndarray:
    values = np.asarray(wave, dtype=np.float32)
    centered = np.pad(values, (N_FFT // 2, N_FFT // 2), mode="reflect")
    frames = np.lib.stride_tricks.sliding_window_view(centered, N_FFT)[::HOP_LENGTH]
    spectrum = np.fft.rfft(frames * _MEL_WINDOW[None, :], axis=1)
    power = np.square(np.abs(spectrum)).astype(np.float32)
    mel = power @ _MEL_FILTERBANK
    normalized = (np.log(MEL_EPSILON + mel.T[None]) - MEL_MEAN) / MEL_STD
    return np.ascontiguousarray(normalized, dtype=np.float32)


def reference_mels(audio_24k: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return the mapper mel and padded/even encoder mel used by AkinVox."""
    wave = np.asarray(audio_24k, dtype=np.float32)
    mapper_mel = _reference_mel(wave)
    padded = np.pad(wave, (REFERENCE_PAD_SAMPLES, REFERENCE_PAD_SAMPLES))
    encoder_mel = _reference_mel(padded)
    even_frames = encoder_mel.shape[-1] - encoder_mel.shape[-1] % 2
    encoder_mel = np.ascontiguousarray(encoder_mel[:, :, :even_frames][:, None])
    return mapper_mel, encoder_mel


@dataclass(frozen=True, slots=True)
class KokoroReferenceState:
    """Validated, reusable AkinVox reference tensors for one exact model build."""

    style: np.ndarray
    memory: np.ndarray
    memory_mask: np.ndarray
    model_fingerprint: str
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        style = np.array(self.style, dtype=np.float32, order="C", copy=True)
        memory = np.array(self.memory, dtype=np.float32, order="C", copy=True)
        memory_mask = np.array(self.memory_mask, dtype=np.bool_, order="C", copy=True)

        if style.shape != (1, 256):
            raise RuntimeContractError("Kokoro reference style must have shape [1, 256]")
        if (
            memory.ndim != 3
            or memory.shape[0] != 1
            or memory.shape[1] == 0
            or memory.shape[2] != 192
        ):
            raise RuntimeContractError(
                "Kokoro reference memory must have shape [1, M, 192] with M > 0"
            )
        if memory_mask.shape != memory.shape[:2]:
            raise RuntimeContractError("Kokoro reference memory mask must have shape [1, M]")
        if not np.all(np.isfinite(style)) or not np.all(np.isfinite(memory)):
            raise RuntimeContractError(
                "Kokoro reference state tensors must contain only finite values"
            )
        if not isinstance(self.model_fingerprint, str) or not self.model_fingerprint.strip():
            raise RuntimeContractError(
                "Kokoro reference state requires a non-empty model fingerprint"
            )
        if not isinstance(self.metadata, Mapping):
            raise RuntimeContractError("Kokoro reference state metadata must be a mapping")

        style.setflags(write=False)
        memory.setflags(write=False)
        memory_mask.setflags(write=False)
        object.__setattr__(self, "style", style)
        object.__setattr__(self, "memory", memory)
        object.__setattr__(self, "memory_mask", memory_mask)
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))


def kokoro_model_fingerprint(installation: Installation) -> str:
    """Hash the exact six graph artifacts and source parameters, without using paths."""
    artifact_hashes = [
        [
            component,
            _artifact_sha256(
                installation,
                "metadata" if component == "source_params" else "model",
                component,
            ),
        ]
        for component in FINGERPRINT_COMPONENTS
    ]
    canonical = json.dumps(artifact_hashes, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(canonical.encode("ascii")).hexdigest()


def _artifact_sha256(installation: Installation, role: str, component: str) -> str:
    try:
        artifact = installation.require_artifact(role, component=component)
    except KeyError as exc:
        raise RuntimeContractError(
            f"Kokoro cloning installation is missing role={role!r}, component={component!r}"
        ) from exc
    digest = artifact.sha256
    if not isinstance(digest, str) or _SHA256_RE.fullmatch(digest) is None:
        raise RuntimeContractError(
            f"Kokoro cloning artifact {component!r} must have a lowercase SHA-256 digest"
        )
    return digest


class KokoroCloningRuntime(SplitKokoroRuntime):
    """ONNX enrollment and reference-conditioned synthesis for Kokoro cloning."""

    COMPONENTS = CLONING_MODEL_COMPONENTS

    def __init__(
        self,
        installation: Installation,
        *,
        session_factory: Callable[[Path, str], Any] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(installation, session_factory=session_factory, **kwargs)
        self.model_fingerprint = kokoro_model_fingerprint(installation)

    def _artifact(self, role: str, *, component: str | None = None) -> Path:
        try:
            return self.installation.artifact(role, component=component).path
        except KeyError as exc:
            detail = f"role={role!r}"
            if component is not None:
                detail += f", component={component!r}"
            raise CapabilityError(f"Kokoro cloning runtime is missing {detail}") from exc

    def _component_path(self, component: str) -> Path:
        return self._artifact("model", component=component)

    def _load_supporting_assets(self) -> None:
        for role in ("config", "bundle", "manifest"):
            try:
                config_path = self._artifact(role)
                break
            except CapabilityError:
                continue
        else:
            raise CapabilityError("Kokoro cloning runtime is missing its config/bundle/manifest")
        try:
            manifest = json.loads(config_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeContractError(
                f"Could not read Kokoro cloning manifest {config_path}: {exc}"
            ) from exc
        if not isinstance(manifest, dict):
            raise RuntimeContractError("Kokoro cloning manifest must contain an object")

        runtime = self.installation.metadata.get("runtime") or {}
        max_tokens = runtime.get("max_tokens", manifest.get("max_tokens"))
        if isinstance(max_tokens, int) and max_tokens > 0:
            self.max_tokens = max_tokens

        source_path = self._source_parameters_path()
        try:
            with np.load(source_path, allow_pickle=False) as source:
                self._source_weight = self._array_from_npz(source, ("weight", "source_weight"))
                self._source_bias = self._array_from_npz(source, ("bias", "source_bias"))
                self._window = self._array_from_npz(source, ("window",))
        except (OSError, ValueError, KeyError) as exc:
            raise RuntimeContractError(
                f"Invalid Kokoro cloning source parameters {source_path}: {exc}"
            ) from exc
        if self._source_weight.ndim != 2 or self._source_bias.ndim != 1:
            raise RuntimeContractError("Kokoro cloning source parameters have invalid shapes")
        if self._source_weight.shape[0] != self._source_bias.shape[0]:
            raise RuntimeContractError("Kokoro cloning source weight and bias shapes do not match")
        if self._window.ndim != 1 or self._window.size != 20:
            raise RuntimeContractError("Kokoro cloning source window must contain 20 values")

    def _source_parameters_path(self) -> Path:
        for role in ("metadata", "source_params"):
            matches = self.installation.artifacts_for(role=role, component="source_params")
            if matches:
                return matches[0].path
        for artifact in self.installation.artifacts:
            if (
                "source_params" in artifact.filename.lower()
                or "source-params" in artifact.filename.lower()
            ):
                return artifact.path
        raise CapabilityError("Kokoro cloning runtime is missing source parameters")

    def diagnostics(self) -> Any:
        return SystemAdapter.diagnostics(self)

    def prepare_reference(
        self,
        reference_token_ids: Sequence[int],
        *,
        audio_24k: np.ndarray,
        audio_16k: np.ndarray,
    ) -> KokoroReferenceState:
        tokens = self._validate_reference_tokens(reference_token_ids)
        wave24, wave16 = self._validate_reference_audio(audio_24k, audio_16k)
        mapper_mel, encoder_mel = reference_mels(wave24)
        reference_ids = np.asarray([[0, *tokens, 0]], dtype=np.int64)
        reference_lengths = np.asarray([len(tokens) + 2], dtype=np.int64)

        encoders = self._get_session("reference_encoders")
        self._require_inputs(encoders, {"mel"}, "reference_encoders")
        encoder_values = self._run(encoders, {"mel": encoder_mel})
        encoder_named = self._named_outputs(encoders, encoder_values)
        raw_sdec = self._output(encoder_named, ("raw_sdec",), 0)
        raw_spred = self._output(encoder_named, ("raw_spred",), 1)
        self._validate_observation(raw_sdec, (1, 128), "raw_sdec")
        self._validate_observation(raw_spred, (1, 128), "raw_spred")

        wavlm_session = self._get_session("reference_wavlm")
        self._require_inputs(wavlm_session, {"input_values"}, "reference_wavlm")
        wavlm_values = self._run(
            wavlm_session, {"input_values": np.ascontiguousarray(wave16[None, :])}
        )
        wavlm_named = self._named_outputs(wavlm_session, wavlm_values)
        wavlm = self._output(wavlm_named, ("wavlm", "embeddings"), 0)
        self._validate_observation(wavlm, (1, 512), "wavlm")
        wavlm_norm = np.linalg.norm(wavlm, axis=-1, keepdims=True)
        wavlm = np.asarray(wavlm / np.maximum(wavlm_norm, 1.0e-8), dtype=np.float32)

        mapper = self._get_session("reference_mapper")
        self._require_inputs(
            mapper,
            {
                "mel",
                "mel_lengths",
                "reference_ids",
                "reference_lengths",
                "wavlm",
                "raw_sdec",
                "raw_spred",
            },
            "reference_mapper",
        )
        mapper_inputs = {
            "mel": mapper_mel,
            "mel_lengths": np.asarray([mapper_mel.shape[-1]], dtype=np.int64),
            "reference_ids": reference_ids,
            "reference_lengths": reference_lengths,
            "wavlm": wavlm,
            "raw_sdec": np.asarray(raw_sdec, dtype=np.float32),
            "raw_spred": np.asarray(raw_spred, dtype=np.float32),
        }
        mapper_values = self._run(mapper, mapper_inputs)
        mapper_named = self._named_outputs(mapper, mapper_values)
        style = self._output(mapper_named, ("style",), 0)
        memory = self._output(mapper_named, ("reference_memory", "memory"), 1)
        memory_mask = self._output(mapper_named, ("reference_mask", "memory_mask"), 2)
        return KokoroReferenceState(
            style=style,
            memory=memory,
            memory_mask=memory_mask,
            model_fingerprint=self.model_fingerprint,
            metadata={
                "format": "akinvox-cloning-reference-v1",
                "sample_rate": 24000,
                "reference_seconds": len(wave24) / 24000.0,
            },
        )

    def infer(
        self,
        token_ids: Sequence[int],
        *,
        reference: KokoroReferenceState | None = None,
        speed: float = 1.0,
        seed: int = 1234,
    ) -> InferenceResult:
        if speed != 1.0:
            raise CapabilityError("Kokoro cloning-onnx-v1 does not support speed control")
        if not isinstance(reference, KokoroReferenceState):
            raise RuntimeContractError("Kokoro cloning inference requires a KokoroReferenceState")
        if reference.model_fingerprint != self.model_fingerprint:
            raise RuntimeContractError("Kokoro reference state belongs to a different model build")
        tokens = self._validate_text_tokens(token_ids)
        if reference.style.shape != (1, 256):
            raise RuntimeContractError("Kokoro reference style must have shape [1, 256]")
        style_acou = np.ascontiguousarray(reference.style[:, :128])
        style_dur = np.ascontiguousarray(reference.style[:, 128:])
        if style_acou.shape != (1, 128) or style_dur.shape != (1, 128):
            raise RuntimeContractError(
                "Kokoro cloning style split must be 128 acoustic and 128 duration values"
            )

        input_ids = np.asarray([[0, *tokens, 0]], dtype=np.int64)
        prosody = self._get_session("prosody")
        self._require_inputs(
            prosody,
            {"input_ids", "style_dur", "reference_memory", "reference_mask"},
            "prosody",
        )
        prosody_values = self._run(
            prosody,
            {
                "input_ids": input_ids,
                "style_dur": style_dur,
                "reference_memory": reference.memory,
                "reference_mask": reference.memory_mask,
            },
        )
        prosody_named = self._named_outputs(prosody, prosody_values)
        pred_dur = self._output(prosody_named, ("pred_dur", "duration"), 0)
        try:
            duration_values = np.asarray(pred_dur, dtype=np.float64).reshape(-1)
        except (TypeError, ValueError) as exc:
            raise RuntimeContractError("Kokoro cloning durations must be numeric") from exc
        if duration_values.size != input_ids.shape[1] or not np.all(np.isfinite(duration_values)):
            raise RuntimeContractError(
                "Kokoro cloning durations must be finite and match the input phones"
            )
        if np.any(duration_values < 1) or np.any(duration_values != np.floor(duration_values)):
            raise RuntimeContractError(
                "Kokoro cloning durations must be integers of at least one frame"
            )
        if np.any(duration_values > 4000) or int(np.sum(duration_values)) > 4000:
            raise RuntimeContractError("Kokoro cloning duration expansion exceeds 4000 frames")
        durations = duration_values.astype(np.int64)
        expanded_frames = int(np.sum(durations))
        if expanded_frames <= 0:
            raise RuntimeContractError("Kokoro cloning duration expansion must not be empty")

        d = np.asarray(self._output(prosody_named, ("d", "duration_embedding"), 1))
        t_en = np.asarray(self._output(prosody_named, ("t_en", "asr"), 2))
        if d.ndim != 3 or d.shape[0] != 1 or not np.all(np.isfinite(d)):
            raise RuntimeContractError(
                "Kokoro cloning duration embeddings have incompatible values"
            )
        if d.shape[1] == durations.size:
            duration_context = d.transpose(0, 2, 1)
        elif d.shape[2] == durations.size:
            duration_context = d
        else:
            raise RuntimeContractError("Kokoro cloning durations and embeddings do not match")
        if (
            t_en.ndim != 3
            or t_en.shape[0] != 1
            or t_en.shape[1] != 512
            or t_en.shape[2] != durations.size
            or not np.all(np.isfinite(t_en))
        ):
            raise RuntimeContractError("Kokoro cloning text embeddings have incompatible shapes")
        index = np.repeat(np.arange(durations.size, dtype=np.int64), durations)
        en = np.ascontiguousarray(duration_context[:, :, index])
        asr = np.ascontiguousarray(t_en[:, :, index])

        curves = self._get_session("curves")
        self._require_inputs(curves, {"en", "style_dur"}, "curves")
        curve_values = self._run(curves, {"en": en, "style_dur": style_dur})
        curve_named = self._named_outputs(curves, curve_values)
        f0_curve = self._output(curve_named, ("f0_curve", "f0"), 0)
        n_curve = self._output(curve_named, ("n_curve", "noise"), 1)
        if not np.all(np.isfinite(f0_curve)) or not np.all(np.isfinite(n_curve)):
            raise RuntimeContractError("Kokoro cloning curves must contain only finite values")
        har = self._harmonic_source(f0_curve, seed)

        decoder = self._get_session("decoder")
        self._require_inputs(
            decoder, {"asr", "f0_curve", "n_curve", "style_acou", "har"}, "decoder"
        )
        decoder_values = self._run(
            decoder,
            {
                "asr": asr,
                "f0_curve": f0_curve,
                "n_curve": n_curve,
                "style_acou": style_acou,
                "har": har,
            },
        )
        decoder_named = self._named_outputs(decoder, decoder_values)
        audio = self._canonical_audio(self._output(decoder_named, ("audio", "output"), 0))
        outputs = {
            "pred_dur": durations,
            "f0_curve": np.asarray(f0_curve),
            "n_curve": np.asarray(n_curve),
        }
        return InferenceResult(
            audio=audio,
            sample_rate=int(
                self.installation.sample_rate or self._runtime_value("sample_rate", 24000)
            ),
            timings=durations,
            outputs=outputs,
            metadata={
                "system": "kokoro",
                "layout": CLONING_LAYOUT,
                "seed": int(seed),
                "reference_fingerprint": self._reference_fingerprint(reference),
                "speed": 1.0,
            },
        )

    def _validate_text_tokens(self, token_ids: Sequence[int]) -> list[int]:
        tokens = list(token_ids)
        if not tokens:
            raise RuntimeContractError("Kokoro cloning inference requires at least one token")
        if len(tokens) + 2 > self.max_tokens:
            raise RuntimeContractError(
                f"Kokoro cloning token sequence exceeds the {self.max_tokens}-token limit"
            )
        if any(
            isinstance(token, bool) or not isinstance(token, (int, np.integer)) for token in tokens
        ):
            raise RuntimeContractError("Kokoro cloning token ids must be integers")
        return [int(token) for token in tokens]

    @staticmethod
    def _reference_fingerprint(reference: KokoroReferenceState) -> str:
        digest = hashlib.sha256(reference.model_fingerprint.encode("ascii"))
        for value in (reference.style, reference.memory, reference.memory_mask):
            array = np.ascontiguousarray(value)
            digest.update(array.dtype.str.encode("ascii"))
            digest.update(json.dumps(array.shape, separators=(",", ":")).encode("ascii"))
            digest.update(array.tobytes())
        return digest.hexdigest()

    def _validate_reference_tokens(self, token_ids: Sequence[int]) -> list[int]:
        tokens = list(token_ids)
        if not tokens:
            raise RuntimeContractError("Kokoro reference enrollment requires at least one token")
        if len(tokens) + 2 > self.max_tokens:
            raise RuntimeContractError(
                f"Kokoro reference token sequence exceeds the {self.max_tokens}-token limit"
            )
        if any(
            isinstance(token, bool) or not isinstance(token, (int, np.integer)) for token in tokens
        ):
            raise RuntimeContractError("Kokoro reference token ids must be integers")
        return [int(token) for token in tokens]

    @staticmethod
    def _validate_reference_audio(
        audio_24k: np.ndarray, audio_16k: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        wave24 = KokoroCloningRuntime._mono_float_audio(audio_24k, "audio_24k")
        wave16 = KokoroCloningRuntime._mono_float_audio(audio_16k, "audio_16k")
        seconds24 = len(wave24) / 24000.0
        seconds16 = len(wave16) / 16000.0
        if not REFERENCE_MIN_SECONDS <= seconds24 <= REFERENCE_MAX_SECONDS:
            raise RuntimeContractError("Kokoro reference audio must be 3–30 seconds")
        if abs(seconds24 - seconds16) > AUDIO_DURATION_TOLERANCE_SECONDS:
            raise RuntimeContractError("Kokoro reference audio sample-rate durations do not agree")
        for wave, name in ((wave24, "audio_24k"), (wave16, "audio_16k")):
            rms = np.sqrt(np.mean(np.square(wave, dtype=np.float64)))
            if rms < REFERENCE_SPEECH_RMS_MINIMUM:
                raise RuntimeContractError(f"Kokoro reference {name} is silent or too quiet")
            clipped = np.mean(np.abs(wave) >= REFERENCE_CLIPPING_THRESHOLD)
            if clipped > REFERENCE_CLIPPING_FRACTION_MAXIMUM:
                raise RuntimeContractError(f"Kokoro reference {name} is severely clipped")
        return wave24, wave16

    @staticmethod
    def _mono_float_audio(value: np.ndarray, name: str) -> np.ndarray:
        audio = np.asarray(value)
        if audio.ndim != 1 or not np.issubdtype(audio.dtype, np.floating):
            raise RuntimeContractError(f"{name} must be a mono one-dimensional float array")
        if not np.all(np.isfinite(audio)):
            raise RuntimeContractError(f"{name} must contain only finite samples")
        return np.ascontiguousarray(audio, dtype=np.float32)

    @staticmethod
    def _validate_observation(value: np.ndarray, shape: tuple[int, ...], name: str) -> np.ndarray:
        observation = np.asarray(value, dtype=np.float32)
        if observation.shape != shape or not np.all(np.isfinite(observation)):
            raise RuntimeContractError(f"Kokoro {name} output must be finite with shape {shape}")
        return observation

    @staticmethod
    def _require_inputs(session: Any, required: set[str], component: str) -> None:
        names = SplitKokoroRuntime._session_names(session)
        if names and not required <= names:
            raise RuntimeContractError(
                f"Kokoro {component} graph is missing inputs: "
                + ", ".join(sorted(required - names))
            )
