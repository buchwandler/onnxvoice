from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from ..errors import CapabilityError, RuntimeContractError
from ..runtime import OnnxSession
from ..types import InferenceResult, Installation
from .base import SystemAdapter

HARMONICS = 9
UPSAMPLE_SCALE = 300
SAMPLE_RATE = 24000
VOICED_THRESHOLD = 10.0
SINE_AMP = 0.1
NOISE_STD = 0.003
N_FFT = 20
HOP = 5


class SplitKokoroRuntime(SystemAdapter):
    """Runtime mechanics for catalog layout ``split-onnx-v1``."""

    system = "kokoro"
    COMPONENTS = ("prosody", "curves", "decoder")

    def __init__(
        self,
        installation: Installation,
        *,
        session_factory: Callable[[Path, str], Any] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(installation, **kwargs)
        self._session_factory = session_factory
        self._sessions: dict[str, Any] = {}
        self.max_tokens = self._runtime_value("max_tokens", 510)
        self._source_weight: np.ndarray
        self._source_bias: np.ndarray
        self._window: np.ndarray
        self._load_supporting_assets()
        for component in self.COMPONENTS:
            self._component_path(component)

    def _runtime_value(self, key: str, default: Any) -> Any:
        runtime = self.installation.metadata.get("runtime") or {}
        return runtime.get(key, self.installation.metadata.get(key, default))

    def _artifact(self, role: str, *, component: str | None = None) -> Path:
        try:
            return self.installation.artifact(role, component=component).path
        except KeyError as exc:
            detail = f"role={role!r}" + (f", component={component!r}" if component else "")
            raise CapabilityError(f"Kokoro split runtime is missing {detail}") from exc

    def _component_path(self, component: str) -> Path:
        try:
            return self._artifact("model", component=component)
        except CapabilityError:
            return self._artifact(component)

    def _load_supporting_assets(self) -> None:
        config_path: Path
        try:
            config_path = self._artifact("config")
        except CapabilityError:
            try:
                config_path = self._artifact("manifest")
            except CapabilityError as exc:
                raise CapabilityError("Kokoro split runtime is missing its config/manifest") from exc
        try:
            manifest = json.loads(config_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeContractError(f"Could not read Kokoro split manifest {config_path}: {exc}") from exc
        if not isinstance(manifest, dict):
            raise RuntimeContractError("Kokoro split manifest must contain an object")
        runtime = self.installation.metadata.get("runtime") or {}
        max_tokens = runtime.get("max_tokens", manifest.get("max_tokens"))
        if isinstance(max_tokens, int) and max_tokens > 0:
            self.max_tokens = max_tokens

        source_path = self._find_artifact(("source_params", "source-params", "metadata"))
        if source_path is None:
            raise CapabilityError("Kokoro split runtime is missing source parameters")
        try:
            with np.load(source_path, allow_pickle=False) as source:
                self._source_weight = self._array_from_npz(source, ("weight", "source_weight"))
                self._source_bias = self._array_from_npz(source, ("bias", "source_bias"))
                self._window = self._array_from_npz(source, ("window",))
        except (OSError, ValueError, KeyError) as exc:
            raise RuntimeContractError(f"Invalid Kokoro split source parameters {source_path}: {exc}") from exc
        if self._source_weight.ndim != 2 or self._source_bias.ndim != 1:
            raise RuntimeContractError("Kokoro split source parameters have invalid shapes")
        if self._source_weight.shape[0] != self._source_bias.shape[0]:
            raise RuntimeContractError("Kokoro split source weight and bias shapes do not match")
        if self._window.ndim != 1 or self._window.size != N_FFT:
            raise RuntimeContractError("Kokoro split source window must contain 20 values")

        voices_path = self._find_artifact(("voices",))
        if voices_path is None:
            raise CapabilityError("Kokoro split runtime is missing voices")
        try:
            with np.load(voices_path, allow_pickle=False):
                pass
        except (OSError, ValueError) as exc:
            raise RuntimeContractError(f"Invalid Kokoro split voices {voices_path}: {exc}") from exc

    @staticmethod
    def _array_from_npz(source: Any, names: tuple[str, ...]) -> np.ndarray:
        for name in names:
            if name in source.files:
                return np.asarray(source[name])
        raise KeyError(names[0])

    def _find_artifact(self, roles: tuple[str, ...]) -> Path | None:
        for role in roles:
            matches = self.installation.artifacts_for(role=role)
            if matches:
                return matches[0].path
        for artifact in self.installation.artifacts:
            name = artifact.filename.lower()
            if any(role in name for role in roles) and artifact.path.suffix == ".npz":
                return artifact.path
        return None

    def _get_session(self, component: str) -> Any:
        if component not in self._sessions:
            path = self._component_path(component)
            if self._session_factory is None:
                session = OnnxSession(
                    path,
                    component=component,
                    providers=self.providers,
                    provider_options=self.provider_options,
                    session_options=self.session_options,
                )
            else:
                session = self._session_factory(path, component)
            self._sessions[component] = session
        return self._sessions[component]

    @staticmethod
    def _session_names(session: Any) -> set[str]:
        names = getattr(session, "input_names", None)
        if names is not None:
            return set(names)
        getter = getattr(session, "get_inputs", None)
        return {str(node.name) for node in getter()} if getter is not None else set()

    @staticmethod
    def _run(session: Any, inputs: dict[str, np.ndarray]) -> list[Any]:
        try:
            return list(session.run(inputs))
        except TypeError:
            return list(session.run(None, inputs))

    @staticmethod
    def _named_outputs(session: Any, values: list[Any]) -> dict[str, np.ndarray]:
        names = tuple(getattr(session, "output_names", ()))
        if len(names) != len(values):
            names = tuple(f"output_{index}" for index in range(len(values)))
        return {name: np.asarray(value) for name, value in zip(names, values, strict=True)}

    @staticmethod
    def _output(named: Mapping[str, np.ndarray], names: tuple[str, ...], index: int) -> np.ndarray:
        for name in names:
            if name in named:
                return named[name]
        values = tuple(named.values())
        if index >= len(values):
            raise RuntimeContractError(f"Kokoro split graph is missing output {names[0]!r}")
        return values[index]
    def diagnostics(self) -> Any:
        for component in self.COMPONENTS:
            self._get_session(component)
        return super().diagnostics()


    def infer(
        self,
        token_ids: Sequence[int],
        **kwargs: Any,
    ) -> InferenceResult:
        try:
            style = kwargs.pop("style")
        except KeyError as exc:
            raise RuntimeContractError("A model-ready Kokoro style must be provided") from exc
        speed = kwargs.pop("speed", 1.0)
        seed = kwargs.pop("seed", 1234)
        if kwargs:
            raise RuntimeContractError(
                "Unsupported Kokoro split inference arguments: " + ", ".join(sorted(kwargs))
            )
        token_values = list(token_ids)
        if not token_values:
            raise RuntimeContractError("Kokoro split inference requires at least one token")
        if isinstance(self.max_tokens, int) and self.max_tokens > 0 and len(token_values) > self.max_tokens:
            raise RuntimeContractError(
                f"Kokoro token sequence has {len(token_values)} tokens, maximum is {self.max_tokens}"
            )
        style_value = np.asarray(style, dtype=np.float32)
        if style_value.ndim == 1:
            style_value = style_value[None, :]
        if style_value.ndim != 2 or style_value.shape[0] != 1:
            raise RuntimeContractError("Kokoro split style must be one model-ready row")
        duration_width = self._runtime_value("style_duration", None)
        acoustic_width = self._runtime_value("style_acoustic", None)
        if duration_width is None or acoustic_width is None:
            dimensions = self._runtime_value("style_dimensions", {})
            if isinstance(dimensions, Mapping):
                duration_width = dimensions.get("duration", duration_width)
                acoustic_width = dimensions.get("acoustic", acoustic_width)
        if duration_width is None:
            duration_width = style_value.shape[1] // 2
        if acoustic_width is None:
            acoustic_width = style_value.shape[1] - int(duration_width)
        if int(duration_width) + int(acoustic_width) != style_value.shape[1]:
            raise RuntimeContractError("Kokoro split style shape does not match runtime dimensions")
        style_acou = np.ascontiguousarray(style_value[:, : int(acoustic_width)])
        style_dur = np.ascontiguousarray(style_value[:, int(acoustic_width) :])

        prosody = self._get_session("prosody")
        names = self._session_names(prosody)
        required = {"input_ids", "style_dur", "speed"}
        if names and not required <= names:
            raise RuntimeContractError(
                "Kokoro split prosody graph is missing inputs: "
                + ", ".join(sorted(required - names))
            )
        prosody_values = self._run(
            prosody,
            {"input_ids": np.asarray([[0, *token_values, 0]], dtype=np.int64), "style_dur": style_dur, "speed": np.asarray([speed], dtype=np.float32)},
        )
        prosody_named = self._named_outputs(prosody, prosody_values)
        pred_dur = np.asarray(self._output(prosody_named, ("pred_dur", "duration"), 0)).reshape(-1)
        if np.any(pred_dur < 0) or not np.all(np.isfinite(pred_dur)):
            raise RuntimeContractError("Kokoro split duration output must be finite and non-negative")
        pred_dur_int = pred_dur.astype(np.int64)
        d = self._output(prosody_named, ("d", "duration_embedding"), 1)
        t_en = self._output(prosody_named, ("t_en", "asr"), 2)
        if d.ndim < 3 or t_en.ndim < 3 or len(pred_dur_int) == 0:
            raise RuntimeContractError("Kokoro split prosody outputs have incompatible shapes")
        if d.shape[-1] != len(pred_dur_int) and d.shape[1] != len(pred_dur_int):
            raise RuntimeContractError("Kokoro split duration and embedding shapes do not match")
        index = np.repeat(np.arange(len(pred_dur_int), dtype=np.int64), pred_dur_int)
        try:
            en = np.ascontiguousarray(d.transpose(0, 2, 1)[:, :, index])
            asr = np.ascontiguousarray(t_en[:, :, index])
        except (IndexError, ValueError) as exc:
            raise RuntimeContractError(f"Kokoro split duration expansion failed: {exc}") from exc

        curves = self._get_session("curves")
        curve_names = self._session_names(curves)
        if curve_names and not {"en", "style_dur"} <= curve_names:
            raise RuntimeContractError("Kokoro split curves graph has an incompatible input contract")
        curve_values = self._run(curves, {"en": en, "style_dur": style_dur})
        curve_named = self._named_outputs(curves, curve_values)
        f0_curve = self._output(curve_named, ("f0_curve", "f0"), 0)
        n_curve = self._output(curve_named, ("n_curve", "noise"), 1)
        har = self._harmonic_source(f0_curve, seed)

        decoder = self._get_session("decoder")
        decoder_names = self._session_names(decoder)
        required = {"asr", "f0_curve", "n_curve", "style_acou", "har"}
        if decoder_names and not required <= decoder_names:
            raise RuntimeContractError(
                "Kokoro split decoder graph is missing inputs: "
                + ", ".join(sorted(required - decoder_names))
            )
        decoder_values = self._run(
            decoder,
            {"asr": asr, "f0_curve": f0_curve, "n_curve": n_curve, "style_acou": style_acou, "har": har},
        )
        decoder_named = self._named_outputs(decoder, decoder_values)
        audio = self._output(decoder_named, ("audio", "output"), 0)
        audio = self._canonical_audio(audio)
        outputs = {"pred_dur": pred_dur, "f0_curve": np.asarray(f0_curve), "n_curve": np.asarray(n_curve)}
        return InferenceResult(
            audio=audio,
            sample_rate=int(self.installation.sample_rate or self._runtime_value("sample_rate", SAMPLE_RATE)),
            timings=pred_dur,
            outputs=outputs,
            metadata={"system": "kokoro", "layout": "split-onnx-v1", "speed": speed, "seed": seed},
        )

    def _harmonic_source(self, f0_curve: np.ndarray, seed: int) -> np.ndarray:
        f0_values = np.asarray(f0_curve, dtype=np.float64)
        if f0_values.ndim == 1:
            f0_values = f0_values[None, :]
        if f0_values.ndim != 2:
            raise RuntimeContractError("Kokoro split f0 curve must be two-dimensional")
        rng = np.random.default_rng(seed)
        f0 = np.repeat(f0_values, UPSAMPLE_SCALE, axis=1)[..., None]
        rad = f0 * np.arange(1, HARMONICS + 1, dtype=np.float64) / SAMPLE_RATE
        rad -= np.floor(rad)
        rand_ini = rng.random((f0.shape[0], HARMONICS))
        rand_ini[:, 0] = 0.0
        rad[:, 0, :] += rand_ini
        phase = np.cumsum(self._resample(rad, 1 / UPSAMPLE_SCALE), axis=1) * 2 * np.pi
        phase = self._resample(phase * UPSAMPLE_SCALE, UPSAMPLE_SCALE)
        voiced = (f0 > VOICED_THRESHOLD).astype(np.float64)
        amplitude = voiced * NOISE_STD + (1 - voiced) * SINE_AMP / 3
        noise = rng.standard_normal((f0.shape[0], f0.shape[1], HARMONICS))
        waves = np.sin(phase) * SINE_AMP * voiced + amplitude * noise
        if waves.shape[-1] != self._source_weight.shape[1]:
            raise RuntimeContractError("Kokoro split source parameter width does not match harmonics")
        merged = np.tanh(waves @ self._source_weight.T + self._source_bias)
        return self._stft(merged[0, :, 0].astype(np.float32))

    @staticmethod
    def _resample(values: np.ndarray, scale: float) -> np.ndarray:
        length = values.shape[1]
        output_length = int(length * scale)
        if length <= 0 or output_length <= 0:
            raise RuntimeContractError("Kokoro split source curve is empty")
        source = (np.arange(output_length, dtype=np.float64) + 0.5) / scale - 0.5
        np.clip(source, 0, length - 1, out=source)
        lower = np.floor(source).astype(np.int64)
        upper = np.minimum(lower + 1, length - 1)
        weight = (source - lower)[None, :, None]
        return values[:, lower] * (1 - weight) + values[:, upper] * weight

    def _stft(self, audio: np.ndarray) -> np.ndarray:
        if audio.size < N_FFT:
            raise RuntimeContractError("Kokoro split source output is shorter than one STFT frame")
        padded = np.pad(audio, N_FFT // 2, mode="reflect")
        frames = np.lib.stride_tricks.as_strided(
            padded,
            shape=((len(padded) - N_FFT) // HOP + 1, N_FFT),
            strides=(padded.strides[0] * HOP, padded.strides[0]),
        )
        spectrum = np.fft.rfft(frames * self._window, axis=1).T[None]
        return np.concatenate([np.abs(spectrum), np.angle(spectrum)], axis=1).astype(np.float32)

    @staticmethod
    def _canonical_audio(value: np.ndarray) -> np.ndarray:
        audio = np.asarray(value).squeeze()
        if audio.ndim != 1:
            if audio.ndim == 2 and 1 in audio.shape:
                audio = audio.reshape(-1)
            else:
                raise RuntimeContractError("Kokoro split audio output must be one-dimensional")
        return audio.astype(np.float32, copy=False)
