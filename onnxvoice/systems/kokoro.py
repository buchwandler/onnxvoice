from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from ..errors import CapabilityError, RuntimeContractError
from ..inventory import normalize_language_tag
from ..runtime import OnnxSession
from ..types import InferenceResult, RuntimeDiagnostic
from .base import SystemAdapter
from .kokoro_cloning import CLONING_LAYOUT, KokoroCloningRuntime, KokoroReferenceState
from .kokoro_inno import (
    INNO_ENROLLER_ID,
    INNO_SUPPORTED_LAYOUTS,
    InnoVoiceTuner,
    KokoroVoicePack,
    validate_inno_installation,
)
from .kokoro_language_router import LANGUAGE_ROUTER_LAYOUT, KokoroLanguageRouterRuntime
from .kokoro_split import SplitKokoroRuntime
from .kokoro_static import infer_static_kokoro


class KokoroAdapter(SystemAdapter):
    system = "kokoro"

    def __init__(self, *args: Any, session_factory: Any | None = None, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._split_runtime: SplitKokoroRuntime | None = None
        self._cloning_runtime: KokoroCloningRuntime | None = None
        self._language_router_runtime: KokoroLanguageRouterRuntime | None = None
        self._inno_runtime: InnoVoiceTuner | None = None
        self._split_session_factory = session_factory
        runtime = self.installation.metadata.get("runtime") or {}
        if isinstance(runtime, Mapping):
            capabilities = runtime.get("voice_enrollers")
            if isinstance(capabilities, (list, tuple)) and any(
                isinstance(capability, Mapping) and capability.get("id") == INNO_ENROLLER_ID
                for capability in capabilities
            ):
                validate_inno_installation(self.installation)

    def _uses_split_runtime(self) -> bool:
        runtime = self.installation.metadata.get("runtime") or {}
        return runtime.get("layout") in {"split", "multi", "split-onnx-v1"}

    def _uses_cloning_runtime(self) -> bool:
        runtime = self.installation.metadata.get("runtime") or {}
        return runtime.get("layout") == CLONING_LAYOUT

    def _uses_language_router_runtime(self) -> bool:
        runtime = self.installation.metadata.get("runtime") or {}
        return runtime.get("layout") == LANGUAGE_ROUTER_LAYOUT

    def _split(self) -> SplitKokoroRuntime:
        if self._split_runtime is None:
            self._split_runtime = SplitKokoroRuntime(
                self.installation,
                session_factory=self._split_session_factory,
                providers=self.providers,
                provider_options=self.provider_options,
                session_options=self.session_options,
            )
        return self._split_runtime

    def _cloning(self) -> KokoroCloningRuntime:
        if self._cloning_runtime is None:
            self._cloning_runtime = KokoroCloningRuntime(
                self.installation,
                session_factory=self._split_session_factory,
                providers=self.providers,
                provider_options=self.provider_options,
                session_options=self.session_options,
            )
        return self._cloning_runtime

    def _language_router(self) -> KokoroLanguageRouterRuntime:
        if self._language_router_runtime is None:
            self._language_router_runtime = KokoroLanguageRouterRuntime(
                self.installation,
                session_factory=self._split_session_factory,
                providers=self.providers,
                provider_options=self.provider_options,
                session_options=self.session_options,
            )
        return self._language_router_runtime

    def _inno(self) -> InnoVoiceTuner:
        if self._inno_runtime is None:
            self._inno_runtime = InnoVoiceTuner(
                self.installation,
                session_factory=self._split_session_factory,
                providers=self.providers,
                provider_options=self.provider_options,
                session_options=self.session_options,
            )
        return self._inno_runtime

    def enroll_voice(
        self,
        audio: np.ndarray,
        *,
        sample_rate: int,
        enroller: str = INNO_ENROLLER_ID,
        options: Mapping[str, Any] | None = None,
    ) -> KokoroVoicePack:
        """Enroll a reference with an advertised Kokoro voice-pack tuner."""
        if enroller != INNO_ENROLLER_ID:
            raise CapabilityError(f"Unsupported Kokoro voice enroller {enroller!r}")
        if self._uses_cloning_runtime():
            raise CapabilityError(
                "Inno voice enrollment is available only on ordinary Kokoro layouts"
            )
        runtime_metadata = self.installation.metadata.get("runtime") or {}
        layout = runtime_metadata.get("layout", "single")
        if layout not in INNO_SUPPORTED_LAYOUTS:
            raise CapabilityError(
                f"Inno voice enrollment does not support Kokoro layout {layout!r}"
            )
        return self._inno().enroll(audio, sample_rate=sample_rate, options=options)

    def prepare_reference(
        self,
        reference_token_ids: Sequence[int],
        *,
        audio_24k: np.ndarray,
        audio_16k: np.ndarray,
    ) -> KokoroReferenceState:
        if not self._uses_cloning_runtime():
            raise CapabilityError("Reference enrollment requires the Kokoro cloning-onnx-v1 layout")
        return self._cloning().prepare_reference(
            reference_token_ids, audio_24k=audio_24k, audio_16k=audio_16k
        )

    @property
    def session(self) -> OnnxSession:
        if self._session is None:
            models = [
                artifact
                for artifact in self.installation.artifacts
                if artifact.role == "model" and artifact.component != "inno_voicepack"
            ]
            runtime = self.installation.metadata.get("runtime") or {}
            if len(models) != 1 or runtime.get("layout") in {
                "split",
                "multi",
                "split-onnx-v1",
                CLONING_LAYOUT,
                LANGUAGE_ROUTER_LAYOUT,
            }:
                raise CapabilityError(
                    "Split, cloning, language-router, or multi-component Kokoro layouts do not expose a single session"
                )
            self._session = OnnxSession(
                models[0].path,
                providers=self.providers,
                provider_options=self.provider_options,
                session_options=self.session_options,
            )
        return self._session

    def infer(
        self,
        token_ids: Sequence[int],
        *,
        style: np.ndarray | Sequence[float] | None = None,
        reference: KokoroReferenceState | None = None,
        language: str | None = None,
        speed: float = 1.0,
        seed: int = 1234,
        **_: Any,
    ) -> InferenceResult:
        if language is not None:
            if not isinstance(language, str):
                raise RuntimeContractError("Kokoro language must be a non-empty language tag")
            normalized_language = normalize_language_tag(language)
            if not normalized_language or any(
                not part or any(char.isspace() for char in part)
                for part in normalized_language.split("-")
            ):
                raise RuntimeContractError("Kokoro language must be a non-empty language tag")
        if self._uses_cloning_runtime():
            if style is not None:
                raise CapabilityError(
                    "Static style input is not supported by the Kokoro cloning layout"
                )
            if reference is None:
                raise RuntimeContractError(
                    "A Kokoro reference state must be provided for cloning inference"
                )
            return self._cloning().infer(token_ids, reference=reference, speed=speed, seed=seed)
        if self._uses_language_router_runtime():
            return self._language_router().infer(
                token_ids,
                style=style,
                reference=reference,
                language=language,
                speed=speed,
                seed=seed,
            )
        if reference is not None:
            raise CapabilityError("Kokoro reference state requires the cloning-onnx-v1 layout")
        if self._uses_split_runtime():
            if style is None:
                raise RuntimeContractError("A model-ready Kokoro style must be provided")
            return self._split().infer(token_ids, style=style, speed=speed, seed=seed)
        if style is None:
            raise RuntimeContractError("A model-ready Kokoro style must be provided")
        return infer_static_kokoro(
            self.installation,
            self.session,
            token_ids,
            style=style,
            speed=speed,
        )

    def close(self) -> None:
        if self._split_runtime is not None:
            self._split_runtime.close()
            self._split_runtime = None
        if self._language_router_runtime is not None:
            self._language_router_runtime.close()
            self._language_router_runtime = None
        if self._cloning_runtime is not None:
            self._cloning_runtime.close()
            self._cloning_runtime = None
        if self._inno_runtime is not None:
            self._inno_runtime.close()
            self._inno_runtime = None
        super().close()

    def diagnostics(self) -> RuntimeDiagnostic:
        if self._split_runtime is not None:
            base = self._split_runtime.diagnostics()
        elif self._cloning_runtime is not None:
            base = self._cloning_runtime.diagnostics()
        elif self._uses_language_router_runtime():
            base = self._language_router().diagnostics()
        else:
            base = super().diagnostics()
        if self._inno_runtime is None:
            return base
        inno = self._inno_runtime.diagnostics()
        return RuntimeDiagnostic(
            system=base.system,
            ref=base.ref,
            layout=base.layout,
            sessions=(*base.sessions, *inno.sessions),
        )
