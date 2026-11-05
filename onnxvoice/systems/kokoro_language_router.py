from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from ..errors import CapabilityError, RuntimeContractError
from ..inventory import language_base
from ..runtime import OnnxSession
from ..types import InferenceResult, Installation, RuntimeDiagnostic
from .base import SystemAdapter
from .kokoro_static import infer_static_kokoro

LANGUAGE_ROUTER_LAYOUT = "language-router-onnx-v1"
LANGUAGE_MODELS = {"en": "model_en", "de": "model_de"}


class KokoroLanguageRouterRuntime(SystemAdapter):
    """Static Kokoro inference routed to a graph by explicit language."""

    system = "kokoro"

    def __init__(
        self,
        installation: Installation,
        *,
        session_factory: Callable[[Path, str], Any] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(installation, **kwargs)
        self._session_factory = session_factory
        self.language_models = self._parse_language_models()
        self._sessions: dict[str, Any] = {}

    def _parse_language_models(self) -> dict[str, str]:
        runtime = self.installation.metadata.get("runtime")
        if not isinstance(runtime, Mapping):
            raise RuntimeContractError("Kokoro language-router runtime metadata must be an object")
        configured = runtime.get("language_models")
        if not isinstance(configured, Mapping):
            raise RuntimeContractError("Kokoro language-router requires runtime.language_models")

        models: dict[str, str] = {}
        for raw_language, component in configured.items():
            language = language_base(raw_language)
            if not language or language in models:
                raise RuntimeContractError(
                    "Kokoro language_models has an invalid or duplicate language"
                )
            if not isinstance(component, str):
                raise RuntimeContractError(
                    f"Kokoro language_models entry for {language!r} must name a component"
                )
            models[language] = component
        if models != LANGUAGE_MODELS:
            raise RuntimeContractError(
                "Kokoro language-router language_models must map en to model_en and de to model_de"
            )

        language_codes = runtime.get("language_codes")
        if not isinstance(language_codes, (list, tuple)):
            raise RuntimeContractError("Kokoro language-router requires runtime.language_codes")
        normalized_codes = {language_base(code) for code in language_codes}
        if not set(models) <= normalized_codes:
            raise RuntimeContractError(
                "Kokoro language-router language_codes must include en and de"
            )
        return models

    def _component_path(self, language: str, component: str) -> Path:
        candidates = self.installation.artifacts_for("model", component=component)
        quality = self.installation.selected_quality
        if quality is not None:
            selected = tuple(artifact for artifact in candidates if artifact.quality == quality)
            if selected:
                candidates = selected
            elif not (len(candidates) == 1 and candidates[0].quality is None):
                candidates = ()
        if len(candidates) != 1:
            raise CapabilityError(
                "Kokoro language-router requires exactly one model artifact for "
                f"language={language!r}, component={component!r}, quality={quality!r}"
            )
        return candidates[0].path

    def _session_for_language(self, language: str) -> Any:
        if language not in self._sessions:
            component = self.language_models[language]
            path = self._component_path(language, component)
            label = f"{language}:{component}"
            if self._session_factory is None:
                session = OnnxSession(
                    path,
                    component=label,
                    providers=self.providers,
                    provider_options=self.provider_options,
                    session_options=self.session_options,
                )
            else:
                session = self._session_factory(path, label)
            self._sessions[language] = session
        return self._sessions[language]

    def infer(
        self,
        token_ids: Sequence[int],
        *,
        style: np.ndarray | Sequence[float] | None = None,
        reference: Any | None = None,
        language: str | None = None,
        speed: float = 1.0,
        seed: int = 1234,
        **kwargs: Any,
    ) -> InferenceResult:
        del seed
        if kwargs:
            raise RuntimeContractError(
                "Unsupported Kokoro language-router inference arguments: "
                + ", ".join(sorted(kwargs))
            )
        if reference is not None:
            raise RuntimeContractError(
                "Reference conditioning is not supported by the Kokoro language-router layout"
            )
        if language is None or not isinstance(language, str) or not language.strip():
            raise RuntimeContractError(
                "A language is required for Kokoro language-router inference"
            )
        normalized_language = language_base(language)
        if normalized_language not in self.language_models:
            raise RuntimeContractError(
                f"Unsupported Kokoro language-router language {language!r}; "
                "supported languages are en and de"
            )
        if style is None:
            raise RuntimeContractError("A model-ready Kokoro style must be provided")
        session = self._session_for_language(normalized_language)
        return infer_static_kokoro(
            self.installation,
            session,
            token_ids,
            style=style,
            speed=speed,
        )

    def diagnostics(self) -> RuntimeDiagnostic:
        base = super().diagnostics()
        return RuntimeDiagnostic(
            system=base.system,
            ref=base.ref,
            layout=base.layout,
            sessions=base.sessions,
            properties={"language_models": dict(self.language_models)},
        )
