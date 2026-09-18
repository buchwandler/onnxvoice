from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from typing import Any

from ..runtime import OnnxSession
from ..types import InferenceResult, Installation, RuntimeDiagnostic


class SystemAdapter(ABC):
    """Base lifecycle for a producer-neutral voice runtime adapter."""

    system: str

    def __init__(
        self,
        installation: Installation,
        *,
        providers: str | Sequence[str] | None = None,
        provider_options: Sequence[dict[str, Any]] | Mapping[str, dict[str, Any]] | None = None,
        session_options: Any | None = None,
    ) -> None:
        self.installation = installation
        self.providers = providers
        self.provider_options = provider_options
        self.session_options = session_options
        self._session: OnnxSession | Any | None = None

    @abstractmethod
    def infer(self, token_ids: Sequence[int], **kwargs: Any) -> InferenceResult:
        raise NotImplementedError

    def _owned_sessions(self) -> tuple[Any, ...]:
        """Return all lazily-created sessions owned by this adapter."""
        sessions: list[Any] = []
        single = getattr(self, "_session", None)
        if single is not None:
            sessions.append(single)
        multiple = getattr(self, "_sessions", None)
        if isinstance(multiple, Mapping):
            sessions.extend(session for session in multiple.values() if session is not None)
        elif isinstance(multiple, Sequence) and not isinstance(multiple, (str, bytes)):
            sessions.extend(session for session in multiple if session is not None)
        unique: list[Any] = []
        for session in sessions:
            if not any(session is existing for existing in unique):
                unique.append(session)
        return tuple(unique)

    def close(self) -> None:
        """Close every owned session. Repeated calls are safe."""
        sessions = self._owned_sessions()
        self._session = None
        multiple = getattr(self, "_sessions", None)
        if isinstance(multiple, (dict, list)):
            multiple.clear()
        for session in sessions:
            close = getattr(session, "close", None)
            if close is not None:
                close()

    def diagnostics(self) -> RuntimeDiagnostic:
        """Return producer-neutral diagnostics for every created session."""
        diagnostics = tuple(
            diagnostic
            for session in self._owned_sessions()
            if (diagnostic := getattr(session, "diagnostics", lambda: None)()) is not None
        )
        runtime = self.installation.metadata.get("runtime") or {}
        return RuntimeDiagnostic(
            system=self.installation.system,
            ref=self.installation.ref,
            layout=str(runtime.get("layout", "single")),
            sessions=diagnostics,
        )

    def __enter__(self) -> SystemAdapter:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
