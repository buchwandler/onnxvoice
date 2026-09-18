from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from typing import Any

from ..runtime import OnnxSession
from ..types import InferenceResult, Installation


class SystemAdapter(ABC):
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
        self._session: OnnxSession | None = None

    @property
    @abstractmethod
    def session(self) -> OnnxSession:
        raise NotImplementedError

    @abstractmethod
    def infer(self, token_ids: Sequence[int], **kwargs: Any) -> InferenceResult:
        raise NotImplementedError

    def close(self) -> None:
        if self._session is not None:
            self._session.close()
            self._session = None

    def __enter__(self) -> SystemAdapter:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
