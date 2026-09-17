from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any

from ..runtime import OnnxSession
from ..types import AudioResult, Installation


class SystemAdapter(ABC):
    system: str

    def __init__(
        self,
        installation: Installation,
        *,
        providers: Sequence[str] | None = None,
        provider_options: Sequence[dict[str, Any]] | None = None,
    ) -> None:
        self.installation = installation
        self.providers = providers
        self.provider_options = provider_options
        self._session: OnnxSession | None = None

    @property
    @abstractmethod
    def session(self) -> OnnxSession:
        raise NotImplementedError

    @abstractmethod
    def infer(self, tokens: Sequence[int], **kwargs: Any) -> AudioResult:
        raise NotImplementedError

    def close(self) -> None:
        if self._session is not None:
            self._session.close()
            self._session = None

    def __enter__(self) -> SystemAdapter:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
