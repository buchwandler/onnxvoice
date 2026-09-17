from __future__ import annotations

from ..errors import UnsupportedSystemError
from .base import SystemAdapter
from .kokoro import KokoroAdapter
from .piper import PiperAdapter

_ADAPTERS: dict[str, type[SystemAdapter]] = {
    PiperAdapter.system: PiperAdapter,
    KokoroAdapter.system: KokoroAdapter,
}


def register_adapter(system: str, adapter: type[SystemAdapter]) -> None:
    _ADAPTERS[system.lower()] = adapter


def get_adapter(system: str) -> type[SystemAdapter]:
    try:
        return _ADAPTERS[system.lower()]
    except KeyError as exc:
        raise UnsupportedSystemError(f"No adapter registered for {system!r}") from exc


def registered_systems() -> tuple[str, ...]:
    return tuple(sorted(_ADAPTERS))


__all__ = [
    "SystemAdapter",
    "PiperAdapter",
    "KokoroAdapter",
    "register_adapter",
    "get_adapter",
    "registered_systems",
]
