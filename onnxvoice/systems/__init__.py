from __future__ import annotations

from ..errors import UnsupportedSystemError
from .base import SystemAdapter
from .kitten import KittenAdapter
from .kokoro import KokoroAdapter
from .kokoro_cloning import KokoroCloningRuntime, KokoroReferenceState
from .kokoro_inno import KokoroVoicePack
from .kokoro_split import SplitKokoroRuntime
from .piper import PiperAdapter
from .pocket import PocketAdapter
from .supertonic import SupertonicAdapter

_ADAPTERS: dict[str, type[SystemAdapter]] = {
    PiperAdapter.system: PiperAdapter,
    KokoroAdapter.system: KokoroAdapter,
    PocketAdapter.system: PocketAdapter,
    SupertonicAdapter.system: SupertonicAdapter,
    KittenAdapter.system: KittenAdapter,
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
    "PocketAdapter",
    "SupertonicAdapter",
    "KittenAdapter",
    "SplitKokoroRuntime",
    "KokoroCloningRuntime",
    "KokoroReferenceState",
    "KokoroVoicePack",
    "register_adapter",
    "get_adapter",
    "registered_systems",
]
