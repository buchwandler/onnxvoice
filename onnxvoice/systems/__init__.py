from __future__ import annotations

from ..errors import UnsupportedSystemError
from .base import SystemAdapter
from .chatterbox import ChatterboxAdapter, ChatterboxVoiceState
from .inflect import InflectAdapter
from .kitten import KittenAdapter
from .kitten2 import Kitten2Adapter
from .kokoro import KokoroAdapter
from .kokoro_cloning import KokoroCloningRuntime, KokoroReferenceState
from .kokoro_inno import KokoroVoicePack
from .kokoro_split import SplitKokoroRuntime
from .piper import PiperAdapter
from .pocket import PocketAdapter
from .qwen3 import Qwen3Adapter
from .qwen3_audio import Qwen3VoiceState
from .sopro import SoproAdapter, SoproRun
from .sopro_frontend import SoproTextInput, SoproVoiceState
from .sopro_generation import SoproGenerationDefaults
from .sopro_streaming import SoproStreamChunk
from .supertonic import SupertonicAdapter

_ADAPTERS: dict[str, type[SystemAdapter]] = {
    PiperAdapter.system: PiperAdapter,
    KokoroAdapter.system: KokoroAdapter,
    PocketAdapter.system: PocketAdapter,
    SupertonicAdapter.system: SupertonicAdapter,
    SoproAdapter.system: SoproAdapter,
    KittenAdapter.system: KittenAdapter,
    Kitten2Adapter.system: Kitten2Adapter,
    InflectAdapter.system: InflectAdapter,
    Qwen3Adapter.system: Qwen3Adapter,
    ChatterboxAdapter.system: ChatterboxAdapter,
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
    "SoproAdapter",
    "SoproRun",
    "SoproTextInput",
    "SoproVoiceState",
    "SoproGenerationDefaults",
    "SoproStreamChunk",
    "KittenAdapter",
    "Kitten2Adapter",
    "Qwen3Adapter",
    "ChatterboxAdapter",
    "ChatterboxVoiceState",
    "Qwen3VoiceState",
    "InflectAdapter",
    "SplitKokoroRuntime",
    "KokoroCloningRuntime",
    "KokoroReferenceState",
    "KokoroVoicePack",
    "register_adapter",
    "get_adapter",
    "registered_systems",
]
