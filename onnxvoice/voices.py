"""Catalog-derived voice references and voice iteration."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass

from .catalog import parse_ref
from .types import CatalogItem


@dataclass(frozen=True, slots=True)
class ParsedVoiceRef:
    system: str
    asset_id: str
    voice_id: str | None

    @property
    def backing_ref(self) -> str:
        return f"{self.system}:{self.asset_id}"


def parse_voice_ref(ref: str) -> ParsedVoiceRef:
    """Parse a semantic catalog reference with an optional child voice ID."""
    if not isinstance(ref, str) or not ref or ref != ref.strip():
        raise ValueError(f"Invalid voice reference: {ref!r}")

    backing_ref, separator, voice_id = ref.partition("/")
    if separator and (not voice_id or "/" in voice_id):
        raise ValueError(f"Invalid voice reference: {ref!r}")

    system, asset_id = parse_ref(backing_ref)
    return ParsedVoiceRef(system, asset_id, voice_id if separator else None)


def catalog_voice_ids(item: CatalogItem) -> tuple[str, ...]:
    """Return voice IDs explicitly exposed by a normalized catalog item."""
    if item.kind == "voice":
        return (item.id,)
    return item.voices


def iter_catalog_voices(items: Iterable[CatalogItem]) -> Iterator[tuple[CatalogItem, str]]:
    """Yield each normalized catalog item paired with its exposed voice ID."""
    for item in items:
        for voice_id in catalog_voice_ids(item):
            yield item, voice_id


__all__ = ["ParsedVoiceRef", "catalog_voice_ids", "iter_catalog_voices", "parse_voice_ref"]
