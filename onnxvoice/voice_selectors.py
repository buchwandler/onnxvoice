"""Stable short selectors for catalog voice identities.

This module is deliberately independent of ONNX Runtime.  Selectors are
persisted registry assignments, not positions in a catalog response.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from functools import lru_cache
from importlib import resources
from pathlib import Path
from typing import Any

from .errors import (
    VoiceSelectorError,
    VoiceSelectorNotFoundError,
    VoiceSelectorRegistryError,
    VoiceSelectorRetiredError,
)
from .types import CatalogItem, VoiceIdentity, VoiceRecord

VOICE_ENGINE_CODES: dict[str, str] = {
    "kokoro": "ko",
    "piper": "pi",
}
ENGINE_CODE_SYSTEMS: dict[str, str] = {code: system for system, code in VOICE_ENGINE_CODES.items()}
_SUPPORTED_SCHEMA = 1
_LANGUAGE_RE = re.compile(r"^[a-z0-9]+(?:_[a-z0-9]+)*$")
_ENGINE_CODE_RE = re.compile(r"^[a-z]{2}$")


def _language_key(value: str) -> str:
    if not isinstance(value, str) or not value:
        raise VoiceSelectorError("selector language must be a non-empty string")
    normalized = value.casefold().replace("-", "_")
    if not _LANGUAGE_RE.fullmatch(normalized):
        raise VoiceSelectorError(f"invalid selector language key: {value!r}")
    return normalized


def _registry_language_key(value: Any) -> str:
    if not isinstance(value, str) or value != _language_key(value):
        raise VoiceSelectorRegistryError(f"invalid registry language key: {value!r}")
    return value


def _engine_code(value: str, *, require_registered: bool = False) -> str:
    if not isinstance(value, str):
        raise VoiceSelectorError("engine code must be a string")
    value = value.casefold()
    if not _ENGINE_CODE_RE.fullmatch(value):
        raise VoiceSelectorError(f"invalid engine code: {value!r}")
    if require_registered and value not in ENGINE_CODE_SYSTEMS:
        raise VoiceSelectorError(f"unknown engine code: {value!r}")
    return value


def _canonical_selector(language: str, engine_code: str, slot: int) -> str:
    return f"{_language_key(language)}-{_engine_code(engine_code)}-{slot}"


def format_voice_selector(language: str, engine_code: str, slot: int) -> str:
    """Return the one canonical spelling of a short voice selector."""
    language_key = _language_key(language)
    code = _engine_code(engine_code, require_registered=True)
    if isinstance(slot, bool) or not isinstance(slot, int) or slot < 1:
        raise VoiceSelectorError("selector slot must be a positive integer")
    return f"{language_key}-{code}-{slot}"


@dataclass(frozen=True, slots=True)
class ParsedVoiceSelector:
    """Parsed selector components in canonical form."""

    language: str
    engine_code: str
    slot: int

    @property
    def selector(self) -> str:
        return f"{self.language}-{self.engine_code}-{self.slot}"


def parse_voice_selector(value: str) -> ParsedVoiceSelector:
    """Parse a selector by splitting from the right.

    Regional input such as ``en-us-ko-1`` is accepted and canonicalized to
    ``en_us-ko-1``.  Unknown two-letter engine codes remain syntactically
    parseable and are rejected by registry resolution.
    """
    if not isinstance(value, str) or not value or value != value.strip():
        raise VoiceSelectorError(f"invalid voice selector: {value!r}")
    try:
        language_part, engine_code, slot_text = value.rsplit("-", 2)
    except ValueError as exc:
        raise VoiceSelectorError(f"invalid voice selector: {value!r}") from exc
    language = _language_key(language_part)
    code = _engine_code(engine_code)
    if not slot_text.isdecimal() or (len(slot_text) > 1 and slot_text.startswith("0")):
        raise VoiceSelectorError(f"invalid selector slot: {slot_text!r}")
    slot = int(slot_text)
    if slot < 1:
        raise VoiceSelectorError("selector slot must be positive")
    return ParsedVoiceSelector(language, code, slot)


def is_voice_selector(value: str) -> bool:
    """Return whether *value* has valid selector syntax."""
    try:
        parse_voice_selector(value)
    except VoiceSelectorError:
        return False
    return True


class VoiceSelectorRegistry:
    """Validated immutable view of the authoritative selector registry."""

    def __init__(self, engine_codes: Mapping[str, str], identities: Iterable[VoiceIdentity]):
        codes = {str(system): str(code) for system, code in engine_codes.items()}
        if len(set(codes.values())) != len(codes):
            raise VoiceSelectorRegistryError("engine codes must be unique")
        for system, code in codes.items():
            if not system or not _ENGINE_CODE_RE.fullmatch(code):
                raise VoiceSelectorRegistryError(
                    f"invalid engine code mapping: {system!r} -> {code!r}"
                )
        identities_tuple = tuple(identities)
        by_selector: dict[str, VoiceIdentity] = {}
        by_voice: dict[tuple[str, str, str], VoiceIdentity] = {}
        for identity in identities_tuple:
            if identity.system not in codes:
                raise VoiceSelectorRegistryError(f"unknown registry system: {identity.system!r}")
            if codes[identity.system] != identity.engine_code:
                raise VoiceSelectorRegistryError(
                    f"engine code mismatch for {identity.system!r}: {identity.engine_code!r}"
                )
            expected_selector = _canonical_selector(
                identity.language, identity.engine_code, identity.slot
            )
            if identity.selector != expected_selector:
                raise VoiceSelectorRegistryError(
                    f"selector does not match registry fields: {identity.selector!r}"
                )
            if identity.state not in {"active", "retired"}:
                raise VoiceSelectorRegistryError(f"invalid selector state: {identity.state!r}")
            if identity.selector in by_selector:
                raise VoiceSelectorRegistryError(f"duplicate selector: {identity.selector}")
            if identity.canonical_key in by_voice:
                raise VoiceSelectorRegistryError(
                    f"duplicate canonical voice identity: {identity.canonical_key!r}"
                )
            by_selector[identity.selector] = identity
            by_voice[identity.canonical_key] = identity
        self.engine_codes = dict(codes)
        self._identities = tuple(
            sorted(identities_tuple, key=lambda item: (item.language, item.engine_code, item.slot))
        )
        self._by_selector = by_selector
        self._by_voice = by_voice

    @classmethod
    def from_data(cls, data: Mapping[str, Any]) -> VoiceSelectorRegistry:
        if not isinstance(data, Mapping):
            raise VoiceSelectorRegistryError("selector registry must be an object")
        if data.get("schema") != _SUPPORTED_SCHEMA:
            raise VoiceSelectorRegistryError(
                f"unsupported selector registry schema: {data.get('schema')!r}"
            )
        raw_codes = data.get("engine_codes")
        if not isinstance(raw_codes, Mapping):
            raise VoiceSelectorRegistryError("selector registry is missing engine_codes")
        engine_codes = {str(system): str(code) for system, code in raw_codes.items()}
        raw_entries = data.get("entries")
        if not isinstance(raw_entries, list):
            raise VoiceSelectorRegistryError("selector registry is missing entries")
        identities: list[VoiceIdentity] = []
        slot_keys: set[tuple[str, str, int]] = set()
        for index, raw in enumerate(raw_entries):
            if not isinstance(raw, Mapping):
                raise VoiceSelectorRegistryError(f"registry entry {index} must be an object")
            try:
                language = _registry_language_key(raw["language"])
                engine_code = str(raw["engine_code"])
                system = str(raw["system"])
                asset_id = str(raw["asset_id"])
                voice_id = str(raw["voice_id"])
                slot = raw["slot"]
                state = str(raw.get("state", "active"))
            except (KeyError, TypeError) as exc:
                raise VoiceSelectorRegistryError(f"registry entry {index} is incomplete") from exc
            if isinstance(slot, bool) or not isinstance(slot, int) or slot < 1:
                raise VoiceSelectorRegistryError(
                    f"invalid slot in registry entry {index}: {slot!r}"
                )
            if not asset_id or not voice_id:
                raise VoiceSelectorRegistryError(
                    f"registry entry {index} has empty identity fields"
                )
            key = (language, engine_code, slot)
            if key in slot_keys:
                raise VoiceSelectorRegistryError(f"duplicate selector slot: {key!r}")
            slot_keys.add(key)
            try:
                selector = _canonical_selector(language, engine_code, slot)
            except VoiceSelectorError as exc:
                raise VoiceSelectorRegistryError(
                    f"invalid registry entry {index} engine code: {engine_code!r}"
                ) from exc
            identities.append(
                VoiceIdentity(
                    selector=selector,
                    language=language,
                    engine_code=engine_code,
                    slot=slot,
                    system=system,
                    asset_id=asset_id,
                    voice_id=voice_id,
                    state=state,
                )
            )
        return cls(engine_codes, identities)

    @classmethod
    def from_file(cls, path: str | Path) -> VoiceSelectorRegistry:
        try:
            data = json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise VoiceSelectorRegistryError(f"cannot read selector registry {path!s}") from exc
        return cls.from_data(data)

    @classmethod
    def from_package(cls) -> VoiceSelectorRegistry:
        try:
            path = resources.files("onnxvoice").joinpath("data/voice_selectors.json")
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, ModuleNotFoundError) as exc:
            raise VoiceSelectorRegistryError("cannot load packaged selector registry") from exc
        return cls.from_data(data)

    @property
    def identities(self) -> tuple[VoiceIdentity, ...]:
        return self._identities

    def resolve(self, value: str, *, include_retired: bool = False) -> VoiceIdentity:
        parsed = parse_voice_selector(value)
        identity = self._by_selector.get(parsed.selector)
        if identity is None:
            raise VoiceSelectorNotFoundError(f"unknown voice selector: {parsed.selector}")
        if identity.state == "retired" and not include_retired:
            raise VoiceSelectorRetiredError(f"voice selector is retired: {identity.selector}")
        return identity

    def selector_for_voice(
        self,
        *,
        system: str,
        asset_id: str,
        voice_id: str,
        include_retired: bool = False,
    ) -> VoiceIdentity | None:
        identity = self._by_voice.get((system, asset_id, voice_id))
        if identity is None:
            return None
        if identity.state == "retired" and not include_retired:
            return None
        return identity

    def iter_identities(
        self,
        *,
        system: str | None = None,
        language: str | None = None,
        include_retired: bool = False,
    ) -> tuple[VoiceIdentity, ...]:
        system_key = system.casefold() if system is not None else None
        language_key = _language_key(language) if language is not None else None
        return tuple(
            identity
            for identity in self._identities
            if (system_key is None or identity.system == system_key)
            and (language_key is None or identity.language == language_key)
            and (include_retired or identity.state != "retired")
        )

    def validate(self) -> None:
        """Re-run construction-time invariants for maintenance callers."""
        type(self)(self.engine_codes, self.identities)


@lru_cache(maxsize=1)
def get_voice_selector_registry() -> VoiceSelectorRegistry:
    """Load the packaged authoritative registry once per process."""
    return VoiceSelectorRegistry.from_package()


def load_voice_selector_registry(
    source: str | Path | Mapping[str, Any] | None = None,
) -> VoiceSelectorRegistry:
    """Load the packaged registry or an explicit JSON path/data mapping."""
    if source is None:
        return get_voice_selector_registry()
    if isinstance(source, Mapping):
        return VoiceSelectorRegistry.from_data(source)
    return VoiceSelectorRegistry.from_file(source)


def resolve_voice_selector(
    value: str,
    *,
    include_retired: bool = False,
    registry: VoiceSelectorRegistry | None = None,
) -> VoiceIdentity:
    """Resolve a short selector to its full canonical identity."""
    return (registry or get_voice_selector_registry()).resolve(
        value, include_retired=include_retired
    )


def selector_for_voice(
    *,
    system: str,
    asset_id: str,
    voice_id: str,
    include_retired: bool = False,
    registry: VoiceSelectorRegistry | None = None,
) -> VoiceIdentity | None:
    """Look up a selector by full canonical voice identity."""
    return (registry or get_voice_selector_registry()).selector_for_voice(
        system=system,
        asset_id=asset_id,
        voice_id=voice_id,
        include_retired=include_retired,
    )


def iter_voice_identities(
    *,
    system: str | None = None,
    language: str | None = None,
    include_retired: bool = False,
    registry: VoiceSelectorRegistry | None = None,
) -> tuple[VoiceIdentity, ...]:
    """Iterate assigned identities in deterministic numeric-slot order."""
    return (registry or get_voice_selector_registry()).iter_identities(
        system=system, language=language, include_retired=include_retired
    )


def catalog_voice_keys(
    items: Iterable[CatalogItem],
) -> tuple[tuple[CatalogItem, str, str], ...]:
    """Flatten catalog items into ``(item, asset_id, voice_id)`` tuples."""
    result: list[tuple[CatalogItem, str, str]] = []
    for item in items:
        if item.system == "piper" and item.kind == "voice":
            result.append((item, item.id, item.id))
        elif item.system == "kokoro" and item.kind == "model":
            result.extend((item, item.id, voice_id) for voice_id in item.voices)
    return tuple(result)


def make_voice_record(
    identity: VoiceIdentity | None,
    *,
    available: bool,
    catalog_item: CatalogItem | None,
    languages: tuple[str, ...] = (),
    gender: str = "unknown",
    voice_id: str | None = None,
) -> VoiceRecord:
    """Small constructor used by manager and maintenance integrations."""
    return VoiceRecord(identity, available, catalog_item, languages, gender, voice_id)


__all__ = [
    "ENGINE_CODE_SYSTEMS",
    "VOICE_ENGINE_CODES",
    "ParsedVoiceSelector",
    "VoiceIdentity",
    "VoiceRecord",
    "VoiceSelectorRegistry",
    "catalog_voice_keys",
    "format_voice_selector",
    "get_voice_selector_registry",
    "is_voice_selector",
    "iter_voice_identities",
    "load_voice_selector_registry",
    "make_voice_record",
    "parse_voice_selector",
    "resolve_voice_selector",
    "selector_for_voice",
]
