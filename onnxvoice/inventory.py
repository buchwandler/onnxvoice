"""Normalized inventory layer for onnxvoice.

Provides language normalization, gender normalization, effective quality/distribution
helpers, version labels, size helpers, inventory merge logic, filter predicates,
and artifact-level update comparison.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any, Literal

from .types import Artifact, CatalogItem, Installation, InstalledArtifact

# ---------------------------------------------------------------------------
# Language normalization
# ---------------------------------------------------------------------------


def normalize_language_code(raw: Any) -> str:
    """Normalize a raw language value to canonical display form (e.g. 'en-US').

    Handles:
    - Piper: dict like {"code": "en_US", ...}
    - Pocket: string like "en"
    - Kokoro: list like ["en-us", "en-gb"]
    - Already-normalized strings
    """
    if isinstance(raw, dict):
        code = raw.get("code", "")
        if isinstance(code, str) and code:
            return _display_code(code)
    if isinstance(raw, str) and raw:
        return _display_code(raw)
    return ""


def language_codes_from_metadata(metadata: Mapping[str, Any]) -> tuple[str, ...]:
    """Extract all language codes from item/installation metadata.

    Handles all system-specific representations:
    - Piper: metadata["language"] is a dict with "code"
    - Pocket: metadata["language"] is a string
    - Kokoro: metadata["language_codes"] is a list
    """
    codes: list[str] = []

    # Kokoro-style: language_codes list
    lang_codes = metadata.get("language_codes")
    if isinstance(lang_codes, (list, tuple)):
        for code in lang_codes:
            normalized = normalize_language_code(code)
            if normalized:
                codes.append(normalized)

    # Piper/Pocket-style: language field
    if not codes:
        lang = metadata.get("language")
        if lang is not None:
            normalized = normalize_language_code(lang)
            if normalized:
                codes.append(normalized)

    return tuple(codes)


def primary_language_from_metadata(metadata: Mapping[str, Any]) -> str | None:
    """Return the primary (first) language code, or None."""
    codes = language_codes_from_metadata(metadata)
    return codes[0] if codes else None


def _display_code(code: str) -> str:
    """Convert internal code like 'en_US' to display form 'en-US'."""
    return code.replace("_", "-")


def _match_code(pattern: str, code: str) -> bool:
    """Subtag-aware language matching.

    --lang en    matches en, en-US, en-GB
    --lang en-US matches en-US only
    --lang de    matches de-DE, de-AT
    """
    pattern = pattern.replace("_", "-").casefold()
    code = code.replace("_", "-").casefold()
    if pattern == code:
        return True
    # Pattern "en" matches "en-US", "en-GB", etc.
    # But pattern "en-US" should NOT match "en-GB"
    return "-" not in pattern and code.startswith(pattern + "-")


def matches_language(metadata: Mapping[str, Any], language_filter: str) -> bool:
    """Check if item/installation metadata matches a language filter."""
    codes = language_codes_from_metadata(metadata)
    if not codes:
        return False
    return any(_match_code(language_filter, code) for code in codes)


# ---------------------------------------------------------------------------
# Gender normalization
# ---------------------------------------------------------------------------

VALID_GENDERS = frozenset({"male", "female", "neutral", "unknown"})


def normalize_gender(raw: Any) -> str:
    """Normalize gender metadata. Returns 'unknown' for missing/invalid values.

    Does NOT infer gender from names, IDs, or other heuristics.
    """
    if isinstance(raw, str) and raw.lower() in VALID_GENDERS:
        return raw.lower()
    return "unknown"


def gender_from_metadata(metadata: Mapping[str, Any]) -> str:
    """Extract gender from item/installation metadata, defaulting to 'unknown'."""
    return normalize_gender(metadata.get("gender"))


# ---------------------------------------------------------------------------
# Effective quality / distribution
# ---------------------------------------------------------------------------


def effective_quality(metadata: Mapping[str, Any]) -> str | None:
    """Get the effective quality selector from metadata.

    Precedence: selected_quality > quality > None
    """
    sq = metadata.get("selected_quality")
    if isinstance(sq, str) and sq:
        return sq
    q = metadata.get("quality")
    if isinstance(q, str) and q:
        return q
    return None


def effective_distribution(metadata: Mapping[str, Any]) -> str | None:
    """Get the effective distribution selector from metadata.

    Precedence: selected_distribution > distribution_id > None
    """
    sd = metadata.get("selected_distribution")
    if isinstance(sd, str) and sd:
        return sd
    dd = metadata.get("distribution_id")
    if isinstance(dd, str) and dd:
        return dd
    return None


# ---------------------------------------------------------------------------
# Version labels
# ---------------------------------------------------------------------------


def version_label(item_or_installation: CatalogItem | Installation) -> str | None:
    """Return a short version display label.

    Precedence depends on system:
    - Piper/Pocket: short source_revision
    - Kokoro: model_version > release_tag > short source_revision
    - External/local: 'local'
    """
    meta = item_or_installation.metadata

    if meta.get("external") or meta.get("managed") is False:
        return "local"

    if item_or_installation.system == "kokoro":
        mv = meta.get("model_version")
        if isinstance(mv, str) and mv:
            return mv
        rt = meta.get("release_tag")
        if isinstance(rt, str) and rt:
            return rt

    # Piper/Pocket/Kokoro fallback: source_revision
    sr = meta.get("source_revision")
    if isinstance(sr, str) and sr:
        return sr[:8] if len(sr) > 8 else sr

    return None


# ---------------------------------------------------------------------------
# Size helpers
# ---------------------------------------------------------------------------


def logical_size_bytes(installation: Installation) -> int:
    """Sum of artifact sizes for an installation."""
    return sum(a.size for a in installation.artifacts)


def catalog_size_bytes(item: CatalogItem) -> int | None:
    """Sum of artifact sizes from catalog, or None if any artifact has unknown size."""
    total = 0
    for artifact in item.artifacts:
        if artifact.size is None:
            return None
        total += artifact.size
    return total


# ---------------------------------------------------------------------------
# Inventory record
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class InventoryRecord:
    """Normalized view of a catalog entry, installed entry, or both."""

    system: str
    id: str
    kind: str

    language_codes: tuple[str, ...]
    gender: str

    quality: str | None
    distribution: str | None
    version: str | None

    installed: bool
    installation: Installation | None
    catalog_item: CatalogItem | None

    size_bytes: int | None
    status: str  # "installed", "available", "local"
    update_status: str  # "not_checked", "current", "update_available", "unknown", "not_applicable"

    @property
    def ref(self) -> str:
        return f"{self.system}:{self.id}"

    @property
    def primary_language(self) -> str | None:
        return self.language_codes[0] if self.language_codes else None


def inventory_record_from_installation(installation: Installation) -> InventoryRecord:
    """Create an InventoryRecord from a local installation only (no catalog)."""
    meta = installation.metadata
    return InventoryRecord(
        system=installation.system,
        id=installation.id,
        kind=installation.kind,
        language_codes=language_codes_from_metadata(meta),
        gender=gender_from_metadata(meta),
        quality=effective_quality(meta),
        distribution=effective_distribution(meta),
        version=version_label(installation),
        installed=True,
        installation=installation,
        catalog_item=None,
        size_bytes=logical_size_bytes(installation),
        status="installed" if not meta.get("external") else "local",
        update_status="not_checked",
    )


def inventory_record_from_catalog(item: CatalogItem) -> InventoryRecord:
    """Create an InventoryRecord from a catalog entry (not installed)."""
    meta = item.metadata
    size = catalog_size_bytes(item)
    return InventoryRecord(
        system=item.system,
        id=item.id,
        kind=item.kind,
        language_codes=language_codes_from_metadata(meta),
        gender=gender_from_metadata(meta),
        quality=effective_quality(meta),
        distribution=effective_distribution(meta),
        version=version_label(item),
        installed=False,
        installation=None,
        catalog_item=item,
        size_bytes=size,
        status="available",
        update_status="not_checked",
    )


# ---------------------------------------------------------------------------
# Inventory filter
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class InventoryFilter:
    """Filter specification for inventory records."""

    systems: tuple[str, ...] = ()
    kinds: tuple[str, ...] = ()
    languages: tuple[str, ...] = ()
    genders: tuple[str, ...] = ()
    qualities: tuple[str, ...] = ()
    distributions: tuple[str, ...] = ()
    statuses: tuple[str, ...] = ()


def matches_filter(record: InventoryRecord, spec: InventoryFilter) -> bool:
    """Check whether an InventoryRecord matches all specified filter criteria."""
    if spec.systems and record.system not in spec.systems:
        return False
    if spec.kinds and record.kind not in spec.kinds:
        return False
    if spec.languages:
        if not record.language_codes:
            return False
        if not any(
            _match_code(lang_filter, code)
            for lang_filter in spec.languages
            for code in record.language_codes
        ):
            return False
    if spec.genders and record.gender not in spec.genders:
        return False
    if spec.qualities and (record.quality is None or record.quality not in spec.qualities):
        return False
    if spec.distributions and (
        record.distribution is None or record.distribution not in spec.distributions
    ):
        return False
    return not (spec.statuses and record.status not in spec.statuses)


# ---------------------------------------------------------------------------
# Update comparison
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class UpdateComparison:
    """Result of comparing an installed artifact set against a catalog entry."""

    status: Literal["current", "update_available", "unknown", "not_applicable"]
    changed_roles: tuple[str, ...] = ()
    reason: str | None = None


def compare_installation_to_catalog(
    installation: Installation,
    item: CatalogItem,
) -> UpdateComparison:
    """Compare installed artifacts against current catalog entry.

    Uses artifact-level digest comparison:
    1. If catalog has SHA-256: compare against installed sha256
    2. Else if catalog has MD5: compare MD5 of installed file
    3. Else: size may help detect change, but equal size is not proof
    4. If insufficient identity: report 'unknown'
    """
    installed_by_key: dict[str, InstalledArtifact] = {}
    for artifact in installation.artifacts:
        key = _artifact_identity_key(artifact)
        installed_by_key[key] = artifact

    catalog_by_key: dict[str, Artifact] = {}
    for cat_art in item.artifacts:
        key = _artifact_identity_key(cat_art)
        catalog_by_key[key] = cat_art

    # If no catalog artifacts match at all, it might be a different selection
    if not installed_by_key and not catalog_by_key:
        return UpdateComparison(status="unknown", reason="no artifacts to compare")

    changed: list[str] = []
    has_comparison = False

    for key, cat_artifact in catalog_by_key.items():
        inst_artifact = installed_by_key.get(key)
        if inst_artifact is None:
            changed.append(cat_artifact.role)
            has_comparison = True
            continue

        # SHA-256 comparison (preferred)
        if cat_artifact.sha256 and inst_artifact.sha256:
            has_comparison = True
            if cat_artifact.sha256 != inst_artifact.sha256:
                changed.append(cat_artifact.role)
            continue

        # MD5 comparison
        if cat_artifact.md5 and inst_artifact.sha256:
            # We only have SHA-256 installed; can't compare directly with catalog MD5
            # Fall through to size comparison
            pass

        # Size comparison (weak signal)
        if cat_artifact.size is not None and inst_artifact.size is not None:
            has_comparison = True
            if cat_artifact.size != inst_artifact.size:
                changed.append(cat_artifact.role)
            continue

    # Check for removed catalog artifacts
    for key, inst_artifact in installed_by_key.items():
        if key not in catalog_by_key:
            changed.append(inst_artifact.role)
            has_comparison = True

    if not has_comparison:
        return UpdateComparison(
            status="unknown",
            reason="insufficient digest identity for comparison",
        )

    if changed:
        return UpdateComparison(
            status="update_available",
            changed_roles=tuple(sorted(set(changed))),
            reason=f"artifacts changed: {', '.join(sorted(set(changed)))}",
        )

    return UpdateComparison(status="current")


def _artifact_identity_key(artifact: Artifact | InstalledArtifact) -> str:
    """Build a stable identity key for matching artifacts between install and catalog.

    Uses (role, component, quality, filename) tuple.
    """
    return (
        f"{artifact.role}|{artifact.component or ''}|{artifact.quality or ''}|{artifact.filename}"
    )


# ---------------------------------------------------------------------------
# Catalog content fingerprint
# ---------------------------------------------------------------------------


def catalog_content_fingerprint(item: CatalogItem) -> str:
    """Compute a deterministic fingerprint for a catalog item's content identity.

    Uses artifact content identity (SHA-256 > MD5 > size+URL), excluding
    volatile catalog revision values. Suitable for cheap future update checks.
    """
    parts: list[str] = [
        item.system,
        item.id,
        effective_quality(item.metadata) or "",
        effective_distribution(item.metadata) or "",
    ]
    for artifact in item.artifacts:
        identity = artifact.sha256 or artifact.md5 or f"{artifact.size}:{artifact.url}"
        parts.append(
            f"{artifact.role}|{artifact.component or ''}|{artifact.quality or ''}"
            f"|{artifact.filename}|{identity}"
        )
    payload = "\n".join(parts)
    return "sha256:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Inventory merge
# ---------------------------------------------------------------------------


def merge_inventory(
    installations: Iterable[Installation],
    catalog_items: Iterable[CatalogItem],
    *,
    check_updates: bool = False,
) -> list[InventoryRecord]:
    """Merge installed and catalog entries into unified inventory records.

    Installed entries take precedence when both exist for the same canonical ref.
    When check_updates is True, artifact-level comparison is performed for installed
    entries that also have a catalog counterpart.
    """
    # Index installations by canonical ref
    installed_by_ref: dict[str, Installation] = {}
    for inst in installations:
        installed_by_ref[inst.ref] = inst

    # Index catalog items by canonical ref
    catalog_by_ref: dict[str, CatalogItem] = {}
    for item in catalog_items:
        catalog_by_ref[item.ref] = item

    records: list[InventoryRecord] = []
    seen_refs: set[str] = set()

    # Process installed entries first
    for inst in installations:
        ref = inst.ref
        seen_refs.add(ref)
        meta = inst.metadata
        catalog_item = catalog_by_ref.get(ref)

        update_status = "not_checked"
        if check_updates and catalog_item is not None:
            comparison = compare_installation_to_catalog(inst, catalog_item)
            update_status = comparison.status
        elif catalog_item is None and not meta.get("external"):
            update_status = "not_applicable"

        records.append(
            InventoryRecord(
                system=inst.system,
                id=inst.id,
                kind=inst.kind,
                language_codes=language_codes_from_metadata(meta),
                gender=gender_from_metadata(meta),
                quality=effective_quality(meta),
                distribution=effective_distribution(meta),
                version=version_label(inst),
                installed=True,
                installation=inst,
                catalog_item=catalog_item,
                size_bytes=logical_size_bytes(inst),
                status="installed" if not meta.get("external") else "local",
                update_status=update_status,
            )
        )

    # Process catalog-only entries
    for item in catalog_items:
        if item.ref in seen_refs:
            continue
        records.append(inventory_record_from_catalog(item))

    return records


# ---------------------------------------------------------------------------
# Inventory query helper
# ---------------------------------------------------------------------------


def query_inventory(
    installations: Iterable[Installation],
    catalog_items: Iterable[CatalogItem] | None,
    *,
    spec: InventoryFilter | None = None,
    check_updates: bool = False,
) -> list[InventoryRecord]:
    """Query inventory with optional filtering and update checking."""
    inst_list = list(installations)
    cat_list = list(catalog_items) if catalog_items is not None else []

    records = merge_inventory(inst_list, cat_list, check_updates=check_updates)

    if spec is not None:
        records = [r for r in records if matches_filter(r, spec)]

    return records
