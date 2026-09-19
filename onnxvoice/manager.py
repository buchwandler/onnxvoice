from __future__ import annotations

import builtins
import hashlib
import json
import warnings
from collections.abc import Mapping, Sequence
from contextlib import suppress
from pathlib import Path
from typing import Any

from .catalog import CatalogClient, filter_items, parse_ref
from .checksums import digest_file
from .errors import AssetNotFoundError, NotInstalledError
from .inventory import InventoryRecord
from .store import AssetStore, ProgressCallback
from .systems import get_adapter
from .types import CatalogItem, Installation, InstalledArtifact
from .voice_selectors import (
    VoiceIdentity,
    VoiceRecord,
    catalog_voice_keys,
    iter_voice_identities,
    make_voice_record,
    selector_for_voice,
)
from .voice_selectors import resolve_voice_selector as _resolve_voice_selector

CatalogResults = list[CatalogItem] | list[Installation]
Installations = list[Installation]


class OnnxVoice:
    """High-level catalog, store and local runtime facade."""

    def __init__(
        self,
        *,
        cache_dir: str | Path | None = None,
        catalog_sources: dict[str, str] | None = None,
        offline: bool = False,
    ) -> None:
        self.store = AssetStore(cache_dir, offline=offline)
        self.catalog = CatalogClient(
            cache_dir=Path(cache_dir) if cache_dir is not None else None,
            sources=catalog_sources,
            offline=offline,
        )

    def list(
        self,
        system: str | None = None,
        *,
        installed: bool = False,
        language: str | None = None,
        quality: str | None = None,
        refresh: bool = False,
        progress: ProgressCallback | None = None,
    ) -> CatalogResults:
        from .inventory import matches_language

        if installed:
            installations = self.store.installed(system)
            result: list[Installation] = installations
            if language:
                result = [inst for inst in result if matches_language(inst.metadata, language)]
            if quality:
                result = [
                    inst
                    for inst in result
                    if inst.metadata.get("selected_quality") == quality
                    or inst.metadata.get("quality") == quality
                    or any(a.quality == quality for a in inst.artifacts)
                ]
            return result
        if system is None:
            raise ValueError("system is required unless installed=True")
        return filter_items(
            self.catalog.list(system, refresh=refresh, progress=progress),
            language=language,
            quality=quality,
        )

    def resolve_voice_selector(
        self,
        value: str,
        *,
        include_retired: bool = False,
    ) -> VoiceIdentity:
        """Resolve a stable short selector without opening or selecting an asset."""
        return _resolve_voice_selector(value, include_retired=include_retired)

    def list_voices(
        self,
        system: str | None = None,
        *,
        language: str | None = None,
        refresh: bool = False,
        include_retired: bool = False,
        include_unassigned: bool = True,
        progress: ProgressCallback | None = None,
    ) -> list[VoiceRecord]:
        """List flattened catalog voices joined to stable selector identities."""
        from .inventory import language_codes_from_metadata, matches_language
        from .voice_selectors import get_voice_selector_registry

        systems = (system.casefold(),) if system is not None else ("kokoro", "piper")
        unsupported = set(systems) - {"kokoro", "piper"}
        if unsupported:
            raise ValueError(
                f"voice selectors are not available for: {', '.join(sorted(unsupported))}"
            )

        items: list[CatalogItem] = []
        for catalog_system in systems:
            items.extend(self.catalog.list(catalog_system, refresh=refresh, progress=progress))
        registry = get_voice_selector_registry()
        records: list[VoiceRecord] = []
        seen: set[tuple[str, str, str]] = set()

        for item, asset_id, voice_id in catalog_voice_keys(items):
            key = (item.system, asset_id, voice_id)
            if key in seen:
                continue
            seen.add(key)
            identity = selector_for_voice(
                system=item.system,
                asset_id=asset_id,
                voice_id=voice_id,
                include_retired=True,
                registry=registry,
            )
            if identity is not None and identity.state == "retired" and not include_retired:
                continue
            if identity is None and not include_unassigned:
                continue
            metadata = item.metadata
            if language is not None and not matches_language(metadata, language):
                continue
            records.append(
                make_voice_record(
                    identity,
                    available=identity is not None and identity.state == "active",
                    catalog_item=item,
                    languages=language_codes_from_metadata(metadata),
                    gender=str(metadata.get("gender") or "unknown")
                    if item.system == "piper"
                    else "unknown",
                    voice_id=voice_id,
                )
            )

        for identity in iter_voice_identities(
            system=system, include_retired=include_retired, registry=registry
        ):
            if identity.canonical_key in seen:
                continue
            if language is not None and identity.language != language.casefold().replace("-", "_"):
                continue
            records.append(
                make_voice_record(
                    identity,
                    available=False,
                    catalog_item=None,
                    languages=(identity.language,),
                    gender="unknown",
                )
            )

        return sorted(
            records,
            key=lambda record: (
                record.selector is None,
                record.selector or "",
                record.system or "",
                record.asset_id or "",
                record.voice_id or "",
            ),
        )

    def install(
        self,
        ref: str,
        *,
        quality: str | None = None,
        distribution: str | None = None,
        refresh: bool = False,
        force: bool = False,
        progress: ProgressCallback | None = None,
    ) -> Installation:
        item = self.catalog.resolve(
            ref,
            refresh=refresh,
            quality=quality,
            distribution=distribution,
            progress=progress,
        )
        # Apply selection identity when explicit selectors were provided.
        # Only *explicit* selectors (passed by the caller) affect the
        # storage identity; catalog-chosen defaults do not.
        item = self._with_selection_identity(
            item,
            quality=quality if quality is not None else None,
            distribution=distribution if distribution is not None else None,
        )
        return self.store.install(item, force=force, progress=progress)

    def open(
        self,
        ref: str | Installation,
        *,
        quality: str | None = None,
        distribution: str | None = None,
        providers: str | Sequence[str] | None = None,
        provider: str | None = None,
        provider_options: Sequence[dict[str, Any]] | dict[str, dict[str, Any]] | None = None,
        session_options: Any | None = None,
    ):
        """Open an existing local installation without catalog or network access."""
        installation = (
            ref
            if isinstance(ref, Installation)
            else self.resolve(ref, quality=quality, distribution=distribution)
        )
        self.store.verify(installation)
        if provider is not None:
            if providers is not None:
                raise ValueError("Use provider or providers, not both")
            providers = provider
        adapter = get_adapter(installation.system)
        return adapter(
            installation,
            providers=providers,
            provider_options=provider_options,
            session_options=session_options,
        )

    @staticmethod
    def open_local(
        *,
        system: str,
        model: str | Path | None = None,
        artifacts: Mapping[str, str | Path] | None = None,
        config: str | Path | None = None,
        voices: str | Path | None = None,
        files: Mapping[str, str | Path] | None = None,
        artifact_metadata: Mapping[str, Mapping[str, Any]] | None = None,
        metadata: Mapping[str, Any] | None = None,
        runtime: Mapping[str, Any] | None = None,
        sample_rate: int | None = None,
        providers: str | Sequence[str] | None = None,
        provider: str | None = None,
        provider_options: Sequence[dict[str, Any]] | dict[str, dict[str, Any]] | None = None,
        session_options: Any | None = None,
    ):
        """Open explicit local files without registering or copying them."""
        if provider is not None:
            if providers is not None:
                raise ValueError("Use provider or providers, not both")
            providers = provider
        if artifacts is not None and model is not None:
            raise ValueError("Use model or artifacts, not both")
        if files is not None and model is not None:
            raise ValueError("Use model or files, not both")
        if files is not None and artifacts is not None:
            raise ValueError("Use artifacts or files, not both")
        if files is not None:
            if config is not None:
                raise ValueError("config must be included in files")
            if voices is not None:
                raise ValueError("voices must be included in files")
            merged_files: dict[str, str | Path] = dict(files)
        elif artifacts is not None:
            if config is not None or voices is not None:
                raise ValueError("config and voices must be included in artifacts")
            merged_files = dict(artifacts)
        else:
            if model is None:
                raise ValueError("model, artifacts, or files is required")
            merged_files = {"model": model}
            if config is not None:
                merged_files["config"] = config
            if voices is not None:
                merged_files["voices"] = voices

        local_artifacts: list[InstalledArtifact] = []
        for key, raw_path in merged_files.items():
            if key in {"prosody", "curves", "decoder"}:
                local_artifacts.append(
                    OnnxVoice._local_artifact(
                        "model",
                        raw_path,
                        component=key,
                        extra_metadata=(artifact_metadata or {}).get(key),
                    )
                )
            else:
                local_artifacts.append(
                    OnnxVoice._local_artifact(
                        key,
                        raw_path,
                        extra_metadata=(artifact_metadata or {}).get(key),
                    )
                )
        model_artifact = next(
            (artifact for artifact in local_artifacts if artifact.role == "model"), None
        )
        if model_artifact is None:
            model_artifact = local_artifacts[0]
        merged_metadata: dict[str, Any] = {
            "external": True,
            "managed": False,
            "runtime": dict(runtime or {}),
            **(metadata or {}),
        }
        installation = Installation(
            system=system,
            id=f"local-{Path(model_artifact.filename).stem}",
            kind="external",
            path=model_artifact.path.parent,
            artifacts=tuple(local_artifacts),
            sample_rate=sample_rate,
            metadata=merged_metadata,
        )
        adapter = get_adapter(system)
        return adapter(
            installation,
            providers=providers,
            provider_options=provider_options,
            session_options=session_options,
        )

    def resolve(
        self,
        ref: str,
        *,
        quality: str | None = None,
        distribution: str | None = None,
        download: bool | None = None,
        refresh: bool | None = None,
        progress: ProgressCallback | None = None,
    ) -> Installation:
        """Resolve and verify an existing installation only."""
        if download:
            raise ValueError("resolve() is local-only; call install() before open()")
        if refresh:
            raise ValueError("resolve() is local-only; catalog refresh belongs to install()")
        if progress is not None:
            raise ValueError("resolve() is local-only and does not accept progress callbacks")
        system, item_id = parse_ref(ref)
        storage_id = self._selection_storage_id(item_id, quality=quality, distribution=distribution)
        try:
            installation = self.store.get(system, storage_id)
        except AssetNotFoundError as exc:
            parts = []
            if distribution is not None:
                parts.append(f"distribution={distribution!r}")
            if quality is not None:
                parts.append(f"quality={quality!r}")
            detail = f" with {', '.join(parts)}" if parts else ""
            raise NotInstalledError(
                f"Not installed: {ref}{detail}. Run onnxvoice.install(...) first."
            ) from exc
        self.store.verify(installation)
        # Validate that the stored selection matches the requested one
        if quality is not None and installation.metadata.get("selected_quality") != quality:
            raise NotInstalledError(
                f"Not installed: {ref} with quality={quality!r}. Run onnxvoice.install(...) first."
            )
        if (
            distribution is not None
            and installation.metadata.get("selected_distribution") != distribution
        ):
            raise NotInstalledError(
                f"Not installed: {ref} with distribution={distribution!r}. Run onnxvoice.install(...) first."
            )
        return installation

    def installed(self, system: str | None = None) -> Installations:
        return self.store.installed(system)

    def find_installed(self, ref: str) -> builtins.list[Installation]:
        """Find all installed variants matching a canonical ref.

        Returns installations whose system and canonical id match.
        """
        from .catalog import parse_ref

        system, item_id = parse_ref(ref)
        result = []
        for inst in self.store.installed(system):
            if inst.id == item_id:
                result.append(inst)
        return result

    def inventory(
        self,
        *,
        system: str | None = None,
        language: str | None = None,
        gender: str | None = None,
        kind: str | None = None,
        quality: str | None = None,
        distribution: str | None = None,
        status: str | None = None,
        installed_only: bool = False,
        refresh: bool = False,
        check_updates: bool = False,
        progress: ProgressCallback | None = None,
    ) -> builtins.list[InventoryRecord]:
        """Query merged inventory with optional filtering and update checking."""
        from .inventory import InventoryFilter, query_inventory

        installations = self.store.installed(system)
        catalog_items = None
        if not installed_only:
            try:
                if system:
                    catalog_items = self.catalog.list(system, refresh=refresh, progress=progress)
                else:
                    all_items = []
                    for sys in self.catalog.systems():
                        with suppress(Exception):
                            all_items.extend(
                                self.catalog.list(sys, refresh=refresh, progress=progress)
                            )  # Partial catalog failure is OK
                    catalog_items = all_items
            except Exception:
                if installed_only:
                    catalog_items = []
                else:
                    raise

        if catalog_items is None:
            catalog_items = []

        # Build filter spec
        statuses: tuple[str, ...] = ()
        if status:
            statuses = (status,)

        spec = InventoryFilter(
            systems=(system,) if system else (),
            kinds=(kind,) if kind else (),
            languages=(language,) if language else (),
            genders=(gender,) if gender else (),
            qualities=(quality,) if quality else (),
            distributions=(distribution,) if distribution else (),
            statuses=statuses,
        )

        return query_inventory(
            installations,
            catalog_items,
            spec=spec,
            check_updates=check_updates,
        )

    def check_updates(
        self,
        *,
        system: str | None = None,
        language: str | None = None,
        kind: str | None = None,
        refresh: bool = True,
        progress: ProgressCallback | None = None,
    ) -> builtins.list[InventoryRecord]:
        """Check for available updates. Refreshes catalogs by default."""
        return self.inventory(
            system=system,
            language=language,
            kind=kind,
            refresh=refresh,
            check_updates=True,
            progress=progress,
        )

    def update(
        self,
        ref: str,
        *,
        quality: str | None = None,
        distribution: str | None = None,
        force: bool = False,
        refresh: bool = True,
        progress: ProgressCallback | None = None,
    ) -> Installation:
        """Update an existing installation with fresh catalog data.

        Preserves the current effective quality/distribution selection.
        Atomically replaces the installation.
        """
        # Find existing installation
        existing = self.resolve(ref, quality=quality, distribution=distribution)

        # Preserve existing selection
        eff_quality = quality or existing.metadata.get("selected_quality")
        eff_distribution = distribution or existing.metadata.get("selected_distribution")

        # Resolve fresh catalog entry with same selection
        item = self.catalog.resolve(
            ref,
            refresh=refresh,
            quality=eff_quality,
            distribution=eff_distribution,
            progress=progress,
        )
        item = self._with_selection_identity(
            item,
            quality=eff_quality,
            distribution=eff_distribution,
        )

        # Atomic replace
        return self.store.replace(existing, item, progress=progress)

    def where(
        self,
        ref: str,
        *,
        quality: str | None = None,
        distribution: str | None = None,
    ) -> Path:
        """Return the installation path for a selected installation."""
        return self.resolve(ref, quality=quality, distribution=distribution).path

    def remove(
        self,
        ref: str,
        *,
        quality: str | None = None,
        distribution: str | None = None,
    ) -> None:
        """Remove a selected installation."""
        system, item_id = parse_ref(ref)
        storage_id = self._selection_storage_id(item_id, quality=quality, distribution=distribution)
        self.store.remove(system, storage_id)

    def verify(
        self,
        ref: str,
        *,
        quality: str | None = None,
        distribution: str | None = None,
    ) -> None:
        """Verify checksums for a selected installation."""
        installation = self.resolve(ref, quality=quality, distribution=distribution)
        self.store.verify(installation)

    def show(
        self,
        ref: str,
        *,
        quality: str | None = None,
        distribution: str | None = None,
    ) -> Installation:
        """Return a selected installation (resolves and verifies)."""
        return self.resolve(ref, quality=quality, distribution=distribution)

    def import_model(
        self,
        *,
        system: str,
        item_id: str,
        model: str | Path,
        config: str | Path | None = None,
        voices: str | Path | None = None,
        sample_rate: int | None = None,
        force: bool = False,
        progress: ProgressCallback | None = None,
    ) -> Installation:
        files: dict[str, str | Path] = {"model": model}
        if config is not None:
            files["config"] = config
        if voices is not None:
            files["voices"] = voices
        return self.store.import_files(
            system=system,
            item_id=item_id,
            files=files,
            sample_rate=sample_rate,
            force=force,
            progress=progress,
        )

    def load(
        self,
        ref: str | Installation,
        *,
        quality: str | None = None,
        distribution: str | None = None,
        download: bool | None = None,
        providers: str | Sequence[str] | None = None,
        provider: str | None = None,
        provider_options: Sequence[dict[str, Any]] | dict[str, dict[str, Any]] | None = None,
        session_options: Any | None = None,
    ):
        """Deprecated alias for :meth:`open` with no acquisition behavior."""
        warnings.warn("load() is deprecated; use open()", DeprecationWarning, stacklevel=2)
        if download:
            raise ValueError("load(download=True) is obsolete; call install() before open()")
        return self.open(
            ref,
            quality=quality,
            distribution=distribution,
            providers=providers,
            provider=provider,
            provider_options=provider_options,
            session_options=session_options,
        )

    @staticmethod
    def load_local(**kwargs: Any):
        """Deprecated alias for :meth:`open_local`."""
        warnings.warn(
            "load_local() is deprecated; use open_local()", DeprecationWarning, stacklevel=2
        )
        return OnnxVoice.open_local(**kwargs)

    @staticmethod
    def _local_artifact(
        role: str,
        raw_path: str | Path,
        *,
        component: str | None = None,
        extra_metadata: Mapping[str, Any] | None = None,
    ) -> InstalledArtifact:
        path = Path(raw_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        metadata: dict[str, Any] = {}
        if extra_metadata:
            metadata.update(extra_metadata)
        return InstalledArtifact(
            role=role,
            filename=path.name,
            path=path,
            sha256=digest_file(path),
            size=path.stat().st_size,
            component=component,
            metadata=metadata,
        )

    @staticmethod
    def _selection_storage_id(
        item_id: str,
        *,
        quality: str | None = None,
        distribution: str | None = None,
    ) -> str:
        """Compute deterministic storage identity from explicit selectors.

        Returns the canonical *item_id* when no explicit selectors are
        given, preserving the stable default installation path.  When any
        explicit selector is present the result is *item_id* plus a
        ``--sel-<digest>`` suffix derived from the canonical JSON of the
        selector tuple.
        """
        if quality is None and distribution is None:
            return item_id
        payload = json.dumps(
            {"distribution": distribution, "quality": quality},
            sort_keys=True,
            separators=(",", ":"),
        )
        digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
        return f"{item_id}--sel-{digest}"

    @classmethod
    def _with_selection_identity(
        cls,
        item: CatalogItem,
        *,
        quality: str | None = None,
        distribution: str | None = None,
    ) -> CatalogItem:
        """Return a copy of *item* annotated with explicit selection metadata.

        When *distribution* or *quality* is non-None the returned item's
        metadata includes ``selected_distribution``, ``selected_quality``,
        and ``cache_id`` so the store can address the correct physical
        installation.
        """
        metadata = dict(item.metadata)
        if distribution is not None:
            metadata["selected_distribution"] = distribution
        if quality is not None:
            metadata["selected_quality"] = quality
        storage_id = cls._selection_storage_id(item.id, quality=quality, distribution=distribution)
        if storage_id != item.id:
            metadata["cache_id"] = storage_id
        return CatalogItem(
            system=item.system,
            id=item.id,
            kind=item.kind,
            artifacts=item.artifacts,
            aliases=item.aliases,
            sample_rate=item.sample_rate,
            voices=item.voices,
            default_voice=item.default_voice,
            metadata=metadata,
        )
