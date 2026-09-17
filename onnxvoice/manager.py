from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from .catalog import CatalogClient, filter_items, parse_ref
from .store import AssetStore
from .systems import get_adapter
from .types import CatalogItem, Installation


class OnnxVoice:
    """High-level catalog, store and runtime facade."""

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
    ) -> list[CatalogItem] | list[Installation]:
        if installed:
            return self.store.installed(system)
        if system is None:
            raise ValueError("system is required unless installed=True")
        return filter_items(
            self.catalog.list(system, refresh=refresh),
            language=language,
            quality=quality,
        )

    def install(
        self,
        ref: str,
        *,
        quality: str | None = None,
        refresh: bool = False,
        force: bool = False,
    ) -> Installation:
        item = self.catalog.resolve(ref, refresh=refresh, quality=quality)
        return self.store.install(item, force=force)

    def resolve(
        self,
        ref: str,
        *,
        quality: str | None = None,
        download: bool = True,
        refresh: bool = False,
    ) -> Installation:
        system, item_id = parse_ref(ref)
        if self.store.is_installed(system, item_id):
            installation = self.store.get(system, item_id)
            self.store.verify(installation)
            if quality is None or installation.metadata.get("selected_quality") == quality:
                return installation
        if not download:
            raise FileNotFoundError(f"Not installed: {ref}")
        return self.install(ref, quality=quality, refresh=refresh, force=quality is not None)

    def installed(self, system: str | None = None) -> list[Installation]:
        return self.store.installed(system)

    def where(self, ref: str) -> Path:
        system, item_id = parse_ref(ref)
        return self.store.where(system, item_id)

    def remove(self, ref: str) -> None:
        system, item_id = parse_ref(ref)
        self.store.remove(system, item_id)

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
        )

    def load(
        self,
        ref: str | Installation,
        *,
        quality: str | None = None,
        download: bool = True,
        providers: Sequence[str] | None = None,
        provider_options: Sequence[dict[str, Any]] | None = None,
    ):
        installation = (
            ref
            if isinstance(ref, Installation)
            else self.resolve(ref, quality=quality, download=download)
        )
        adapter = get_adapter(installation.system)
        return adapter(
            installation,
            providers=providers,
            provider_options=provider_options,
        )
