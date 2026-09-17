from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from .catalog import CatalogClient, filter_items, parse_ref
from .checksums import digest_file
from .store import AssetStore, ProgressCallback
from .systems import get_adapter
from .types import CatalogItem, Installation, InstalledArtifact


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
        progress: ProgressCallback | None = None,
    ) -> list[CatalogItem] | list[Installation]:
        if installed:
            return self.store.installed(system)
        if system is None:
            raise ValueError("system is required unless installed=True")
        return filter_items(
            self.catalog.list(system, refresh=refresh, progress=progress),
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
        progress: ProgressCallback | None = None,
    ) -> Installation:
        item = self.catalog.resolve(ref, refresh=refresh, quality=quality, progress=progress)
        return self.store.install(item, force=force, progress=progress)

    def resolve(
        self,
        ref: str,
        *,
        quality: str | None = None,
        download: bool = True,
        refresh: bool = False,
        progress: ProgressCallback | None = None,
    ) -> Installation:
        system, item_id = parse_ref(ref)
        if self.store.is_installed(system, item_id):
            installation = self.store.get(system, item_id)
            self.store.verify(installation)
            if quality is None or installation.metadata.get("selected_quality") == quality:
                return installation
        if not download:
            raise FileNotFoundError(f"Not installed: {ref}")
        return self.install(
            ref,
            quality=quality,
            refresh=refresh,
            progress=progress,
            force=quality is not None,
        )

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
        download: bool = True,
        providers: str | Sequence[str] | None = None,
        provider: str | None = None,
        provider_options: Sequence[dict[str, Any]] | dict[str, dict[str, Any]] | None = None,
    ):
        installation = (
            ref
            if isinstance(ref, Installation)
            else self.resolve(ref, quality=quality, download=download)
        )
        if provider is not None:
            if providers is not None:
                raise ValueError("Use provider or providers, not both")
            providers = provider
        adapter = get_adapter(installation.system)
        return adapter(
            installation,
            providers=providers,
            provider_options=provider_options,
        )

    @staticmethod
    def load_local(
        *,
        system: str,
        model: str | Path,
        config: str | Path | None = None,
        voices: str | Path | None = None,
        sample_rate: int | None = None,
        providers: str | Sequence[str] | None = None,
        provider: str | None = None,
        provider_options: Sequence[dict[str, Any]] | dict[str, dict[str, Any]] | None = None,
    ):
        """Load local assets without registering or copying them into the cache."""
        if provider is not None:
            if providers is not None:
                raise ValueError("Use provider or providers, not both")
            providers = provider
        artifacts = [OnnxVoice._local_artifact("model", model)]
        if config is not None:
            artifacts.append(OnnxVoice._local_artifact("config", config))
        if voices is not None:
            artifacts.append(OnnxVoice._local_artifact("voices", voices))
        model_path = Path(model).expanduser().resolve()
        installation = Installation(
            system=system,
            id=f"local-{model_path.stem}",
            kind="external",
            path=model_path.parent,
            artifacts=tuple(artifacts),
            sample_rate=sample_rate,
            metadata={"external": True, "managed": False},
        )
        adapter = get_adapter(system)
        return adapter(
            installation,
            providers=providers,
            provider_options=provider_options,
        )

    @staticmethod
    def _local_artifact(role: str, raw_path: str | Path) -> InstalledArtifact:
        path = Path(raw_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        return InstalledArtifact(
            role=role,
            filename=path.name,
            path=path,
            sha256=digest_file(path),
            size=path.stat().st_size,
        )
