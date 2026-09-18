from __future__ import annotations

import hashlib
import warnings
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .catalog import CatalogClient, filter_items, parse_ref
from .checksums import digest_file
from .errors import AssetNotFoundError, NotInstalledError
from .store import AssetStore, ProgressCallback
from .systems import get_adapter
from .types import CatalogItem, Installation, InstalledArtifact

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
        selected_distribution = distribution or item.metadata.get("distribution_id")
        if isinstance(selected_distribution, str):
            item = self._with_distribution_identity(
                item, selected_distribution, cache=distribution is not None
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
        storage_id = self._distribution_cache_id(item_id, distribution)
        try:
            installation = self.store.get(system, storage_id)
        except AssetNotFoundError as exc:
            detail = f" with distribution={distribution!r}" if distribution else ""
            raise NotInstalledError(
                f"Not installed: {ref}{detail}. Run onnxvoice.install(...) first."
            ) from exc
        self.store.verify(installation)
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
    def _distribution_cache_id(item_id: str, distribution: str | None) -> str:
        if distribution is None:
            return item_id
        digest = hashlib.sha256(distribution.encode("utf-8")).hexdigest()[:16]
        return f"{item_id}--dist-{digest}"

    @classmethod
    def _with_distribution_identity(
        cls, item: CatalogItem, distribution: str, *, cache: bool = True
    ) -> CatalogItem:
        metadata = {**item.metadata, "selected_distribution": distribution}
        if cache:
            metadata["cache_id"] = cls._distribution_cache_id(item.id, distribution)
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
