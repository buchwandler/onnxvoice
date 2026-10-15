from __future__ import annotations

import builtins
import hashlib
import json
import os
import warnings
from collections.abc import Mapping, Sequence
from contextlib import suppress
from pathlib import Path
from typing import Any

from .catalog import CatalogClient, filter_items, parse_ref
from .checksums import digest_file
from .errors import AssetNotFoundError, NotInstalledError, RuntimeContractError, VoiceNotFoundError
from .inventory import (
    InventoryRecord,
    matches_language,
    voice_language_codes_from_catalog,
    voice_metadata_from_catalog,
)
from .store import AssetStore, ProgressCallback
from .systems import get_adapter
from .types import CatalogItem, Installation, InstalledArtifact, VoiceRecord
from .voices import iter_catalog_voices, parse_voice_ref

CatalogResults = list[CatalogItem] | list[Installation]
Installations = list[Installation]


class OnnxVoice:
    """High-level facade for catalogs, managed assets, and local runtimes.


    Parameters
    ----------
    cache_dir : str or pathlib.Path, optional
        Shared asset and catalog cache location.
    catalog_sources : dict, optional
        System-to-source overrides for catalog loading.
    offline : bool, default=False
        Read local/cache data without making network requests.
    """

    def __init__(
        self,
        *,
        cache_dir: str | Path | None = None,
        catalog_sources: dict[str, str] | None = None,
        offline: bool = False,
    ) -> None:
        """Create a manager using the selected cache and catalog configuration."""
        cache_root: Path | None = Path(cache_dir) if cache_dir is not None else None
        if cache_root is None:
            environment_cache_dir = os.environ.get("ONNXVOICE_CACHE_DIR")
            if environment_cache_dir:
                cache_root = Path(environment_cache_dir)
        self.store = AssetStore(cache_root, offline=offline)
        self.catalog = CatalogClient(
            cache_dir=cache_root,
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
        """List catalog items or installed assets for one system.


        Catalog listing may access the configured source when cached data is not usable.
        Set ``installed=True`` for local-only results.
        """
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

    def resolve_voice(
        self,
        ref: str,
        *,
        refresh: bool = False,
        progress: ProgressCallback | None = None,
    ) -> VoiceRecord:
        """Resolve a semantic voice reference against the normalized catalog."""
        parsed = parse_voice_ref(ref)
        item = self.catalog.resolve(parsed.backing_ref, refresh=refresh, progress=progress)

        if item.kind == "voice":
            if parsed.voice_id is not None and parsed.voice_id != item.id:
                raise VoiceNotFoundError(f"Voice {parsed.voice_id!r} is not declared by {item.ref}")
            voice_id = item.id
        else:
            if parsed.voice_id is None:
                raise VoiceNotFoundError(f"A voice ID is required for {item.ref}")
            if parsed.voice_id not in item.voices:
                raise VoiceNotFoundError(f"Voice {parsed.voice_id!r} is not declared by {item.ref}")
            voice_id = parsed.voice_id

        return VoiceRecord(
            catalog_item=item,
            voice_id=voice_id,
            metadata=voice_metadata_from_catalog(item, voice_id),
            languages=voice_language_codes_from_catalog(item, voice_id),
        )

    def list_voices(
        self,
        system: str | None = None,
        *,
        language: str | None = None,
        refresh: bool = False,
        progress: ProgressCallback | None = None,
    ) -> builtins.list[VoiceRecord]:
        """List current voices exposed by normalized catalog items."""
        systems = (system.casefold(),) if system is not None else self.catalog.systems()
        items: list[CatalogItem] = []
        for catalog_system in systems:
            items.extend(self.catalog.list(catalog_system, refresh=refresh, progress=progress))

        records: list[VoiceRecord] = []
        seen: set[tuple[str, str, str]] = set()
        for item, voice_id in iter_catalog_voices(items):
            key = (item.system, item.id, voice_id)
            if key in seen:
                continue
            seen.add(key)

            metadata = voice_metadata_from_catalog(item, voice_id)
            languages = voice_language_codes_from_catalog(item, voice_id)
            if language is not None and not matches_language(
                {"language_codes": languages}, language
            ):
                continue
            records.append(
                VoiceRecord(
                    catalog_item=item,
                    voice_id=voice_id,
                    metadata=metadata,
                    languages=languages,
                )
            )

        return sorted(records, key=lambda record: (record.system, record.asset_id, record.voice_id))

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
        """Resolve and install a catalog asset into the shared store.


        Catalog access or asset download can perform network I/O. Explicit quality
        and distribution selections are retained in the installation identity.
        """
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
        """Open and verify an existing managed installation without acquisition.


        Accepts an ``Installation`` or a reference already installed in this cache.
        Catalog lookup and network access are not performed.
        """
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
        adapter_kwargs: dict[str, Any] = {
            "providers": providers,
            "provider_options": provider_options,
            "session_options": session_options,
        }
        if installation.system == "pocket":
            adapter_kwargs["voice_cache_dir"] = self.store.root / "pocket-voice-states"
            adapter_kwargs["offline"] = self.store.offline
        return adapter(installation, **adapter_kwargs)

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
        voice_cache_dir: str | Path | None = None,
        offline: bool = False,
    ):
        """Open explicit files without copying or registering them in the store.


        Supply exactly one of ``model``, ``artifacts``, or ``files``. Named component
        mappings support multi-file runtimes; paths remain unmanaged.
        """
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

        local_artifacts: builtins.list[InstalledArtifact]
        qwen3_runtime: dict[str, Any] | None = None
        if system.lower() == "qwen3":
            runtime_input = runtime
            if runtime_input is None:
                runtime_input = (metadata or {}).get("runtime", {})
            local_artifacts, qwen3_runtime = OnnxVoice._qwen3_local_artifacts(
                merged_files,
                runtime=runtime_input,
                sample_rate=sample_rate,
                artifact_metadata=artifact_metadata,
            )
        else:
            local_artifacts = []
            model_components = {
                "duration_predictor",
                "text_encoder",
                "vector_estimator",
                "vocoder",
            }
            for key, raw_path in merged_files.items():
                component = None
                if system == "supertonic" and key in model_components:
                    role = "model"
                    component = key
                elif ":" in key:
                    role, _, component = key.partition(":")
                    if not role or not component or ":" in component:
                        if key.startswith("voice_style:"):
                            raise ValueError("voice_style keys must include a component name")
                        raise ValueError("Artifact keys must use non-empty role:component values")
                elif key in {"prosody", "curves", "decoder"}:
                    role = "model"
                    component = key
                else:
                    role = key
                local_artifacts.append(
                    OnnxVoice._local_artifact(
                        role,
                        raw_path,
                        component=component,
                        extra_metadata=(artifact_metadata or {}).get(key),
                    )
                )
        model_artifact = next(
            (
                artifact
                for artifact in local_artifacts
                if artifact.role == "model"
                and not (system == "kokoro" and artifact.component == "inno_voicepack")
            ),
            None,
        )
        if model_artifact is None:
            model_artifact = next(
                (artifact for artifact in local_artifacts if artifact.role == "model"), None
            )
        if system.lower() == "sopro" and model_artifact is None:
            model_artifact = next(
                (artifact for artifact in local_artifacts if artifact.role == "manifest"), None
            )
        if model_artifact is None:
            model_artifact = local_artifacts[0]
        merged_metadata: dict[str, Any] = {
            "external": True,
            "managed": False,
            "runtime": dict(runtime or {}),
            **(metadata or {}),
        }
        installation_system = system
        installation_sample_rate = sample_rate
        if qwen3_runtime is not None:
            merged_metadata["runtime"] = qwen3_runtime
            installation_system = "qwen3"
            installation_sample_rate = 24000
        installation = Installation(
            system=installation_system,
            id=f"local-{Path(model_artifact.filename).stem}",
            kind="external",
            path=model_artifact.path.parent,
            artifacts=tuple(local_artifacts),
            sample_rate=installation_sample_rate,
            metadata=merged_metadata,
        )
        adapter = get_adapter(system)
        adapter_kwargs: dict[str, Any] = {
            "providers": providers,
            "provider_options": provider_options,
            "session_options": session_options,
        }
        if system == "pocket":
            adapter_kwargs["voice_cache_dir"] = voice_cache_dir
            adapter_kwargs["offline"] = offline
        return adapter(installation, **adapter_kwargs)

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
        """List verified managed installations without catalog access."""
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
        """Import a model and optional config/voices files as a managed installation.


        This convenience method handles a single model plus optional companion files.
        Use ``open_local`` with a complete component mapping for multi-component runtimes.
        """
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
    @staticmethod
    def _local_artifact(
        role: str,
        raw_path: str | Path,
        *,
        component: str | None = None,
        quality: str | None = None,
        filename: str | None = None,
        extra_metadata: Mapping[str, Any] | None = None,
    ) -> InstalledArtifact:
        path = Path(raw_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        metadata: dict[str, Any] = {}
        if extra_metadata:
            metadata.update(extra_metadata)
        artifact_filename = filename or path.name
        return InstalledArtifact(
            role=role,
            filename=artifact_filename,
            path=path,
            sha256=digest_file(path),
            size=path.stat().st_size,
            quality=quality,
            component=component,
            format=Path(artifact_filename).suffix.lower().lstrip("."),
            metadata=metadata,
        )

    @staticmethod
    def _qwen3_local_artifacts(
        files: Mapping[str, str | Path],
        *,
        runtime: Any,
        sample_rate: int | None,
        artifact_metadata: Mapping[str, Mapping[str, Any]] | None = None,
    ) -> tuple[builtins.list[InstalledArtifact], dict[str, Any]]:
        """Normalize and validate unmanaged Qwen3 files without catalog/store access."""
        if not isinstance(runtime, Mapping):
            raise RuntimeContractError("Qwen3 local open requires runtime metadata")
        if runtime.get("layout") != "qwen3-wavekat-v1":
            raise RuntimeContractError("Qwen3 local runtime layout must be 'qwen3-wavekat-v1'")
        quality = runtime.get("quality")
        if not isinstance(quality, str) or quality not in {"fp32", "int4"}:
            raise RuntimeContractError("Qwen3 local runtime quality must be 'fp32' or 'int4'")
        runtime_rate = runtime.get("sample_rate", 24000)
        effective_rate = sample_rate if sample_rate is not None else runtime_rate
        if (
            isinstance(effective_rate, bool)
            or not isinstance(effective_rate, int)
            or effective_rate != 24000
            or runtime_rate != 24000
        ):
            raise RuntimeContractError("Qwen3 WaveKat local bundles require sample_rate=24000")

        tokenizer_files = {
            "tokenizer_json": "tokenizer.json",
            "vocab": "vocab.json",
            "merges": "merges.txt",
            "tokenizer_config": "tokenizer_config.json",
            "special_tokens": "special_tokens_map.json",
            "added_tokens": "added_tokens.json",
        }
        generation_components = {"talker_prefill", "talker_decode", "code_predictor", "vocoder"}
        encoder_components = {"speaker_encoder", "tokenizer_encoder"}
        embedding_components = {
            "text_embedding",
            "text_projection_fc1_weight",
            "text_projection_fc1_bias",
            "text_projection_fc2_weight",
            "text_projection_fc2_bias",
            "talker_codec_embedding",
            *(f"cp_codec_embedding_{index}" for index in range(15)),
        }
        optional_embeddings = {"small_to_mtp_projection_bias", "small_to_mtp_projection_weight"}

        parsed: dict[tuple[str, str | None], tuple[str | Path, str]] = {}
        for key, path in files.items():
            if not isinstance(key, str) or not key:
                raise RuntimeContractError("Qwen3 local artifact keys must be non-empty strings")
            if key == "config":
                role, component = "config", None
            else:
                parts = key.split(":")
                if len(parts) != 2 or not all(parts):
                    raise RuntimeContractError(f"Invalid Qwen3 local artifact key {key!r}")
                role, component = parts
                if role not in {"tokenizer", "embedding", "model", "model_data"}:
                    raise RuntimeContractError(f"Unsupported Qwen3 local artifact role {role!r}")
            identity = (role, component)
            if identity in parsed:
                raise RuntimeContractError(f"Duplicate Qwen3 local artifact {key!r}")
            parsed[identity] = (path, key)

        encoder_models = {
            component
            for role, component in parsed
            if role == "model" and component in encoder_components
        }
        encoder_data = {
            component
            for role, component in parsed
            if role == "model_data" and component in encoder_components
        }
        if encoder_models or encoder_data:
            if encoder_models != encoder_components or encoder_data != encoder_components:
                raise RuntimeContractError(
                    "Qwen3 Base local bundle requires both encoder graphs and data files"
                )
            inferred_model_type = "base"
        else:
            inferred_model_type = "voice_design"
        model_type = runtime.get("model_type")
        if (
            not isinstance(model_type, str)
            or model_type not in {"base", "voice_design"}
            or model_type != inferred_model_type
        ):
            raise RuntimeContractError(
                "Qwen3 local runtime model_type must be 'base' or 'voice_design' and match its encoder inventory"
            )

        expected_models = set(generation_components)
        if model_type == "base":
            expected_models.update(encoder_components)
        expected: set[tuple[str, str | None]] = {("config", None)}
        expected.update(("tokenizer", component) for component in tokenizer_files)
        expected.update(("embedding", component) for component in embedding_components)
        expected.update(("model", component) for component in expected_models)
        expected.update(("model_data", component) for component in expected_models)
        actual = set(parsed)
        allowed = expected | {("embedding", component) for component in optional_embeddings}
        missing = expected - actual
        extra_keys = actual - allowed
        if missing or extra_keys:
            raise RuntimeContractError(
                f"Incomplete Qwen3 local artifact inventory; missing={sorted(missing, key=repr)}, extra={sorted(extra_keys, key=repr)}"
            )

        artifacts: list[InstalledArtifact] = []
        for (role, component), (path, key) in parsed.items():
            if role == "config":
                relative_path = "config.json"
                artifact_quality = None
            elif role == "tokenizer":
                if component not in tokenizer_files:
                    raise RuntimeContractError(f"Unknown Qwen3 tokenizer component {component!r}")
                relative_path = f"tokenizer/{tokenizer_files[component]}"
                artifact_quality = None
            elif role == "embedding":
                if component not in embedding_components | optional_embeddings:
                    raise RuntimeContractError(f"Unknown Qwen3 embedding component {component!r}")
                relative_path = f"embeddings/{component}.npy"
                artifact_quality = None
            else:
                if component not in expected_models:
                    raise RuntimeContractError(f"Unknown Qwen3 model component {component!r}")
                shared_encoder = component in encoder_components
                graph_path = (
                    f"{component}.onnx" if shared_encoder else f"{quality}/{component}.onnx"
                )
                relative_path = graph_path + (".data" if role == "model_data" else "")
                artifact_quality = None if shared_encoder else quality
            raw_extra = (artifact_metadata or {}).get(key, {})
            if not isinstance(raw_extra, Mapping):
                raise RuntimeContractError(f"Qwen3 artifact metadata for {key!r} must be an object")
            extra_metadata = dict(raw_extra)
            extra_metadata["path"] = relative_path
            artifacts.append(
                OnnxVoice._local_artifact(
                    role,
                    path,
                    component=component,
                    quality=artifact_quality,
                    filename=relative_path,
                    extra_metadata=extra_metadata,
                )
            )

        by_identity = {(artifact.role, artifact.component): artifact for artifact in artifacts}
        for component in expected_models:
            graph = by_identity[("model", component)]
            data = by_identity[("model_data", component)]
            expected_graph_name = f"{component}.onnx"
            if (
                graph.path.name != expected_graph_name
                or data.path.name != expected_graph_name + ".data"
            ):
                raise RuntimeContractError(
                    f"Qwen3 local graph and external data filenames disagree for {component!r}"
                )
            if data.path.resolve() != Path(str(graph.path.resolve()) + ".data"):
                raise RuntimeContractError(
                    f"Qwen3 local external data must be adjacent to its {component!r} graph"
                )

        normalized_runtime = {
            **runtime,
            "layout": "qwen3-wavekat-v1",
            "model_type": model_type,
            "quality": quality,
            "sample_rate": 24000,
        }
        return artifacts, normalized_runtime

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
