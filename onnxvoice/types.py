from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np

MANIFEST_SCHEMA_VERSION = 3


def validate_safe_component(value: str, *, field_name: str) -> str:
    """Validate a single filesystem path component."""
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field_name} must be a non-empty string")
    if value in {".", ".."} or "/" in value or "\\" in value or "\x00" in value:
        raise ValueError(f"Unsafe {field_name}: {value!r}")
    if Path(value).is_absolute() or Path(value).name != value:
        raise ValueError(f"Unsafe {field_name}: {value!r}")
    return value


def validate_relative_path(value: str, *, field_name: str) -> str:
    """Validate an artifact path relative to an installation root."""
    if not isinstance(value, str) or not value or "\x00" in value or "\\" in value:
        raise ValueError(f"Unsafe {field_name}: {value!r}")
    path = Path(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError(f"Unsafe {field_name}: {value!r}")
    for part in path.parts:
        validate_safe_component(part, field_name=field_name)
    return value


@dataclass(frozen=True, slots=True)
class Artifact:
    """Remote or local artifact metadata belonging to a catalog item.


    ``role`` identifies the artifact's purpose. ``component`` and ``quality`` disambiguate
    multi-file and variant layouts; checksums and size are optional catalog pins.
    """

    role: str
    filename: str
    url: str | None = None
    size: int | None = None
    sha256: str | None = None
    md5: str | None = None
    quality: str | None = None
    component: str | None = None
    format: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class CatalogItem:
    """Normalized installable item from a system catalog.


    ``ref`` is the canonical ``system:id`` reference. ``artifacts`` describe the files;
    ``voices`` and ``default_voice`` describe catalog voice identities when applicable.
    """

    system: str
    id: str
    kind: str
    artifacts: tuple[Artifact, ...]
    aliases: tuple[str, ...] = ()
    sample_rate: int | None = None
    voices: tuple[str, ...] = ()
    default_voice: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def timing_output(self) -> str | None:
        """Return the declared auxiliary timing output, if valid."""
        runtime = self.metadata.get("runtime")
        if not isinstance(runtime, Mapping):
            return None
        value = runtime.get("timings_output")
        return value if isinstance(value, str) and value else None

    @property
    def ref(self) -> str:
        return f"{self.system}:{self.id}"

    @property
    def selected_distribution(self) -> str | None:
        """The explicit distribution selector, if any."""
        value = self.metadata.get("distribution_id")
        return value if isinstance(value, str) else None

    @property
    def distribution_choices(self) -> tuple[str, ...]:
        """Available distribution ids for this catalog item."""
        value = self.metadata.get("distribution_choices")
        if not isinstance(value, (list, tuple)):
            return ()
        return tuple(str(x) for x in value)

    @property
    def selected_quality(self) -> str | None:
        """The explicit quality selector, if any."""
        value = self.metadata.get("selected_quality")
        return value if isinstance(value, str) else None

    def artifacts_for(
        self,
        role: str | None = None,
        *,
        component: str | None = None,
        quality: str | None = None,
    ) -> tuple[Artifact, ...]:
        """Return catalog artifacts matching the supplied metadata."""
        return tuple(
            artifact
            for artifact in self.artifacts
            if (role is None or artifact.role == role)
            and (component is None or artifact.component == component)
            and (quality is None or artifact.quality == quality)
        )

    def artifact(
        self,
        role: str,
        *,
        component: str | None = None,
        quality: str | None = None,
    ) -> Artifact:
        """Return the first artifact matching the supplied metadata."""
        candidates = self.artifacts_for(role, component=component, quality=quality)
        if not candidates:
            details = [f"role={role!r}"]
            if component is not None:
                details.append(f"component={component!r}")
            if quality is not None:
                details.append(f"quality={quality!r}")
            raise KeyError(f"No artifact {', '.join(details)} in {self.ref}")
        return candidates[0]

    def require_artifact(
        self,
        role: str,
        *,
        component: str | None = None,
        quality: str | None = None,
    ) -> Artifact:
        """Return exactly one artifact or raise on ambiguity."""
        candidates = self.artifacts_for(role, component=component, quality=quality)
        if len(candidates) == 0:
            details = [f"role={role!r}"]
            if component is not None:
                details.append(f"component={component!r}")
            if quality is not None:
                details.append(f"quality={quality!r}")
            raise KeyError(f"No artifact {', '.join(details)} in {self.ref}")
        if len(candidates) > 1:
            raise KeyError(
                f"Ambiguous artifact lookup for role={role!r} in {self.ref}: "
                f"{len(candidates)} matches. Specify component or quality."
            )
        return candidates[0]


@dataclass(frozen=True, slots=True)
class VoiceMetadata:
    """Authoritative descriptive language and gender metadata for a voice."""

    language: str
    locale: str
    language_label: str
    gender: str = "unknown"


@dataclass(frozen=True, slots=True)
class VoiceRecord:
    """A current voice exposed by a normalized catalog item."""

    catalog_item: CatalogItem
    voice_id: str
    metadata: VoiceMetadata
    languages: tuple[str, ...] = ()

    @property
    def system(self) -> str:
        return self.catalog_item.system

    @property
    def asset_id(self) -> str:
        return self.catalog_item.id

    @property
    def backing_ref(self) -> str:
        return self.catalog_item.ref

    @property
    def ref(self) -> str:
        if self.catalog_item.kind == "voice" and self.voice_id == self.catalog_item.id:
            return self.catalog_item.ref
        return f"{self.catalog_item.ref}/{self.voice_id}"

    @property
    def language(self) -> str:
        return self.metadata.language

    @property
    def locale(self) -> str:
        return self.metadata.locale

    @property
    def language_label(self) -> str:
        return self.metadata.language_label

    @property
    def gender(self) -> str:
        return self.metadata.gender


@dataclass(frozen=True, slots=True)
class InstalledArtifact:
    """Verified file installed under a managed installation.


    The record includes the role, path, size, digest, and optional component, quality,
    format, and metadata from the selected catalog artifact.
    """

    role: str
    filename: str
    path: Path
    sha256: str
    size: int
    quality: str | None = None
    component: str | None = None
    format: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class Installation:
    """Verified managed representation of a catalog item or imported model.


    ``id`` is the canonical item ID; ``storage_id`` can distinguish explicit
    quality/distribution selections. Use ``artifacts_for`` or ``require_artifact`` to
    query installed files by role, component, and quality.
    """

    system: str
    id: str
    kind: str
    path: Path
    artifacts: tuple[InstalledArtifact, ...]
    sample_rate: int | None = None
    voices: tuple[str, ...] = ()
    default_voice: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    storage_id: str | None = None

    @property
    def timing_output(self) -> str | None:
        """Return the declared auxiliary timing output, if valid."""
        runtime = self.metadata.get("runtime")
        if not isinstance(runtime, Mapping):
            return None
        value = runtime.get("timings_output")
        return value if isinstance(value, str) and value else None

    @property
    def ref(self) -> str:
        """Canonical catalog reference (system:id), never the physical storage key."""
        return f"{self.system}:{self.id}"

    @property
    def selected_distribution(self) -> str | None:
        """The explicit distribution selector, if any."""
        value = self.metadata.get("selected_distribution")
        return value if isinstance(value, str) else None

    @property
    def selected_quality(self) -> str | None:
        """The explicit quality selector, if any."""
        value = self.metadata.get("selected_quality")
        return value if isinstance(value, str) else None

    def artifacts_for(
        self,
        role: str | None = None,
        *,
        component: str | None = None,
        quality: str | None = None,
    ) -> tuple[InstalledArtifact, ...]:
        """Return installed artifacts matching the supplied catalog metadata."""
        return tuple(
            artifact
            for artifact in self.artifacts
            if (role is None or artifact.role == role)
            and (component is None or artifact.component == component)
            and (quality is None or artifact.quality == quality)
        )

    def artifact(
        self,
        role: str,
        *,
        component: str | None = None,
        quality: str | None = None,
    ) -> InstalledArtifact:
        candidates = self.artifacts_for(role, component=component, quality=quality)
        if not candidates:
            details = [f"role={role!r}"]
            if component is not None:
                details.append(f"component={component!r}")
            if quality is not None:
                details.append(f"quality={quality!r}")
            raise KeyError(f"No artifact {', '.join(details)} in {self.ref}")
        return candidates[0]

    def require_artifact(
        self,
        role: str,
        *,
        component: str | None = None,
        quality: str | None = None,
    ) -> InstalledArtifact:
        """Return exactly one artifact or raise on ambiguity."""
        candidates = self.artifacts_for(role, component=component, quality=quality)
        if len(candidates) == 0:
            details = [f"role={role!r}"]
            if component is not None:
                details.append(f"component={component!r}")
            if quality is not None:
                details.append(f"quality={quality!r}")
            raise KeyError(f"No artifact {', '.join(details)} in {self.ref}")
        if len(candidates) > 1:
            raise KeyError(
                f"Ambiguous artifact lookup for role={role!r} in {self.ref}: "
                f"{len(candidates)} matches. Specify component or quality."
            )
        return candidates[0]

    def artifact_path(
        self,
        role: str,
        *,
        component: str | None = None,
        quality: str | None = None,
    ) -> Path:
        """Convenience: return the filesystem path for a single artifact."""
        return self.artifact(role, component=component, quality=quality).path


@dataclass(frozen=True, slots=True)
class SessionDiagnostic:
    """Provider and graph metadata for one ONNX Runtime session."""

    component: str | None
    model_path: Path
    providers_requested: tuple[str, ...]
    providers_active: tuple[str, ...]
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]
    providers_resolved: tuple[str, ...] = ()
    provider_source: str = "unknown"


@dataclass(frozen=True, slots=True)
class RuntimeDistributionInfo:
    """Metadata for a Python distribution that may provide ``onnxruntime``."""

    name: str
    version: str | None
    location: Path | None
    owns_import: bool | None


@dataclass(frozen=True, slots=True)
class ProviderPreference:
    """Ordered provider policy and the configuration source that selected it."""

    requested: tuple[str, ...]
    source: Literal[
        "explicit",
        "ONNXVOICE_PROVIDERS",
        "ONNXVOICE_PROVIDER",
        "user_config",
        "default",
    ]


@dataclass(frozen=True, slots=True)
class OnnxRuntimeStatus:
    """Snapshot of the imported ONNX Runtime build and ONNXVoice provider policy."""

    importable: bool
    module_version: str | None
    module_path: Path | None
    build_device: str | None
    build_info: str | None
    distributions: tuple[RuntimeDistributionInfo, ...]
    imported_distribution: str | None
    distribution_state: Literal["resolved", "ambiguous", "vendor", "not_installed", "import_failed"]
    available_providers: tuple[str, ...]
    configured_providers: tuple[str, ...]
    provider_source: str
    resolved_providers: tuple[str, ...]
    import_error: str | None = None
    resolution_error: str | None = None


@dataclass(frozen=True, slots=True)
class RuntimeDiagnostic:
    """Runtime-level diagnostics containing one or more session records."""

    system: str
    ref: str
    layout: str
    sessions: tuple[SessionDiagnostic, ...]
    properties: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class SequenceLengthContract:
    """Authoritative graph evidence for one sequence axis."""

    kind: Literal["variable", "fixed", "unknown"]
    axis: int
    dimension: int | str | None
    source: str


@dataclass(frozen=True, slots=True)
class TensorSpec:
    """Name, ONNX Runtime type, and declared shape of one graph tensor."""

    name: str
    ort_type: str
    shape: tuple[Any, ...]


@dataclass(frozen=True, slots=True)
class InferenceResult:
    """Canonical inference result returned by a system adapter.


    ``audio`` is a one-dimensional float32 NumPy array and ``sample_rate`` is positive.
    Optional ``timings``, named outputs, and metadata preserve adapter-specific results.
    """

    audio: np.ndarray
    sample_rate: int
    timings: np.ndarray | None = None
    outputs: Mapping[str, np.ndarray] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        audio = np.asarray(self.audio, dtype=np.float32)
        if audio.ndim != 1:
            raise ValueError("audio must be a one-dimensional array")
        if self.sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        object.__setattr__(self, "audio", audio)
        if self.timings is not None:
            object.__setattr__(self, "timings", np.asarray(self.timings))


@dataclass(frozen=True, slots=True)
class CacheFilesystemUsage:
    """Filesystem capacity for the cache's containing volume."""

    total_bytes: int
    used_bytes: int
    free_bytes: int


@dataclass(frozen=True, slots=True)
class CacheFileUsage:
    """One regular file observed by a cache-root scan."""

    path: Path
    category: str
    apparent_bytes: int
    allocated_bytes: int
    device: int
    inode: int
    link_count: int = 1


@dataclass(frozen=True, slots=True)
class CachePathUsage:
    """Aggregated storage usage for one top-level cache path."""

    path: Path
    category: str
    file_count: int
    apparent_bytes: int
    unique_apparent_bytes: int
    allocated_bytes: int


@dataclass(frozen=True, slots=True)
class CacheScan:
    """Consistent, inode-aware inventory of regular files below the cache root."""

    root: Path
    filesystem: CacheFilesystemUsage
    files: tuple[CacheFileUsage, ...]
    top_level_usage: tuple[CachePathUsage, ...]
    apparent_bytes: int
    unique_apparent_bytes: int
    allocated_bytes: int
    unmanaged_paths: tuple[Path, ...] = ()
    legacy_paths: tuple[Path, ...] = ()
    errors: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class InstallationStorageUsage:
    """Logical and reclaimable storage attributed to one installation."""

    ref: str
    storage_id: str
    logical_bytes: int
    immediate_reclaimable_bytes: int
    gc_reclaimable_bytes: int
    immediate_reclaimable_allocated_bytes: int = 0
    gc_reclaimable_allocated_bytes: int = 0


@dataclass(frozen=True, slots=True)
class GcCandidate:
    """One path eligible for a conservative, explicitly categorized GC plan."""

    category: str
    path: Path
    apparent_bytes: int
    allocated_bytes: int
    reason: str


@dataclass(frozen=True, slots=True)
class GcPlan:
    """Immutable preview of cache files eligible for garbage collection."""

    candidates: tuple[GcCandidate, ...]
    apparent_bytes: int
    allocated_bytes: int

    include_legacy: bool = False


@dataclass(frozen=True, slots=True)
class RemovalPlan:
    """Non-mutating estimate of logical and physical impact from removals."""

    installations: tuple[InstallationStorageUsage, ...]
    logical_bytes: int
    immediate_reclaimable_bytes: int
    gc_reclaimable_bytes: int
    total_reclaimable_bytes: int
    immediate_reclaimable_allocated_bytes: int = 0
    gc_reclaimable_allocated_bytes: int = 0
    total_reclaimable_allocated_bytes: int = 0
    newly_orphaned_blob_count: int = 0
    newly_orphaned_blob_bytes: int = 0
    newly_orphaned_auxiliary_count: int = 0
    newly_orphaned_auxiliary_bytes: int = 0


@dataclass(frozen=True, slots=True)
class CacheUsage:
    """Logical, physical, and auxiliary byte/count totals for the cache."""

    root: Path
    installation_count: int

    install_logical_bytes: int
    blob_apparent_bytes: int
    catalog_bytes: int

    unique_file_bytes: int
    orphan_blob_count: int
    orphan_blob_bytes: int

    auxiliary_bytes: int = 0
    pocket_voice_state_count: int = 0
    pocket_voice_state_bytes: int = 0
    orphan_auxiliary_count: int = 0
    orphan_auxiliary_bytes: int = 0

    filesystem_total_bytes: int = 0
    filesystem_used_bytes: int = 0
    filesystem_free_bytes: int = 0
    cache_apparent_bytes: int = 0
    cache_unique_apparent_bytes: int = 0
    cache_allocated_bytes: int = 0
    install_apparent_bytes: int = 0
    metadata_bytes: int = 0
    lock_bytes: int = 0
    other_cache_bytes: int = 0
    legacy_cache_bytes: int = 0
    hardlinked_install_artifact_count: int = 0
    hardlinked_install_bytes: int = 0
    copied_install_artifact_count: int = 0
    copied_install_bytes: int = 0
    missing_blob_artifact_count: int = 0
    referenced_blob_count: int = 0
    referenced_blob_bytes: int = 0
    referenced_blob_allocated_bytes: int = 0
    orphan_blob_allocated_bytes: int = 0
    reclaimable_now_bytes: int = 0
    reclaimable_now_allocated_bytes: int = 0
    unmanaged_paths: tuple[Path, ...] = ()
    legacy_paths: tuple[Path, ...] = ()
    top_level_usage: tuple[CachePathUsage, ...] = ()
    scan_errors: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class GcReport:
    """Garbage-collection counts and byte totals.


    ``removed_bytes`` counts content blobs; auxiliary Pocket state reclamation is
    reported separately.
    """

    removed_blobs: int
    removed_bytes: int
    removed_auxiliary_count: int = 0
    removed_auxiliary_bytes: int = 0
    removed_candidates: int = 0
    removed_apparent_bytes: int = 0
    removed_allocated_bytes: int = 0
    removed_legacy_count: int = 0
    removed_legacy_bytes: int = 0


@dataclass(frozen=True, slots=True)
class AssetProgress:
    """Progress event emitted during asset and installation operations."""

    phase: str
    ref: str | None = None
    artifact: str | None = None
    completed: int | None = None
    total: int | None = None
    message: str | None = None
    role: str | None = None
    target: str | None = None
