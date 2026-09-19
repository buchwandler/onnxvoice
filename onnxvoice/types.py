from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

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
class InstalledArtifact:
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
    component: str | None
    model_path: Path
    providers_requested: tuple[str, ...]
    providers_active: tuple[str, ...]
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class RuntimeDiagnostic:
    system: str
    ref: str
    layout: str
    sessions: tuple[SessionDiagnostic, ...]


@dataclass(frozen=True, slots=True)
class TensorSpec:
    name: str
    ort_type: str
    shape: tuple[Any, ...]


@dataclass(frozen=True, slots=True)
class InferenceResult:
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
class CacheUsage:
    """Report of cache storage usage."""

    root: Path
    installation_count: int

    install_logical_bytes: int
    blob_apparent_bytes: int
    catalog_bytes: int

    unique_file_bytes: int
    orphan_blob_count: int
    orphan_blob_bytes: int


@dataclass(frozen=True, slots=True)
class GcReport:
    """Result of garbage collection."""

    removed_blobs: int
    removed_bytes: int


@dataclass(frozen=True, slots=True)
class AssetProgress:
    phase: str
    ref: str | None = None
    artifact: str | None = None
    completed: int | None = None
    total: int | None = None
    message: str | None = None
    role: str | None = None
    target: str | None = None
