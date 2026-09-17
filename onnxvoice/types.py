from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True, slots=True)
class Artifact:
    role: str
    filename: str
    url: str | None = None
    size: int | None = None
    sha256: str | None = None
    md5: str | None = None
    quality: str | None = None
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


@dataclass(frozen=True, slots=True)
class InstalledArtifact:
    role: str
    filename: str
    path: Path
    sha256: str
    size: int
    quality: str | None = None


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

    @property
    def ref(self) -> str:
        return f"{self.system}:{self.id}"

    def artifact(self, role: str, *, quality: str | None = None) -> InstalledArtifact:
        candidates = [artifact for artifact in self.artifacts if artifact.role == role]
        if quality is not None:
            candidates = [artifact for artifact in candidates if artifact.quality == quality]
        if not candidates:
            detail = f" with quality={quality!r}" if quality is not None else ""
            raise KeyError(f"No artifact role={role!r}{detail} in {self.ref}")
        return candidates[0]


@dataclass(frozen=True, slots=True)
class AudioResult:
    audio: np.ndarray
    sample_rate: int
    metadata: dict[str, Any] = field(default_factory=dict)
