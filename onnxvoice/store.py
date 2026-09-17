from __future__ import annotations

import json
import os
import shutil
import tempfile
import urllib.request
from pathlib import Path
from typing import Any

from platformdirs import user_cache_path

from .checksums import digest_file, verify_file
from .errors import AssetNotFoundError, IntegrityError, OfflineError
from .types import Artifact, CatalogItem, Installation, InstalledArtifact


class AssetStore:
    """Content-addressed local store shared by all onnxvoice consumers."""

    def __init__(self, root: str | Path | None = None, *, offline: bool = False) -> None:
        self.root = Path(
            root or os.environ.get("ONNXVOICE_CACHE_DIR") or user_cache_path("onnxvoice")
        )
        self.offline = offline
        self.blobs = self.root / "blobs" / "sha256"
        self.installs = self.root / "installs"
        self.blobs.mkdir(parents=True, exist_ok=True)
        self.installs.mkdir(parents=True, exist_ok=True)

    def install_path(self, system: str, item_id: str) -> Path:
        return self.installs / system / item_id

    def manifest_path(self, system: str, item_id: str) -> Path:
        return self.install_path(system, item_id) / "manifest.json"

    def is_installed(self, system: str, item_id: str) -> bool:
        return self.manifest_path(system, item_id).is_file()

    def installed(self, system: str | None = None) -> list[Installation]:
        roots = (
            [self.installs / system]
            if system
            else [p for p in self.installs.iterdir() if p.is_dir()]
        )
        result: list[Installation] = []
        for root in roots:
            if not root.exists():
                continue
            for item_dir in sorted(p for p in root.iterdir() if p.is_dir()):
                manifest = item_dir / "manifest.json"
                if manifest.is_file():
                    result.append(self._load_manifest(manifest))
        return result

    def get(self, system: str, item_id: str) -> Installation:
        manifest = self.manifest_path(system, item_id)
        if not manifest.is_file():
            raise AssetNotFoundError(f"Not installed: {system}:{item_id}")
        return self._load_manifest(manifest)

    def where(self, system: str, item_id: str) -> Path:
        return self.get(system, item_id).path

    def remove(self, system: str, item_id: str) -> None:
        path = self.install_path(system, item_id)
        if not path.exists():
            raise AssetNotFoundError(f"Not installed: {system}:{item_id}")
        shutil.rmtree(path)

    def install(
        self,
        item: CatalogItem,
        *,
        force: bool = False,
    ) -> Installation:
        if self.is_installed(item.system, item.id) and not force:
            installation = self.get(item.system, item.id)
            self.verify(installation)
            return installation

        target = self.install_path(item.system, item.id)
        target.parent.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix=f".{item.id}-", dir=target.parent))
        try:
            installed: list[InstalledArtifact] = []
            for artifact in item.artifacts:
                blob, sha256 = self._materialize_artifact(artifact)
                destination = staging / artifact.filename
                destination.parent.mkdir(parents=True, exist_ok=True)
                self._link_or_copy(blob, destination)
                installed.append(
                    InstalledArtifact(
                        role=artifact.role,
                        filename=artifact.filename,
                        path=destination,
                        sha256=sha256,
                        size=destination.stat().st_size,
                        quality=artifact.quality,
                    )
                )
            manifest_data = {
                "schema": 1,
                "system": item.system,
                "id": item.id,
                "kind": item.kind,
                "sample_rate": item.sample_rate,
                "voices": list(item.voices),
                "default_voice": item.default_voice,
                "metadata": item.metadata,
                "artifacts": [
                    {
                        "role": a.role,
                        "filename": a.filename,
                        "sha256": a.sha256,
                        "size": a.size,
                        "quality": a.quality,
                    }
                    for a in installed
                ],
            }
            (staging / "manifest.json").write_text(
                json.dumps(manifest_data, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
            if target.exists():
                shutil.rmtree(target)
            os.replace(staging, target)
            return self.get(item.system, item.id)
        except Exception:
            shutil.rmtree(staging, ignore_errors=True)
            raise

    def import_files(
        self,
        *,
        system: str,
        item_id: str,
        files: dict[str, str | Path],
        kind: str = "external",
        sample_rate: int | None = None,
        metadata: dict[str, Any] | None = None,
        force: bool = False,
    ) -> Installation:
        artifacts = []
        for role, raw_path in files.items():
            path = Path(raw_path).expanduser().resolve()
            if not path.is_file():
                raise AssetNotFoundError(str(path))
            artifacts.append(
                Artifact(
                    role=role,
                    filename=path.name,
                    url=path.as_uri(),
                    size=path.stat().st_size,
                    sha256=digest_file(path),
                )
            )
        item = CatalogItem(
            system=system,
            id=item_id,
            kind=kind,
            artifacts=tuple(artifacts),
            sample_rate=sample_rate,
            metadata=metadata or {"external": True},
        )
        return self.install(item, force=force)

    def verify(self, installation: Installation) -> None:
        for artifact in installation.artifacts:
            if not artifact.path.is_file():
                raise IntegrityError(f"Missing installed artifact: {artifact.path}")
            verify_file(artifact.path, expected_size=artifact.size, sha256=artifact.sha256)

    def gc(self) -> int:
        """Remove blobs no longer referenced by any install manifest."""
        referenced = {
            artifact.sha256
            for installation in self.installed()
            for artifact in installation.artifacts
        }
        removed = 0
        for blob in self.blobs.glob("*/*"):
            if blob.is_file() and blob.name not in referenced:
                blob.unlink()
                removed += 1
        return removed

    def _materialize_artifact(self, artifact: Artifact) -> tuple[Path, str]:
        if not artifact.url:
            raise AssetNotFoundError(f"Artifact {artifact.filename!r} has no URL")
        if artifact.sha256:
            known_blob = self._blob_path(artifact.sha256)
            if known_blob.is_file():
                verify_file(
                    known_blob,
                    expected_size=artifact.size,
                    sha256=artifact.sha256,
                    md5=artifact.md5,
                )
                return known_blob, artifact.sha256

        with tempfile.TemporaryDirectory(prefix="onnxvoice-download-") as temp_dir:
            temp = Path(temp_dir) / artifact.filename
            self._download(artifact.url, temp)
            verify_file(
                temp,
                expected_size=artifact.size,
                sha256=artifact.sha256,
                md5=artifact.md5,
            )
            sha256 = digest_file(temp)
            blob = self._blob_path(sha256)
            blob.parent.mkdir(parents=True, exist_ok=True)
            if not blob.exists():
                try:
                    os.replace(temp, blob)
                except OSError:
                    shutil.copy2(temp, blob)
            return blob, sha256

    def _download(self, source: str, destination: Path) -> None:
        if source.startswith("file://"):
            from urllib.parse import unquote, urlparse

            parsed = urlparse(source)
            shutil.copy2(Path(unquote(parsed.path)), destination)
            return
        if self.offline:
            raise OfflineError(f"Network access disabled while fetching {source}")
        request = urllib.request.Request(source, headers={"User-Agent": "onnxvoice/0"})
        try:
            with (
                urllib.request.urlopen(request, timeout=60) as response,
                destination.open("wb") as out,
            ):
                shutil.copyfileobj(response, out, length=1024 * 1024)
        except Exception as exc:
            raise AssetNotFoundError(f"Could not download {source}: {exc}") from exc

    def _blob_path(self, sha256: str) -> Path:
        return self.blobs / sha256[:2] / sha256

    @staticmethod
    def _link_or_copy(source: Path, destination: Path) -> None:
        try:
            os.link(source, destination)
        except OSError:
            shutil.copy2(source, destination)

    @staticmethod
    def _load_manifest(path: Path) -> Installation:
        data = json.loads(path.read_text(encoding="utf-8"))
        root = path.parent
        artifacts = tuple(
            InstalledArtifact(
                role=raw["role"],
                filename=raw["filename"],
                path=root / raw["filename"],
                sha256=raw["sha256"],
                size=raw["size"],
                quality=raw.get("quality"),
            )
            for raw in data.get("artifacts", ())
        )
        return Installation(
            system=data["system"],
            id=data["id"],
            kind=data.get("kind", "unknown"),
            path=root,
            artifacts=artifacts,
            sample_rate=data.get("sample_rate"),
            voices=tuple(data.get("voices") or ()),
            default_voice=data.get("default_voice"),
            metadata=data.get("metadata") or {},
        )
