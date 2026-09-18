from __future__ import annotations

import json
import os
import shutil
import tempfile
import time
import urllib.request
from collections.abc import Callable
from contextlib import suppress
from pathlib import Path
from typing import Any

from platformdirs import user_cache_path

from .checksums import digest_file, verify_file
from .errors import (
    AssetNotFoundError,
    IntegrityError,
    LockError,
    ManifestError,
    OfflineError,
    UnsafePathError,
)
from .types import (
    MANIFEST_SCHEMA_VERSION,
    Artifact,
    AssetProgress,
    CatalogItem,
    Installation,
    InstalledArtifact,
    validate_relative_path,
    validate_safe_component,
)

ProgressCallback = Callable[[AssetProgress], None]


class FileLock:
    """Small cross-process lock based on an atomically-created lock file."""

    def __init__(
        self,
        path: str | Path,
        *,
        timeout: float = 30.0,
        poll_interval: float = 0.05,
        stale_after: float = 300.0,
    ) -> None:
        self.path = Path(path)
        self.timeout = timeout
        self.poll_interval = poll_interval
        self.stale_after = stale_after
        self._owned = False

    def acquire(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        deadline = time.monotonic() + self.timeout
        while True:
            try:
                fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
            except FileExistsError:
                self._remove_stale()
                if time.monotonic() >= deadline:
                    raise LockError(f"Timed out acquiring lock: {self.path}") from None
                time.sleep(self.poll_interval)
                continue
            except OSError as exc:
                raise LockError(f"Could not acquire lock {self.path}: {exc}") from exc
            with os.fdopen(fd, "w", encoding="utf-8") as lock_file:
                lock_file.write(f"{os.getpid()} {time.time()}\n")
            self._owned = True
            return

    def release(self) -> None:
        if not self._owned:
            return
        try:
            self.path.unlink()
        except FileNotFoundError:
            pass
        except OSError as exc:
            raise LockError(f"Could not release lock {self.path}: {exc}") from exc
        finally:
            self._owned = False

    def __enter__(self) -> FileLock:
        self.acquire()
        return self

    def __exit__(self, *_: object) -> None:
        self.release()

    def _remove_stale(self) -> None:
        try:
            raw = self.path.read_text(encoding="utf-8").split()
            pid = int(raw[0])
            timestamp = float(raw[1])
        except (OSError, ValueError, IndexError):
            return
        if time.time() - timestamp < self.stale_after or _process_exists(pid):
            return
        with suppress(FileNotFoundError):
            self.path.unlink()
            pass


def _process_exists(pid: int) -> bool:
    if pid == os.getpid():
        return True
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


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
        self._safe_component(system, "system")
        self._safe_component(item_id, "item id")
        return self.installs / system / item_id

    def manifest_path(self, system: str, item_id: str) -> Path:
        return self.install_path(system, item_id) / "manifest.json"

    def is_installed(self, system: str, item_id: str) -> bool:
        return self.manifest_path(system, item_id).is_file()

    def installed(self, system: str | None = None) -> list[Installation]:
        roots = (
            [self.installs / self._safe_component(system, "system")]
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
        with FileLock(self._install_lock_path(system, item_id)), FileLock(self._gc_lock_path()):
            path = self.install_path(system, item_id)
            if not path.exists():
                raise AssetNotFoundError(f"Not installed: {system}:{item_id}")
            shutil.rmtree(path)

    def install(
        self,
        item: CatalogItem,
        *,
        force: bool = False,
        progress: ProgressCallback | None = None,
    ) -> Installation:
        storage_id = self._storage_item_id(item)
        with FileLock(self._install_lock_path(item.system, storage_id)):
            return self._install_unlocked(item, force=force, progress=progress)

    def _install_unlocked(
        self,
        item: CatalogItem,
        *,
        force: bool,
        progress: ProgressCallback | None,
    ) -> Installation:
        storage_id = self._storage_item_id(item)
        self._safe_component(item.system, "system")
        self._safe_component(storage_id, "item id")
        for artifact in item.artifacts:
            try:
                validate_relative_path(artifact.filename, field_name="artifact filename")
            except ValueError as exc:
                raise UnsafePathError(str(exc)) from exc
        if self.is_installed(item.system, storage_id) and not force:
            installation = self.get(item.system, storage_id)
            self.verify(installation)
            self._emit(
                progress,
                AssetProgress("install_completed", item.ref, message="already installed"),
            )
            return installation
        target = self.install_path(item.system, storage_id)
        target.parent.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix=f".{storage_id}-", dir=target.parent))
        self._emit(progress, AssetProgress("install_started", item.ref))
        try:
            installed: list[InstalledArtifact] = []
            with FileLock(self._gc_lock_path()):
                for artifact in item.artifacts:
                    blob, sha256 = self._materialize_artifact(
                        artifact,
                        ref=item.ref,
                        progress=progress,
                    )
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
                            component=artifact.component,
                            format=artifact.format,
                            metadata=dict(artifact.metadata),
                        )
                    )
                    self._emit(
                        progress,
                        AssetProgress(
                            "artifact_installed",
                            item.ref,
                            artifact.filename,
                            completed=len(installed),
                            total=len(item.artifacts),
                        ),
                    )
                manifest_data = {
                    "schema": MANIFEST_SCHEMA_VERSION,
                    "system": item.system,
                    "id": storage_id,
                    "kind": item.kind,
                    "sample_rate": item.sample_rate,
                    "voices": list(item.voices),
                    "default_voice": item.default_voice,
                    "metadata": item.metadata,
                    "artifacts": [
                        {
                            "role": artifact.role,
                            "filename": artifact.filename,
                            "sha256": artifact.sha256,
                            "size": artifact.size,
                            "quality": artifact.quality,
                            "component": artifact.component,
                            "format": artifact.format,
                            "metadata": dict(artifact.metadata),
                        }
                        for artifact in installed
                    ],
                }
                (staging / "manifest.json").write_text(
                    json.dumps(manifest_data, indent=2, sort_keys=True) + "\n", encoding="utf-8"
                )
                if target.exists():
                    shutil.rmtree(target)
                os.replace(staging, target)
            installation = self.get(item.system, item.id)
            self._emit(progress, AssetProgress("install_completed", item.ref))
            return installation
        except Exception:
            shutil.rmtree(staging, ignore_errors=True)
            self._emit(progress, AssetProgress("install_failed", item.ref))
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
        progress: ProgressCallback | None = None,
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
        return self.install(item, force=force, progress=progress)

    def verify(self, installation: Installation) -> None:
        root = installation.path.resolve()
        for artifact in installation.artifacts:
            path = artifact.path
            try:
                validate_relative_path(artifact.filename, field_name="artifact filename")
            except ValueError as exc:
                raise UnsafePathError(str(exc)) from exc
            if not path.resolve().is_relative_to(root):
                raise UnsafePathError(f"Artifact escapes installation root: {artifact.filename!r}")
            if not path.is_file():
                raise IntegrityError(f"Missing installed artifact: {path}")
            verify_file(path, expected_size=artifact.size, sha256=artifact.sha256)

    def gc(self) -> int:
        """Remove blobs no longer referenced by any install manifest."""
        with FileLock(self._gc_lock_path()):
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

    def _materialize_artifact(
        self,
        artifact: Artifact,
        *,
        ref: str,
        progress: ProgressCallback | None,
    ) -> tuple[Path, str]:
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
                self._emit(
                    progress,
                    AssetProgress("artifact_cached", ref, artifact.filename, message="blob reused"),
                )
                return known_blob, artifact.sha256

        self._emit(
            progress, AssetProgress("download_started", ref, artifact.filename, total=artifact.size)
        )
        with tempfile.TemporaryDirectory(prefix="onnxvoice-download-") as temp_dir:
            temp = Path(temp_dir) / Path(artifact.filename).name
            self._download(
                artifact.url, temp, ref=ref, artifact=artifact.filename, progress=progress
            )
            verify_file(
                temp,
                expected_size=artifact.size,
                sha256=artifact.sha256,
                md5=artifact.md5,
            )
            sha256 = digest_file(temp)
            blob = self._blob_path(sha256)
            with FileLock(self._blob_lock_path(sha256)):
                blob.parent.mkdir(parents=True, exist_ok=True)
                if not blob.exists():
                    try:
                        os.replace(temp, blob)
                    except OSError:
                        shutil.copy2(temp, blob)
            self._emit(
                progress,
                AssetProgress(
                    "download_completed",
                    ref,
                    artifact.filename,
                    completed=artifact.size,
                    total=artifact.size,
                ),
            )
            return blob, sha256

    def _download(
        self,
        source: str,
        destination: Path,
        *,
        ref: str,
        artifact: str,
        progress: ProgressCallback | None,
    ) -> None:
        if source.startswith("file://"):
            from urllib.parse import unquote, urlparse

            parsed = urlparse(source)
            shutil.copy2(Path(unquote(parsed.path)), destination)
            self._emit(
                progress,
                AssetProgress(
                    "download_progress", ref, artifact, completed=destination.stat().st_size
                ),
            )
            return
        if self.offline:
            raise OfflineError(f"Network access disabled while fetching {source}")
        request = urllib.request.Request(source, headers={"User-Agent": "onnxvoice/0"})
        try:
            with (
                urllib.request.urlopen(request, timeout=60) as response,
                destination.open("wb") as out,
            ):
                total = response.headers.get("Content-Length")
                total_bytes = int(total) if total and total.isdigit() else None
                completed = 0
                while chunk := response.read(1024 * 1024):
                    out.write(chunk)
                    completed += len(chunk)
                    self._emit(
                        progress,
                        AssetProgress("download_progress", ref, artifact, completed, total_bytes),
                    )
        except Exception as exc:
            raise AssetNotFoundError(f"Could not download {source}: {exc}") from exc

    def _blob_path(self, sha256: str) -> Path:
        return self.blobs / sha256[:2] / sha256

    def _install_lock_path(self, system: str, item_id: str) -> Path:
        self._safe_component(system, "system")
        self._safe_component(item_id, "item id")
        return self.root / "locks" / "install" / system / f"{item_id}.lock"

    def _blob_lock_path(self, sha256: str) -> Path:
        return self.root / "locks" / "blob" / f"{sha256}.lock"

    def _gc_lock_path(self) -> Path:
        return self.root / "locks" / "gc.lock"

    @staticmethod
    def _link_or_copy(source: Path, destination: Path) -> None:
        try:
            os.link(source, destination)
        except OSError:
            shutil.copy2(source, destination)

    @staticmethod
    def _emit(progress: ProgressCallback | None, event: AssetProgress) -> None:
        if progress is not None:
            progress(event)

    @classmethod
    def _load_manifest(cls, path: Path) -> Installation:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ManifestError(f"Could not read manifest {path}: {exc}") from exc
        if not isinstance(data, dict):
            raise ManifestError(f"Manifest must contain an object: {path}")

        schema = data.get("schema", 1)
        if schema not in {1, MANIFEST_SCHEMA_VERSION}:
            raise ManifestError(f"Unsupported manifest schema {schema!r} in {path}")
        try:
            system = validate_safe_component(data["system"], field_name="system")
            item_id = validate_safe_component(data["id"], field_name="item id")
        except (KeyError, ValueError) as exc:
            raise UnsafePathError(f"Invalid manifest identity in {path}: {exc}") from exc

        root = path.parent.resolve()
        raw_artifacts = data.get("artifacts")
        if not isinstance(raw_artifacts, list):
            raise ManifestError(f"Manifest artifacts must be a list: {path}")
        artifacts: list[InstalledArtifact] = []
        for raw in raw_artifacts:
            if not isinstance(raw, dict):
                raise ManifestError(f"Manifest artifact must be an object: {path}")
            try:
                filename = validate_relative_path(raw["filename"], field_name="artifact filename")
                role = raw["role"]
                sha256 = raw["sha256"]
                size = raw["size"]
            except (KeyError, ValueError) as exc:
                raise UnsafePathError(f"Invalid artifact in {path}: {exc}") from exc
            if not isinstance(role, str) or not role or not isinstance(sha256, str):
                raise ManifestError(f"Invalid artifact identity in {path}")
            if not isinstance(size, int) or size < 0:
                raise ManifestError(f"Invalid artifact size in {path}")
            artifact_path = root / filename
            if not artifact_path.resolve().is_relative_to(root):
                raise UnsafePathError(f"Artifact escapes installation root: {filename!r}")
            metadata = raw.get("metadata") or {}
            if not isinstance(metadata, dict):
                raise ManifestError(f"Artifact metadata must be an object: {path}")
            artifacts.append(
                InstalledArtifact(
                    role=role,
                    filename=filename,
                    path=artifact_path,
                    sha256=sha256,
                    size=size,
                    quality=raw.get("quality"),
                    component=raw.get("component"),
                    format=raw.get("format"),
                    metadata=metadata,
                )
            )

        metadata = data.get("metadata") or {}
        if not isinstance(metadata, dict):
            raise ManifestError(f"Manifest metadata must be an object: {path}")
        return Installation(
            system=system,
            id=item_id,
            kind=data.get("kind", "unknown"),
            path=root,
            artifacts=tuple(artifacts),
            sample_rate=data.get("sample_rate"),
            voices=tuple(data.get("voices") or ()),
            default_voice=data.get("default_voice"),
            metadata=metadata,
        )

    @staticmethod
    def _storage_item_id(item: CatalogItem) -> str:
        value = item.metadata.get("cache_id")
        return value if isinstance(value, str) and value else item.id

    @staticmethod
    def _safe_component(value: str, field_name: str) -> str:
        try:
            return validate_safe_component(value, field_name=field_name)
        except ValueError as exc:
            raise UnsafePathError(str(exc)) from exc
