from __future__ import annotations

import errno
import json
import os
import re
import shutil
import stat as stat_module
import tempfile
import threading
import time
import urllib.request
import uuid
import weakref
from collections.abc import Callable, Mapping, Sequence
from contextlib import suppress
from dataclasses import replace as dataclass_replace
from pathlib import Path
from typing import Any

from platformdirs import user_cache_path

from .checksums import digest_file, verify_file
from .errors import (
    AssetDownloadError,
    AssetNotFoundError,
    IntegrityError,
    LockError,
    ManifestError,
    OfflineError,
    UnsafePathError,
)
from .huggingface import HuggingFaceSource, download_huggingface_file
from .types import (
    MANIFEST_SCHEMA_VERSION,
    Artifact,
    AssetProgress,
    CacheFilesystemUsage,
    CacheFileUsage,
    CachePathUsage,
    CacheScan,
    CacheUsage,
    CatalogItem,
    GcCandidate,
    GcPlan,
    GcReport,
    Installation,
    InstallationStorageUsage,
    InstalledArtifact,
    RemovalPlan,
    validate_relative_path,
    validate_safe_component,
)

ProgressCallback = Callable[[AssetProgress], None]

_LOCAL_LOCKS: dict[str, threading.Lock] = {}
_LOCAL_LOCKS_GUARD = threading.Lock()
_ACTIVE_FILE_LOCKS: weakref.WeakSet[Any] = weakref.WeakSet()


_INSTALL_MANIFEST_FILENAME = ".onnxvoice-install.json"


def _allocated_bytes(stat_result: os.stat_result) -> int:
    """Return allocated filesystem bytes, falling back to apparent size."""
    blocks = getattr(stat_result, "st_blocks", None)
    if isinstance(blocks, int) and blocks >= 0:
        return blocks * 512
    return stat_result.st_size


def _local_lock_for(path: Path) -> threading.Lock:
    key = os.path.normcase(os.path.abspath(os.fspath(path)))
    with _LOCAL_LOCKS_GUARD:
        return _LOCAL_LOCKS.setdefault(key, threading.Lock())


def _reset_local_locks_after_fork() -> None:
    """Discard inherited thread locks and close child copies of lock handles."""
    global _LOCAL_LOCKS, _LOCAL_LOCKS_GUARD, _ACTIVE_FILE_LOCKS

    for lock in tuple(_ACTIVE_FILE_LOCKS):
        fd = lock._fd
        lock._fd = None
        lock._owned = False
        lock._local_lock = None
        if fd is not None:
            with suppress(OSError):
                os.close(fd)
    _ACTIVE_FILE_LOCKS = weakref.WeakSet()
    _LOCAL_LOCKS = {}
    _LOCAL_LOCKS_GUARD = threading.Lock()


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_reset_local_locks_after_fork)


def _platform_attr(module: Any, name: str, default: Any = None) -> Any:
    """Read an attribute that exists only on one supported platform."""
    return getattr(module, name, default)


def _windows_lock_api() -> Any:
    """Return the Win32 kernel API used for byte-range lock operations."""
    import ctypes
    from ctypes import wintypes

    class Overlapped(ctypes.Structure):
        _fields_ = [
            ("internal", ctypes.c_void_p),
            ("internal_high", ctypes.c_void_p),
            ("offset", wintypes.DWORD),
            ("offset_high", wintypes.DWORD),
            ("event", wintypes.HANDLE),
        ]

    kernel32 = _platform_attr(ctypes, "WinDLL")("kernel32", use_last_error=True)
    lock_file_ex = kernel32.LockFileEx
    lock_file_ex.argtypes = [
        wintypes.HANDLE,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.DWORD,
        ctypes.POINTER(Overlapped),
    ]
    lock_file_ex.restype = wintypes.BOOL
    unlock_file_ex = kernel32.UnlockFileEx
    unlock_file_ex.argtypes = [
        wintypes.HANDLE,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.DWORD,
        ctypes.POINTER(Overlapped),
    ]
    unlock_file_ex.restype = wintypes.BOOL
    return lock_file_ex, unlock_file_ex, Overlapped


def _open_windows_lock_file(path: Path) -> int:
    """Open a lock file with sharing flags that permit contending processes."""
    import ctypes
    import msvcrt
    from ctypes import wintypes

    kernel32 = _platform_attr(ctypes, "WinDLL")("kernel32", use_last_error=True)
    create_file = kernel32.CreateFileW
    create_file.argtypes = [
        wintypes.LPCWSTR,
        wintypes.DWORD,
        wintypes.DWORD,
        ctypes.c_void_p,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.HANDLE,
    ]
    create_file.restype = wintypes.HANDLE
    handle = create_file(
        os.fspath(path),
        0xC0000000,  # GENERIC_READ | GENERIC_WRITE
        0x00000007,  # FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE
        None,
        4,  # OPEN_ALWAYS
        0x00000080,  # FILE_ATTRIBUTE_NORMAL
        None,
    )
    if handle == wintypes.HANDLE(-1).value:
        raise _platform_attr(ctypes, "WinError")(_platform_attr(ctypes, "get_last_error")())
    try:
        return _platform_attr(msvcrt, "open_osfhandle")(
            handle, os.O_RDWR | _platform_attr(os, "O_BINARY", 0)
        )
    except BaseException:
        kernel32.CloseHandle(handle)
        raise


def _open_lock_file(path: Path) -> int:
    """Open a lock file, working around Windows byte-lock sharing semantics."""
    flags = os.O_CREAT | os.O_RDWR
    if os.name == "nt":
        flags |= getattr(os, "O_BINARY", 0)
    try:
        return os.open(path, flags, 0o644)
    except OSError as exc:
        # The CRT may reject opening a file that another process has locked,
        # even though CreateFileW can open it with explicit sharing flags.
        # Preserve other open failures (including test-injected failures).
        if os.name == "nt" and (
            exc.errno == errno.EACCES or getattr(exc, "winerror", None) in (5, 32)
        ):
            return _open_windows_lock_file(path)
        raise


def _try_os_lock(fd: int) -> bool:
    """Try an exclusive non-blocking platform lock; return False if contended."""
    if os.name == "nt":
        import ctypes
        import msvcrt

        lock_file_ex, _, overlapped_type = _windows_lock_api()
        overlapped = overlapped_type()
        flags = 0x00000002 | 0x00000001  # LOCKFILE_EXCLUSIVE_LOCK | FAIL_IMMEDIATELY
        handle = _platform_attr(msvcrt, "get_osfhandle")(fd)
        if lock_file_ex(handle, flags, 0, 1, 0, ctypes.byref(overlapped)):
            return True
        error = _platform_attr(ctypes, "get_last_error")()
        if error in (33, 158, 212):  # lock violation / already locked
            return False
        raise _platform_attr(ctypes, "WinError")(error)

    if os.name == "posix":
        import fcntl

        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            if exc.errno in (errno.EACCES, errno.EAGAIN, errno.EWOULDBLOCK):
                return False
            raise
        return True

    raise OSError(f"No advisory file-lock implementation for os.name={os.name!r}")


def _unlock_os_lock(fd: int) -> None:
    if os.name == "nt":
        import ctypes
        import msvcrt

        _, unlock_file_ex, overlapped_type = _windows_lock_api()
        overlapped = overlapped_type()
        handle = _platform_attr(msvcrt, "get_osfhandle")(fd)
        if not unlock_file_ex(handle, 0, 1, 0, ctypes.byref(overlapped)):
            raise _platform_attr(ctypes, "WinError")(_platform_attr(ctypes, "get_last_error")())
        return
    if os.name == "posix":
        import fcntl

        fcntl.flock(fd, fcntl.LOCK_UN)
        return
    raise OSError(f"No advisory file-lock implementation for os.name={os.name!r}")


class FileLock:
    """Cross-process advisory lock held by an open OS-managed lock-file handle.

    ``stale_after`` is retained as a source-compatible constructor argument. OS
    locks are released by the operating system when their owning process exits,
    so PID/timestamp based stale-lock reclamation is not needed.
    """

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
        self._fd: int | None = None
        self._local_lock: threading.Lock | None = None
        with _LOCAL_LOCKS_GUARD:
            _ACTIVE_FILE_LOCKS.add(self)

    def acquire(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        deadline = time.monotonic() + self.timeout
        local_lock = _local_lock_for(self.path)
        if not local_lock.acquire(timeout=max(0.0, deadline - time.monotonic())):
            raise LockError(f"Timed out acquiring lock: {self.path}") from None
        self._local_lock = local_lock
        try:
            try:
                self._fd = _open_lock_file(self.path)
            except OSError as exc:
                raise LockError(f"Could not acquire lock {self.path}: {exc}") from exc

            while True:
                try:
                    if _try_os_lock(self._fd):
                        self._owned = True
                        return
                except OSError as exc:
                    raise LockError(f"Could not acquire lock {self.path}: {exc}") from exc
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise LockError(f"Timed out acquiring lock: {self.path}") from None
                time.sleep(min(max(0.0, self.poll_interval), remaining))
        except BaseException:
            fd = self._fd
            self._fd = None
            self._owned = False
            self._local_lock = None
            if fd is not None:
                with suppress(OSError):
                    os.close(fd)
            local_lock.release()
            raise

    def release(self) -> None:
        if not self._owned:
            return
        fd = self._fd
        local_lock = self._local_lock
        self._fd = None
        self._local_lock = None
        self._owned = False
        release_error: OSError | None = None
        if fd is not None:
            try:
                _unlock_os_lock(fd)
            except OSError as exc:
                release_error = exc
            try:
                os.close(fd)
            except OSError as exc:
                if release_error is None:
                    release_error = exc
        if local_lock is not None:
            local_lock.release()
        if release_error is not None:
            raise LockError(
                f"Could not release lock {self.path}: {release_error}"
            ) from release_error

    def __enter__(self) -> FileLock:
        self.acquire()
        return self

    def __exit__(self, *_: object) -> None:
        self.release()


class AssetStore:
    """Content-addressed local store shared by all onnxvoice consumers."""

    def __init__(self, root: str | Path | None = None, *, offline: bool = False) -> None:
        """Create the local store and its blob/install roots.


        Parameters
        ----------
        root : str or pathlib.Path, optional
            Cache directory. Defaults to the platform cache or
            ``ONNXVOICE_CACHE_DIR``.
        offline : bool, default=False
            Disallow network access for store operations that can download files.
        """
        self.root = Path(
            root or os.environ.get("ONNXVOICE_CACHE_DIR") or user_cache_path("onnxvoice")
        )
        self.offline = offline
        self.blobs = self.root / "blobs" / "sha256"
        self.installs = self.root / "installs"
        self.blobs.mkdir(parents=True, exist_ok=True)
        self.installs.mkdir(parents=True, exist_ok=True)

    def install_path(self, system: str, item_id: str) -> Path:
        """Return the managed installation directory for a system and item ID."""
        self._safe_component(system, "system")
        self._safe_component(item_id, "item id")
        return self.installs / system / item_id

    def manifest_path(self, system: str, item_id: str) -> Path:
        """Return the internal installation-record path for a managed installation."""
        return self.install_path(system, item_id) / _INSTALL_MANIFEST_FILENAME

    def _find_manifest_path(self, system: str, item_id: str) -> Path:
        """Find the current record or a legacy in-tree manifest."""
        current = self.manifest_path(system, item_id)
        if current.is_file():
            return current
        return self.install_path(system, item_id) / "manifest.json"

    def is_installed(self, system: str, item_id: str) -> bool:
        """Return whether the requested installation has a manifest."""
        return self._find_manifest_path(system, item_id).is_file()

    def installed(self, system: str | None = None) -> list[Installation]:
        """List managed installations, optionally filtered by system."""
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
                manifest = self._find_manifest_path(root.name, item_dir.name)
                if manifest.is_file():
                    result.append(self._load_manifest(manifest))
        return result

    def get(self, system: str, item_id: str) -> Installation:
        """Load one managed installation from its manifest."""
        manifest = self._find_manifest_path(system, item_id)
        if not manifest.is_file():
            raise AssetNotFoundError(f"Not installed: {system}:{item_id}")
        return self._load_manifest(manifest)

    def where(self, system: str, item_id: str) -> Path:
        """Return the path recorded for one managed installation."""
        return self.get(system, item_id).path

    @staticmethod
    def _is_within(path: Path, directory: Path) -> bool:
        return path == directory or directory in path.parents

    @staticmethod
    def _installation_key(installation: Installation) -> tuple[str, str]:
        return installation.system, installation.storage_id or installation.id

    def _plan_gc_after_removal(
        self,
        selected: Sequence[Installation],
        installations: Sequence[Installation],
        scan: CacheScan,
    ) -> GcPlan:
        """Simulate safe GC after removing selected installation directories."""
        selected_keys = {self._installation_key(installation) for installation in selected}
        selected_roots = tuple(installation.path for installation in selected)

        def inode_key(file: CacheFileUsage) -> tuple[Any, ...]:
            if file.inode:
                return ("inode", file.device, file.inode)
            return ("path", os.fspath(file.path))

        removed_link_counts: dict[tuple[Any, ...], int] = {}
        for file in scan.files:
            if any(self._is_within(file.path, root) for root in selected_roots):
                key = inode_key(file)
                removed_link_counts[key] = removed_link_counts.get(key, 0) + 1

        remaining_files = []
        for file in scan.files:
            if any(self._is_within(file.path, root) for root in selected_roots):
                continue
            removed_links = removed_link_counts.get(inode_key(file), 0)
            if removed_links:
                file = dataclass_replace(file, link_count=max(1, file.link_count - removed_links))
            remaining_files.append(file)
        remaining_installations = [
            installation
            for installation in installations
            if self._installation_key(installation) not in selected_keys
        ]
        simulated_scan = dataclass_replace(scan, files=tuple(remaining_files))
        return self._build_gc_plan_unlocked(remaining_installations, simulated_scan)

    def _calculate_removal_reclaimability(
        self,
        selected: Sequence[Installation],
        installations: Sequence[Installation],
        scan: CacheScan,
        *,
        gc: bool,
    ) -> tuple[int, int, int, int]:
        """Estimate bytes freed immediately and after safe GC for selected views."""
        selected_roots = tuple(installation.path for installation in selected)

        def inode_key(file: CacheFileUsage) -> tuple[Any, ...]:
            if file.inode:
                return ("inode", file.device, file.inode)
            return ("path", os.fspath(file.path))

        files_by_inode: dict[tuple[Any, ...], list[CacheFileUsage]] = {}
        removed_by_inode: dict[tuple[Any, ...], list[CacheFileUsage]] = {}
        for file in scan.files:
            key = inode_key(file)
            files_by_inode.setdefault(key, []).append(file)
            if any(self._is_within(file.path, root) for root in selected_roots):
                removed_by_inode.setdefault(key, []).append(file)

        immediate_bytes = 0
        immediate_allocated_bytes = 0
        for key, removed_files in removed_by_inode.items():
            link_count = max(file.link_count for file in removed_files)
            if len(removed_files) == len(files_by_inode[key]) and link_count <= len(removed_files):
                immediate_bytes += removed_files[0].apparent_bytes
                immediate_allocated_bytes += removed_files[0].allocated_bytes

        gc_bytes = 0
        gc_allocated_bytes = 0
        if gc:
            gc_plan = self._plan_gc_after_removal(selected, installations, scan)
            gc_bytes = gc_plan.apparent_bytes
            gc_allocated_bytes = gc_plan.allocated_bytes
        return immediate_bytes, immediate_allocated_bytes, gc_bytes, gc_allocated_bytes

    def plan_remove(
        self, installations: Sequence[Installation], *, gc: bool = False
    ) -> RemovalPlan:
        """Plan multi-installation removal without changing cache contents."""
        if not installations:
            raise ValueError("At least one installation must be selected for removal.")
        with FileLock(self._gc_lock_path()):
            current_installations = self.installed()
            current_by_key = {
                self._installation_key(installation): installation
                for installation in current_installations
            }
            requested_keys = dict.fromkeys(
                self._installation_key(installation) for installation in installations
            )
            missing = [key for key in requested_keys if key not in current_by_key]
            if missing:
                refs = ", ".join(f"{system}:{storage_id}" for system, storage_id in missing)
                raise AssetNotFoundError(f"Not installed: {refs}")
            selected = tuple(current_by_key[key] for key in requested_keys)
            scan = self._scan_cache()
            immediate, immediate_allocated, gc_bytes, gc_allocated = (
                self._calculate_removal_reclaimability(selected, current_installations, scan, gc=gc)
            )
            newly_orphaned_blob_count = 0
            newly_orphaned_blob_bytes = 0
            newly_orphaned_auxiliary_count = 0
            newly_orphaned_auxiliary_bytes = 0
            if gc:
                baseline_gc = self._build_gc_plan_unlocked(current_installations, scan)
                post_removal_gc = self._plan_gc_after_removal(selected, current_installations, scan)
                baseline_blobs = {
                    candidate.path
                    for candidate in baseline_gc.candidates
                    if candidate.category == "orphan_blob"
                }
                new_blobs = [
                    candidate
                    for candidate in post_removal_gc.candidates
                    if candidate.category == "orphan_blob" and candidate.path not in baseline_blobs
                ]
                newly_orphaned_blob_count = len(new_blobs)
                newly_orphaned_blob_bytes = sum(candidate.apparent_bytes for candidate in new_blobs)
                baseline_state_dirs = {
                    candidate.path.parent
                    for candidate in baseline_gc.candidates
                    if candidate.category == "orphan_pocket_voice_state"
                    and candidate.path.suffix == ".json"
                }
                post_state_dirs = {
                    candidate.path.parent
                    for candidate in post_removal_gc.candidates
                    if candidate.category == "orphan_pocket_voice_state"
                    and candidate.path.suffix == ".json"
                }
                new_state_dirs = post_state_dirs - baseline_state_dirs
                newly_orphaned_auxiliary_count = len(new_state_dirs)
                newly_orphaned_auxiliary_bytes = sum(
                    candidate.apparent_bytes
                    for candidate in post_removal_gc.candidates
                    if candidate.category == "orphan_pocket_voice_state"
                    and candidate.path.parent in new_state_dirs
                )
            installation_usage = []
            for installation in selected:
                one_immediate, one_allocated, one_gc, one_gc_allocated = (
                    self._calculate_removal_reclaimability(
                        (installation,), current_installations, scan, gc=gc
                    )
                )
                installation_usage.append(
                    InstallationStorageUsage(
                        ref=installation.ref,
                        storage_id=installation.storage_id or installation.id,
                        logical_bytes=sum(artifact.size for artifact in installation.artifacts),
                        immediate_reclaimable_bytes=one_immediate,
                        gc_reclaimable_bytes=one_gc,
                        immediate_reclaimable_allocated_bytes=one_allocated,
                        gc_reclaimable_allocated_bytes=one_gc_allocated,
                    )
                )
            logical_bytes = sum(
                artifact.size for installation in selected for artifact in installation.artifacts
            )
            return RemovalPlan(
                installations=tuple(installation_usage),
                logical_bytes=logical_bytes,
                immediate_reclaimable_bytes=immediate,
                gc_reclaimable_bytes=gc_bytes,
                total_reclaimable_bytes=immediate + gc_bytes,
                immediate_reclaimable_allocated_bytes=immediate_allocated,
                gc_reclaimable_allocated_bytes=gc_allocated,
                total_reclaimable_allocated_bytes=immediate_allocated + gc_allocated,
                newly_orphaned_blob_count=newly_orphaned_blob_count,
                newly_orphaned_blob_bytes=newly_orphaned_blob_bytes,
                newly_orphaned_auxiliary_count=newly_orphaned_auxiliary_count,
                newly_orphaned_auxiliary_bytes=newly_orphaned_auxiliary_bytes,
            )

    def remove(self, system: str, item_id: str) -> InstallationStorageUsage:
        """Remove one installation view and report bytes immediately reclaimed."""
        with FileLock(self._install_lock_path(system, item_id)), FileLock(self._gc_lock_path()):
            path = self.install_path(system, item_id)
            if not path.exists():
                raise AssetNotFoundError(f"Not installed: {system}:{item_id}")
            installations = self.installed()
            installation = next(
                (
                    item
                    for item in installations
                    if self._installation_key(item) == (system, item_id)
                ),
                Installation(
                    system=system,
                    id=item_id,
                    kind="unknown",
                    path=path,
                    artifacts=(),
                    storage_id=item_id,
                ),
            )
            scan = self._scan_cache()
            immediate, immediate_allocated, _, _ = self._calculate_removal_reclaimability(
                (installation,), installations, scan, gc=False
            )
            shutil.rmtree(path)
            return InstallationStorageUsage(
                ref=installation.ref,
                storage_id=installation.storage_id or installation.id,
                logical_bytes=sum(artifact.size for artifact in installation.artifacts),
                immediate_reclaimable_bytes=immediate,
                gc_reclaimable_bytes=0,
                immediate_reclaimable_allocated_bytes=immediate_allocated,
            )

    def install(
        self,
        item: CatalogItem,
        *,
        force: bool = False,
        progress: ProgressCallback | None = None,
    ) -> Installation:
        """Install and verify catalog artifacts in the shared content-addressed store.


        Existing installations are verified and reused unless ``force=True``.
        """
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
        if item.system == "pocket" and item.metadata.get("canonical_catalog") is True:
            for artifact in item.artifacts:
                if (
                    not isinstance(artifact.size, int)
                    or isinstance(artifact.size, bool)
                    or artifact.size <= 0
                ):
                    raise IntegrityError(
                        f"Canonical Pocket artifact {artifact.filename!r} has no positive size"
                    )
                if (
                    not isinstance(artifact.sha256, str)
                    or re.fullmatch(r"[0-9a-f]{64}", artifact.sha256) is None
                ):
                    raise IntegrityError(
                        f"Canonical Pocket artifact {artifact.filename!r} has no valid SHA-256"
                    )
        for artifact in item.artifacts:
            try:
                validate_relative_path(artifact.filename, field_name="artifact filename")
            except ValueError as exc:
                raise UnsafePathError(str(exc)) from exc
        if any(artifact.filename == _INSTALL_MANIFEST_FILENAME for artifact in item.artifacts):
            raise UnsafePathError(
                f"Artifact path {_INSTALL_MANIFEST_FILENAME!r} is reserved for installation metadata"
            )
        expected_bytes = self._expected_install_bytes(item.artifacts)
        if self.is_installed(item.system, storage_id) and not force:
            installation = self.get(item.system, storage_id)
            target = self.install_path(item.system, storage_id)
            self._emit(
                progress,
                AssetProgress(
                    "install_started", item.ref, total=expected_bytes, target=str(target)
                ),
            )
            self.verify(installation)
            self._emit(
                progress,
                AssetProgress(
                    "install_completed", item.ref, message="already installed", target=str(target)
                ),
            )
            return installation
        target = self.install_path(item.system, storage_id)
        target.parent.mkdir(parents=True, exist_ok=True)
        staging = target.parent / f".{storage_id}-{uuid.uuid4().hex}"
        backup: Path | None = None
        committed = False
        try:
            staging.mkdir()
            self._emit(
                progress,
                AssetProgress(
                    "install_started", item.ref, total=expected_bytes, target=str(target)
                ),
            )
            installed: list[InstalledArtifact] = []
            with FileLock(self._gc_lock_path()):
                for artifact in item.artifacts:
                    artifact_target = str(target / artifact.filename)
                    blob, sha256 = self._materialize_artifact(
                        artifact,
                        ref=item.ref,
                        progress=progress,
                        target=artifact_target,
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
                            role=artifact.role,
                            target=str(target / artifact.filename),
                        ),
                    )
                manifest_data = {
                    "schema": MANIFEST_SCHEMA_VERSION,
                    "system": item.system,
                    "id": item.id,
                    "storage_id": storage_id,
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
                (staging / _INSTALL_MANIFEST_FILENAME).write_text(
                    json.dumps(manifest_data, indent=2, sort_keys=True) + "\n", encoding="utf-8"
                )
                backup = self._commit_staging(staging, target)
                committed = True
            installation = self.get(item.system, storage_id)
            self._emit(progress, AssetProgress("install_completed", item.ref, target=str(target)))
            if backup is not None:
                shutil.rmtree(backup, ignore_errors=True)
            return installation
        except BaseException:
            with suppress(BaseException):
                shutil.rmtree(staging, ignore_errors=True)
            with suppress(BaseException):
                if backup is not None:
                    self._restore_backup(backup, target)
                elif committed and target.exists():
                    shutil.rmtree(target, ignore_errors=True)
            with suppress(BaseException):
                self._emit(
                    progress,
                    AssetProgress("install_failed", item.ref, target=str(target)),
                )
            raise

    @staticmethod
    def _expected_install_bytes(artifacts: Sequence[Artifact]) -> int | None:
        total = 0
        for artifact in artifacts:
            size = artifact.size
            if not isinstance(size, int) or isinstance(size, bool) or size < 0:
                return None
            total += size
        return total

    @staticmethod
    def _commit_staging(staging: Path, target: Path) -> Path | None:
        backup: Path | None = (
            target.parent / f".onnxvoice-backup-{uuid.uuid4().hex}" if target.exists() else None
        )
        try:
            if backup is not None:
                os.replace(target, backup)
            os.replace(staging, target)
        except BaseException:
            if backup is not None and backup.exists():
                with suppress(BaseException):
                    AssetStore._restore_backup(backup, target)
            elif not staging.exists() and target.exists():
                with suppress(BaseException):
                    shutil.rmtree(target, ignore_errors=True)
            raise
        return backup

    @staticmethod
    def _restore_backup(backup: Path, target: Path) -> None:
        if backup.exists():
            if target.exists():
                shutil.rmtree(target, ignore_errors=True)
            os.replace(backup, target)

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
        """Import explicit files by creating a catalog-like managed installation."""
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
        """Verify every installed artifact's path, size, and SHA-256 digest."""
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

    def _pocket_voice_state_references(
        self, installations: Sequence[Installation]
    ) -> dict[tuple[str, str, str, str, str], set[tuple[int | None, str | None]]]:
        references: dict[tuple[str, str, str, str, str], set[tuple[int | None, str | None]]] = {}

        def add_reference(
            bundle_id: str,
            voice_name: str,
            repository: str,
            revision: str,
            path: str,
            size: int | None,
            sha256: str | None,
        ) -> None:
            identity = (bundle_id, voice_name, repository, revision, path)
            references.setdefault(identity, set()).add((size, sha256))

        for installation in installations:
            if installation.system != "pocket":
                continue
            bundle_id = installation.metadata.get("bundle_name")
            if not isinstance(bundle_id, str) or not bundle_id:
                bundle_id = installation.id
            explicit_names: set[str] = set()
            states = installation.metadata.get("voice_states")
            if isinstance(states, Sequence) and not isinstance(states, (str, bytes, Mapping)):
                for state in states:
                    if not isinstance(state, Mapping):
                        continue
                    name = state.get("name")
                    source = state.get("source")
                    if not isinstance(name, str) or not isinstance(source, Mapping):
                        continue
                    repository = source.get("repository")
                    revision = source.get("revision")
                    path = source.get("path")
                    if not isinstance(repository, str) or not repository:
                        continue
                    if not isinstance(revision, str) or not revision:
                        continue
                    if not isinstance(path, str) or not path:
                        continue
                    size = state.get("size")
                    sha256 = state.get("sha256")
                    add_reference(
                        bundle_id,
                        name,
                        repository,
                        revision,
                        path,
                        size if type(size) is int and size > 0 else None,
                        sha256 if isinstance(sha256, str) else None,
                    )
                    explicit_names.add(name)

            names: set[str] = set()
            for field in ("predefined_voice_names", "predefined_voices"):
                raw_names = installation.metadata.get(field)
                if isinstance(raw_names, Sequence) and not isinstance(
                    raw_names, (str, bytes, Mapping)
                ):
                    names.update(name for name in raw_names if isinstance(name, str))
            if names - explicit_names:
                from .systems.pocket import (
                    PREDEFINED_VOICE_REPOSITORY,
                    PREDEFINED_VOICE_REVISION,
                )

                for name in names - explicit_names:
                    add_reference(
                        bundle_id,
                        name,
                        PREDEFINED_VOICE_REPOSITORY,
                        PREDEFINED_VOICE_REVISION,
                        f"languages/{bundle_id}/embeddings/{name}.safetensors",
                        None,
                        None,
                    )
        return references

    @staticmethod
    def _pocket_voice_state_key(
        key: Mapping[str, Any],
    ) -> tuple[tuple[str, str, str, str, str], tuple[int | None, str | None] | None] | None:
        bundle_id = key.get("bundle_id")
        voice_name = key.get("voice_name")
        repository = key.get("model_repo")
        revision = key.get("model_revision")
        asset_path = key.get("asset_path")
        if (
            key.get("system") != "pocket"
            or key.get("asset_kind") != "predefined_voice_state"
            or not isinstance(bundle_id, str)
            or not bundle_id
            or not isinstance(voice_name, str)
            or not voice_name
            or not isinstance(repository, str)
            or not repository
            or not isinstance(revision, str)
            or not revision
            or not isinstance(asset_path, str)
            or not asset_path
        ):
            return None
        identity = (bundle_id, voice_name, repository, revision, asset_path)
        has_expected_size = "expected_size" in key
        has_expected_sha = "expected_sha256" in key
        if has_expected_size != has_expected_sha:
            return None
        if not has_expected_size:
            expected = None
        else:
            size = key.get("expected_size")
            sha256 = key.get("expected_sha256")
            if type(size) is not int or size <= 0 or not isinstance(sha256, str):
                return None
            expected = (size, sha256)
        return identity, expected

    @staticmethod
    def _pocket_voice_state_is_referenced(
        record: Mapping[str, Any],
        parsed_key: tuple[
            tuple[str, str, str, str, str],
            tuple[int | None, str | None] | None,
        ],
        references: dict[
            tuple[str, str, str, str, str],
            set[tuple[int | None, str | None]],
        ],
    ) -> bool:
        identity, expected = parsed_key
        matching_references = references.get(identity, set())
        if expected is not None:
            return expected in matching_references
        size = record.get("size")
        sha256 = record.get("sha256")
        return any(
            pinned_size is None
            and pinned_sha is None
            or (pinned_size, pinned_sha) == (size, sha256)
            for pinned_size, pinned_sha in matching_references
        )

    def plan_gc(self, *, include_legacy: bool = False) -> GcPlan:
        """Build a conservative preview of data safe to collect."""
        with FileLock(self._gc_lock_path()):
            installations = self.installed()
            return self._build_gc_plan_unlocked(
                installations, self._scan_cache(), include_legacy=include_legacy
            )

    def _build_gc_plan_unlocked(
        self,
        installations: Sequence[Installation],
        scan: CacheScan,
        *,
        include_legacy: bool = False,
    ) -> GcPlan:
        """Build GC candidates from one installation snapshot and cache scan."""
        referenced_blobs = {
            artifact.sha256
            for installation in installations
            for artifact in installation.artifacts
            if artifact.sha256
        }
        file_by_path = {file.path: file for file in scan.files}
        candidates: list[tuple[CacheFileUsage, str, str]] = []
        for file in scan.files:
            if file.category == "blob" and len(file.path.relative_to(self.blobs).parts) == 2:
                if file.path.name not in referenced_blobs:
                    candidates.append(
                        (file, "orphan_blob", "No installed manifest references this blob.")
                    )
            elif include_legacy and file.category == "legacy":
                candidates.append(
                    (
                        file,
                        "legacy_cache",
                        "Explicit cleanup of the recognized retired Pocket prompt cache.",
                    )
                )

        state_root = self.root / "pocket-voice-states"
        state_references = self._pocket_voice_state_references(installations)
        for metadata_path in sorted(state_root.glob("*/*/*.json")):
            metadata_file = file_by_path.get(metadata_path)
            if metadata_file is None or metadata_file.category != "auxiliary":
                continue
            try:
                record = json.loads(metadata_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if not isinstance(record, dict) or not isinstance(record.get("key"), dict):
                continue
            parsed_key = self._pocket_voice_state_key(record["key"])
            if parsed_key is None or self._pocket_voice_state_is_referenced(
                record, parsed_key, state_references
            ):
                continue
            candidates.append(
                (
                    metadata_file,
                    "orphan_pocket_voice_state",
                    "No installed bundle references this Pocket voice-state record.",
                )
            )
            state_file = file_by_path.get(metadata_path.with_suffix(".safetensors"))
            if state_file is not None and state_file.category == "auxiliary":
                candidates.append(
                    (
                        state_file,
                        "orphan_pocket_voice_state",
                        "State payload belongs to an unreferenced Pocket voice-state record.",
                    )
                )

        def inode_key(file: CacheFileUsage) -> tuple[Any, ...]:
            if file.inode:
                return ("inode", file.device, file.inode)
            return ("path", os.fspath(file.path))

        all_link_counts: dict[tuple[Any, ...], int] = {}
        for file in scan.files:
            key = inode_key(file)
            all_link_counts[key] = all_link_counts.get(key, 0) + 1
        candidate_groups: dict[tuple[Any, ...], list[tuple[CacheFileUsage, str, str]]] = {}
        for candidate in candidates:
            candidate_groups.setdefault(inode_key(candidate[0]), []).append(candidate)

        planned: list[GcCandidate] = []
        accounted_inodes: set[tuple[Any, ...]] = set()
        for file, category, reason in sorted(candidates, key=lambda item: os.fspath(item[0].path)):
            key = inode_key(file)
            group = candidate_groups[key]
            group_paths = {entry[0].path for entry in group}
            all_linked_paths = all(
                inode_key(scanned_file) != key or scanned_file.path in group_paths
                for scanned_file in scan.files
            )
            link_count = max(candidate_file.link_count for candidate_file, _, _ in group)
            frees_inode = (
                all_linked_paths
                and len(group_paths) == all_link_counts.get(key, 0)
                and link_count <= len(group_paths)
            )
            allocated = 0
            if frees_inode and key not in accounted_inodes:
                allocated = file.allocated_bytes
                accounted_inodes.add(key)
            planned.append(
                GcCandidate(
                    category=category,
                    path=file.path,
                    apparent_bytes=file.apparent_bytes,
                    allocated_bytes=allocated,
                    reason=reason,
                )
            )

        return GcPlan(
            candidates=tuple(planned),
            apparent_bytes=sum(candidate.apparent_bytes for candidate in planned),
            allocated_bytes=sum(candidate.allocated_bytes for candidate in planned),
            include_legacy=include_legacy,
        )

    def gc(self, plan: GcPlan | None = None, *, include_legacy: bool = False) -> GcReport:
        """Collect unreachable cache data, revalidating a fresh plan under lock.

        A supplied plan is advisory only; candidate paths and reachability are
        recomputed while holding the GC lock before any deletion is attempted.
        """
        if plan is not None:
            include_legacy = plan.include_legacy

        with FileLock(self._gc_lock_path()):
            installations = self.installed()
            scan = self._scan_cache()
            current_plan = self._build_gc_plan_unlocked(
                installations, scan, include_legacy=include_legacy
            )
            referenced_blobs = {
                artifact.sha256
                for installation in installations
                for artifact in installation.artifacts
                if artifact.sha256
            }
            candidates_by_path = {
                candidate.path: candidate for candidate in current_plan.candidates
            }
            removed_paths: set[Path] = set()
            removed_stats: dict[Path, os.stat_result] = {}
            removed_blobs = 0
            removed_blob_bytes = 0
            removed_auxiliary = 0
            removed_auxiliary_bytes = 0
            removed_legacy_count = 0
            removed_legacy_bytes = 0
            removed_apparent_bytes = 0

            def unlink_regular(path: Path) -> os.stat_result | None:
                try:
                    current_stat = os.stat(path, follow_symlinks=False)
                except OSError:
                    return None
                if not stat_module.S_ISREG(current_stat.st_mode):
                    return None
                try:
                    path.unlink()
                except FileNotFoundError:
                    return None
                removed_paths.add(path)
                removed_stats[path] = current_stat
                return current_stat

            for candidate in current_plan.candidates:
                if candidate.category == "orphan_blob":
                    if candidate.path.name in referenced_blobs:
                        continue
                    file_stat = unlink_regular(candidate.path)
                    if file_stat is None:
                        continue
                    removed_blobs += 1
                    removed_blob_bytes += file_stat.st_size
                    removed_apparent_bytes += file_stat.st_size
                elif candidate.category == "legacy_cache":
                    try:
                        relative_parts = candidate.path.relative_to(self.root).parts
                    except ValueError:
                        continue
                    if not relative_parts or relative_parts[0] != "pocket-voice-prompts":
                        continue
                    file_stat = unlink_regular(candidate.path)
                    if file_stat is None:
                        continue
                    removed_legacy_count += 1
                    removed_legacy_bytes += file_stat.st_size
                    removed_apparent_bytes += file_stat.st_size

            state_root = self.root / "pocket-voice-states"
            state_references = self._pocket_voice_state_references(installations)
            metadata_candidates = sorted(
                candidate.path
                for candidate in current_plan.candidates
                if candidate.category == "orphan_pocket_voice_state"
                and candidate.path.suffix == ".json"
            )
            for metadata_path in metadata_candidates:
                cache_dir = metadata_path.parent
                lock_path = state_root / "locks" / f"{cache_dir.name}.lock"
                with FileLock(lock_path):
                    metadata_stat = None
                    try:
                        metadata_stat = os.stat(metadata_path, follow_symlinks=False)
                    except OSError:
                        continue
                    if not stat_module.S_ISREG(metadata_stat.st_mode):
                        continue
                    try:
                        record = json.loads(metadata_path.read_text(encoding="utf-8"))
                    except (OSError, json.JSONDecodeError):
                        continue
                    if not isinstance(record, dict) or not isinstance(record.get("key"), dict):
                        continue
                    parsed_key = self._pocket_voice_state_key(record["key"])
                    if parsed_key is None or self._pocket_voice_state_is_referenced(
                        record, parsed_key, state_references
                    ):
                        continue

                    state_path = metadata_path.with_suffix(".safetensors")
                    removed_record = False
                    for cached_path in (state_path, metadata_path):
                        if cached_path not in candidates_by_path:
                            continue
                        file_stat = unlink_regular(cached_path)
                        if file_stat is None:
                            continue
                        removed_auxiliary_bytes += file_stat.st_size
                        removed_apparent_bytes += file_stat.st_size
                        if cached_path == metadata_path:
                            removed_record = True
                    if removed_record:
                        removed_auxiliary += 1
                    with suppress(OSError):
                        cache_dir.rmdir()
                        cache_dir.parent.rmdir()

            blob_paths = [
                candidate.path
                for candidate in current_plan.candidates
                if candidate.category == "orphan_blob"
            ]
            state_paths = [
                candidate.path
                for candidate in current_plan.candidates
                if candidate.category == "orphan_pocket_voice_state"
            ]
            legacy_paths = [
                candidate.path
                for candidate in current_plan.candidates
                if candidate.category == "legacy_cache"
            ]
            self._prune_empty_dirs(blob_paths, self.blobs)
            self._prune_empty_dirs(state_paths, state_root)
            self._prune_empty_dirs(legacy_paths, self.root)

            planned_by_inode: dict[tuple[Any, ...], list[GcCandidate]] = {}
            for candidate in current_plan.candidates:
                file = next((item for item in scan.files if item.path == candidate.path), None)
                if file is None:
                    continue
                key = (
                    ("inode", file.device, file.inode)
                    if file.inode
                    else ("path", os.fspath(file.path))
                )
                planned_by_inode.setdefault(key, []).append(candidate)
            removed_allocated_bytes = sum(
                sum(candidate.allocated_bytes for candidate in group)
                for group in planned_by_inode.values()
                if group and all(candidate.path in removed_paths for candidate in group)
            )

            return GcReport(
                removed_blobs=removed_blobs,
                removed_bytes=removed_blob_bytes,
                removed_auxiliary_count=removed_auxiliary,
                removed_auxiliary_bytes=removed_auxiliary_bytes,
                removed_candidates=len(removed_paths),
                removed_apparent_bytes=removed_apparent_bytes,
                removed_allocated_bytes=removed_allocated_bytes,
                removed_legacy_count=removed_legacy_count,
                removed_legacy_bytes=removed_legacy_bytes,
            )

    @staticmethod
    def _prune_empty_dirs(paths: Sequence[Path], boundary: Path) -> None:
        """Remove only empty descendants of a known cache subtree."""
        for path in sorted(set(paths), key=os.fspath, reverse=True):
            directory = path.parent
            while directory != boundary and boundary in directory.parents:
                with suppress(OSError):
                    directory.rmdir()
                directory = directory.parent

    def _scan_cache(self) -> CacheScan:
        """Scan every regular cache file once without following symlinks."""
        root_categories = {
            "blobs": "blobs",
            "installs": "installations",
            "catalogs": "catalogs",
            "locks": "locks",
            "pocket-voice-states": "auxiliary",
            "pocket-voice-prompts": "legacy",
        }
        scan_errors: list[str] = []
        root_names: set[str] = set()
        try:
            with os.scandir(self.root) as entries:
                root_names.update(entry.name for entry in entries)
        except OSError as exc:
            scan_errors.append(f"{self.root}: {exc}")

        files: list[CacheFileUsage] = []
        top_level_totals: dict[str, list[int]] = {}
        seen_inodes: set[tuple[Any, ...]] = set()
        apparent_bytes = 0
        unique_apparent_bytes = 0
        allocated_bytes = 0

        def on_walk_error(exc: OSError) -> None:
            location = getattr(exc, "filename", None) or self.root
            scan_errors.append(f"{location}: {exc}")

        for current, directories, filenames in os.walk(
            self.root, topdown=True, onerror=on_walk_error, followlinks=False
        ):
            current_path = Path(current)
            directories[:] = [
                name for name in sorted(directories) if not os.path.islink(current_path / name)
            ]
            for filename in sorted(filenames):
                path = current_path / filename
                try:
                    file_stat = os.stat(path, follow_symlinks=False)
                except OSError as exc:
                    scan_errors.append(f"{path}: {exc}")
                    continue
                if not stat_module.S_ISREG(file_stat.st_mode):
                    continue

                try:
                    relative_parts = path.relative_to(self.root).parts
                except ValueError:
                    continue
                top_name = relative_parts[0] if relative_parts else path.name
                if top_name == "blobs":
                    category = "blob"
                elif top_name == "installs":
                    category = (
                        "metadata"
                        if filename in {_INSTALL_MANIFEST_FILENAME, "manifest.json"}
                        else "install"
                    )
                elif top_name == "catalogs":
                    category = "catalog"
                elif top_name == "locks" or (
                    top_name == "pocket-voice-states" and "locks" in relative_parts[1:]
                ):
                    category = "lock"
                elif top_name == "pocket-voice-states":
                    category = "auxiliary"
                elif top_name == "pocket-voice-prompts":
                    category = "legacy"
                else:
                    category = "other"

                size = file_stat.st_size
                allocated = _allocated_bytes(file_stat)
                device = int(getattr(file_stat, "st_dev", 0))
                inode = int(getattr(file_stat, "st_ino", 0))
                link_count = int(getattr(file_stat, "st_nlink", 1))
                files.append(
                    CacheFileUsage(
                        path=path,
                        category=category,
                        apparent_bytes=size,
                        allocated_bytes=allocated,
                        device=device,
                        inode=inode,
                        link_count=link_count,
                    )
                )
                apparent_bytes += size
                totals = top_level_totals.setdefault(top_name, [0, 0, 0, 0])
                totals[0] += 1
                totals[1] += size
                inode_key: tuple[Any, ...] = (
                    ("inode", device, inode) if inode else ("path", os.fspath(path))
                )
                if inode_key not in seen_inodes:
                    seen_inodes.add(inode_key)
                    unique_apparent_bytes += size
                    allocated_bytes += allocated
                    totals[2] += size
                    totals[3] += allocated

        known_roots = set(root_categories)
        top_level_usage = tuple(
            CachePathUsage(
                path=self.root / name,
                category=root_categories.get(name, "unmanaged"),
                file_count=top_level_totals.get(name, [0, 0, 0, 0])[0],
                apparent_bytes=top_level_totals.get(name, [0, 0, 0, 0])[1],
                unique_apparent_bytes=top_level_totals.get(name, [0, 0, 0, 0])[2],
                allocated_bytes=top_level_totals.get(name, [0, 0, 0, 0])[3],
            )
            for name in sorted(known_roots | root_names)
        )
        unmanaged_paths = tuple(self.root / name for name in sorted(root_names - known_roots))
        legacy_paths = tuple(
            self.root / name for name in sorted(root_names & {"pocket-voice-prompts"})
        )

        try:
            disk = shutil.disk_usage(self.root)
            filesystem = CacheFilesystemUsage(
                total_bytes=disk.total,
                used_bytes=disk.used,
                free_bytes=disk.free,
            )
        except OSError as exc:
            scan_errors.append(f"filesystem capacity for {self.root}: {exc}")
            filesystem = CacheFilesystemUsage(0, 0, 0)

        return CacheScan(
            root=self.root,
            filesystem=filesystem,
            files=tuple(files),
            top_level_usage=top_level_usage,
            apparent_bytes=apparent_bytes,
            unique_apparent_bytes=unique_apparent_bytes,
            allocated_bytes=allocated_bytes,
            unmanaged_paths=unmanaged_paths,
            legacy_paths=legacy_paths,
            errors=tuple(scan_errors),
        )

    def usage(self) -> CacheUsage:
        """Compute logical, whole-root, orphan, and auxiliary cache usage."""
        installations = self.installed()
        scan = self._scan_cache()
        file_by_path = {file.path: file for file in scan.files}
        files_by_category: dict[str, list[CacheFileUsage]] = {}
        for file in scan.files:
            files_by_category.setdefault(file.category, []).append(file)

        install_logical = sum(
            artifact.size for installation in installations for artifact in installation.artifacts
        )
        referenced = {
            artifact.sha256
            for installation in installations
            for artifact in installation.artifacts
            if artifact.sha256
        }
        artifact_paths = {
            artifact.path for installation in installations for artifact in installation.artifacts
        }

        def inode_key(file: CacheFileUsage) -> tuple[Any, ...]:
            if file.inode:
                return ("inode", file.device, file.inode)
            return ("path", os.fspath(file.path))

        # Preserve the historical narrower metric: blobs, manifest-listed install artifacts,
        # and Pocket state files, deduplicated by inode (but not manifests/catalogs/locks).
        legacy_counted_files = [
            file
            for file in scan.files
            if file.category == "blob"
            or file.category == "auxiliary"
            or (file.category == "install" and file.path in artifact_paths)
        ]
        seen_legacy_inodes: set[tuple[Any, ...]] = set()
        unique_file_bytes = 0
        for file in legacy_counted_files:
            key = inode_key(file)
            if key not in seen_legacy_inodes:
                seen_legacy_inodes.add(key)
                unique_file_bytes += file.apparent_bytes

        blob_files = [
            file
            for file in files_by_category.get("blob", ())
            if len(file.path.relative_to(self.blobs).parts) == 2
        ]
        blob_apparent = sum(file.apparent_bytes for file in blob_files)
        referenced_blob_files = [file for file in blob_files if file.path.name in referenced]
        orphan_blob_files = [file for file in blob_files if file.path.name not in referenced]
        referenced_blob_bytes = sum(file.apparent_bytes for file in referenced_blob_files)
        orphan_blob_bytes = sum(file.apparent_bytes for file in orphan_blob_files)

        def unique_allocated(files: Sequence[CacheFileUsage]) -> int:
            seen: set[tuple[Any, ...]] = set()
            allocated = 0
            for file in files:
                key = inode_key(file)
                if key not in seen:
                    seen.add(key)
                    allocated += file.allocated_bytes
            return allocated

        referenced_blob_allocated = unique_allocated(referenced_blob_files)

        hardlinked_count = 0
        hardlinked_bytes = 0
        copied_count = 0
        copied_bytes = 0
        missing_blob_count = 0
        for installation in installations:
            for artifact in installation.artifacts:
                if not artifact.sha256:
                    missing_blob_count += 1
                    continue
                blob_path = self.blobs / artifact.sha256[:2] / artifact.sha256
                blob_file = file_by_path.get(blob_path)
                if blob_file is None:
                    missing_blob_count += 1
                    continue
                install_file = file_by_path.get(artifact.path)
                if install_file is None:
                    continue
                if (
                    blob_file.inode != 0
                    and install_file.inode == blob_file.inode
                    and install_file.device == blob_file.device
                ):
                    hardlinked_count += 1
                    hardlinked_bytes += install_file.apparent_bytes
                else:
                    copied_count += 1
                    copied_bytes += install_file.apparent_bytes

        auxiliary_files = files_by_category.get("auxiliary", [])
        auxiliary_bytes = sum(file.apparent_bytes for file in auxiliary_files)
        pocket_state_count = 0
        pocket_state_bytes = 0
        orphan_auxiliary_count = 0
        orphan_auxiliary_bytes = 0

        state_references = self._pocket_voice_state_references(installations)
        state_root = self.root / "pocket-voice-states"
        for metadata_path in sorted(state_root.glob("*/*/*.json")):
            metadata_file = file_by_path.get(metadata_path)
            if metadata_file is None:
                continue
            try:
                record = json.loads(metadata_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if not isinstance(record, dict) or not isinstance(record.get("key"), dict):
                continue
            parsed_key = self._pocket_voice_state_key(record["key"])
            if parsed_key is None:
                continue

            state_file = file_by_path.get(metadata_path.with_suffix(".safetensors"))
            if state_file is not None:
                pocket_state_count += 1
                pocket_state_bytes += state_file.apparent_bytes
            record_files = [metadata_file]
            if state_file is not None:
                record_files.append(state_file)
            record_size = sum(file.apparent_bytes for file in record_files)
            if not self._pocket_voice_state_is_referenced(record, parsed_key, state_references):
                orphan_auxiliary_count += 1
                orphan_auxiliary_bytes += record_size

        other_cache_bytes = 0
        seen_other_inodes = {inode_key(file) for file in scan.files if file.category != "other"}
        for file in files_by_category.get("other", ()):
            key = inode_key(file)
            if key in seen_other_inodes:
                continue
            seen_other_inodes.add(key)
            other_cache_bytes += file.apparent_bytes

        category_bytes = {
            category: sum(file.apparent_bytes for file in category_files)
            for category, category_files in files_by_category.items()
        }
        gc_plan = self._build_gc_plan_unlocked(installations, scan)
        orphan_blob_allocated = sum(
            candidate.allocated_bytes
            for candidate in gc_plan.candidates
            if candidate.category == "orphan_blob"
        )
        return CacheUsage(
            root=self.root,
            installation_count=len(installations),
            install_logical_bytes=install_logical,
            blob_apparent_bytes=blob_apparent,
            catalog_bytes=category_bytes.get("catalog", 0),
            unique_file_bytes=unique_file_bytes,
            orphan_blob_count=len(orphan_blob_files),
            orphan_blob_bytes=orphan_blob_bytes,
            auxiliary_bytes=auxiliary_bytes,
            pocket_voice_state_count=pocket_state_count,
            pocket_voice_state_bytes=pocket_state_bytes,
            orphan_auxiliary_count=orphan_auxiliary_count,
            orphan_auxiliary_bytes=orphan_auxiliary_bytes,
            filesystem_total_bytes=scan.filesystem.total_bytes,
            filesystem_used_bytes=scan.filesystem.used_bytes,
            filesystem_free_bytes=scan.filesystem.free_bytes,
            cache_apparent_bytes=scan.apparent_bytes,
            cache_unique_apparent_bytes=scan.unique_apparent_bytes,
            cache_allocated_bytes=scan.allocated_bytes,
            install_apparent_bytes=category_bytes.get("install", 0),
            metadata_bytes=category_bytes.get("metadata", 0),
            lock_bytes=category_bytes.get("lock", 0),
            other_cache_bytes=other_cache_bytes,
            legacy_cache_bytes=category_bytes.get("legacy", 0),
            hardlinked_install_artifact_count=hardlinked_count,
            hardlinked_install_bytes=hardlinked_bytes,
            copied_install_artifact_count=copied_count,
            copied_install_bytes=copied_bytes,
            missing_blob_artifact_count=missing_blob_count,
            referenced_blob_count=len(referenced_blob_files),
            referenced_blob_bytes=referenced_blob_bytes,
            referenced_blob_allocated_bytes=referenced_blob_allocated,
            orphan_blob_allocated_bytes=orphan_blob_allocated,
            reclaimable_now_bytes=gc_plan.apparent_bytes,
            reclaimable_now_allocated_bytes=gc_plan.allocated_bytes,
            unmanaged_paths=scan.unmanaged_paths,
            legacy_paths=scan.legacy_paths,
            top_level_usage=scan.top_level_usage,
            scan_errors=scan.errors,
        )

    def replace(
        self,
        existing: Installation,
        item: CatalogItem,
        *,
        progress: ProgressCallback | None = None,
    ) -> Installation:
        """Atomically replace an existing installation with a new catalog item.

        Reuses the same storage path. Old data is removed after the new
        installation is verified.
        """
        storage_id = existing.storage_id or existing.id
        system = existing.system
        target = self.install_path(system, storage_id)

        with FileLock(self._install_lock_path(system, storage_id)):
            # Use the same staging + atomic replace workflow as install
            expected_bytes = self._expected_install_bytes(item.artifacts)
            staging_path = target.parent / f".{storage_id}-{uuid.uuid4().hex}"
            backup: Path | None = None
            committed = False
            try:
                staging_path.mkdir()
                self._emit(
                    progress,
                    AssetProgress(
                        "install_started", item.ref, total=expected_bytes, target=str(target)
                    ),
                )
                installed: list[InstalledArtifact] = []
                with FileLock(self._gc_lock_path()):
                    for artifact in item.artifacts:
                        artifact_target = str(target / artifact.filename)
                        blob, sha256 = self._materialize_artifact(
                            artifact,
                            ref=item.ref,
                            progress=progress,
                            target=artifact_target,
                        )
                        destination = staging_path / artifact.filename
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
                    manifest_data = {
                        "schema": MANIFEST_SCHEMA_VERSION,
                        "system": item.system,
                        "id": item.id,
                        "storage_id": storage_id,
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
                                "component": a.component,
                                "format": a.format,
                                "metadata": dict(a.metadata),
                            }
                            for a in installed
                        ],
                    }
                    (staging_path / _INSTALL_MANIFEST_FILENAME).write_text(
                        json.dumps(manifest_data, indent=2, sort_keys=True) + "\n", encoding="utf-8"
                    )
                    backup = self._commit_staging(staging_path, target)
                    committed = True
                installation = self.get(system, storage_id)
                self._emit(
                    progress, AssetProgress("install_completed", item.ref, target=str(target))
                )
                if backup is not None:
                    shutil.rmtree(backup, ignore_errors=True)
                return installation
            except BaseException:
                with suppress(BaseException):
                    shutil.rmtree(staging_path, ignore_errors=True)
                with suppress(BaseException):
                    if backup is not None:
                        self._restore_backup(backup, target)
                    elif committed and target.exists():
                        shutil.rmtree(target, ignore_errors=True)
                with suppress(BaseException):
                    self._emit(
                        progress,
                        AssetProgress("install_failed", item.ref, target=str(target)),
                    )
                raise

    def _materialize_artifact(
        self,
        artifact: Artifact,
        *,
        ref: str,
        progress: ProgressCallback | None,
        target: str,
    ) -> tuple[Path, str]:
        source_metadata = artifact.metadata.get("source")
        hf_source = None
        if (
            isinstance(source_metadata, Mapping)
            and source_metadata.get("provider") == "huggingface"
        ):
            try:
                hf_source = HuggingFaceSource(
                    repository=source_metadata["repository"],
                    revision=source_metadata["revision"],
                    path=source_metadata["path"],
                    gated=source_metadata.get("gated", False),
                )
            except (KeyError, TypeError, ValueError) as exc:
                raise AssetDownloadError(
                    f"Invalid Hugging Face source metadata for {artifact.filename!r}"
                ) from exc
        if not artifact.url and hf_source is None:
            raise AssetNotFoundError(f"Artifact {artifact.filename!r} has no URL")
        if artifact.sha256:
            known_blob = self._blob_path(artifact.sha256)
            if known_blob.is_file():
                self._emit(
                    progress,
                    AssetProgress(
                        "verify_started", ref, artifact.filename, role=artifact.role, target=target
                    ),
                )
                verify_file(
                    known_blob,
                    expected_size=artifact.size,
                    sha256=artifact.sha256,
                    md5=artifact.md5,
                )
                self._emit(
                    progress,
                    AssetProgress(
                        "verify_completed",
                        ref,
                        artifact.filename,
                        role=artifact.role,
                        target=target,
                    ),
                )
                self._emit(
                    progress,
                    AssetProgress(
                        "artifact_cached",
                        ref,
                        artifact.filename,
                        role=artifact.role,
                        target=target,
                        total=artifact.size,
                    ),
                )
                return known_blob, artifact.sha256

        self._emit(
            progress,
            AssetProgress(
                "download_started",
                ref,
                artifact.filename,
                role=artifact.role,
                target=target,
                total=artifact.size,
            ),
        )
        with tempfile.TemporaryDirectory(prefix="onnxvoice-download-") as temp_dir:
            if hf_source is not None:
                if self.offline:
                    raise OfflineError(
                        f"Network access disabled while fetching {ref}:{artifact.filename}"
                    )
                temp = download_huggingface_file(
                    hf_source, local_dir=Path(temp_dir), offline=self.offline
                )
            else:
                assert artifact.url is not None
                temp = Path(temp_dir) / Path(artifact.filename).name
                self._download(
                    artifact.url,
                    temp,
                    ref=ref,
                    artifact=artifact.filename,
                    role=artifact.role,
                    target=target,
                    progress=progress,
                    expected_size=artifact.size,
                )
            self._emit(
                progress,
                AssetProgress(
                    "verify_started", ref, artifact.filename, role=artifact.role, target=target
                ),
            )
            verify_file(
                temp,
                expected_size=artifact.size,
                sha256=artifact.sha256,
                md5=artifact.md5,
            )
            self._emit(
                progress,
                AssetProgress(
                    "verify_completed", ref, artifact.filename, role=artifact.role, target=target
                ),
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
                    role=artifact.role,
                    target=target,
                ),
            )
            return blob, sha256

    def _download(
        self,
        source: str,
        destination: Path,
        *,
        expected_size: int | None,
        ref: str,
        artifact: str,
        role: str,
        target: str,
        progress: ProgressCallback | None,
    ) -> None:
        if source.startswith("file://"):
            from urllib.parse import urlparse
            from urllib.request import url2pathname

            parsed = urlparse(source)
            shutil.copy2(Path(url2pathname(parsed.path)), destination)
            self._emit(
                progress,
                AssetProgress(
                    "download_progress",
                    ref,
                    artifact,
                    completed=destination.stat().st_size,
                    total=expected_size,
                    role=role,
                    target=target,
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
                total_bytes = int(total) if total and total.isdigit() else expected_size
                completed = 0
                while chunk := response.read(1024 * 1024):
                    out.write(chunk)
                    completed += len(chunk)
                    self._emit(
                        progress,
                        AssetProgress(
                            "download_progress",
                            ref,
                            artifact,
                            completed,
                            total_bytes,
                            role=role,
                            target=target,
                        ),
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
        hardlink = getattr(os, "link", None)
        if hardlink is not None:
            try:
                hardlink(source, destination)
                return
            except (OSError, NotImplementedError):
                pass
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
        if schema not in {1, 2, MANIFEST_SCHEMA_VERSION}:
            raise ManifestError(f"Unsupported manifest schema {schema!r} in {path}")
        try:
            system = validate_safe_component(data["system"], field_name="system")
            raw_id = validate_safe_component(data["id"], field_name="item id")
        except (KeyError, ValueError) as exc:
            raise UnsafePathError(f"Invalid manifest identity in {path}: {exc}") from exc

        # Schema 3 introduces separate storage_id; schema 1/2 used a single id
        # that may have been a hashed storage key for explicit distributions.
        storage_id = data.get("storage_id")
        if storage_id is not None:
            # Schema 3: id is canonical, storage_id is physical
            try:
                storage_id = validate_safe_component(storage_id, field_name="storage id")
            except ValueError as exc:
                raise UnsafePathError(f"Invalid storage id in {path}: {exc}") from exc
            item_id = raw_id
        else:
            # Schema 1/2: recover canonical id from legacy hashed form
            item_id, storage_id = cls._recover_legacy_identity(data, raw_id, path)

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
            storage_id=storage_id,
        )

    _LEGACY_DIST_PATTERN = __import__("re").compile(r"^(.+)--dist-[0-9a-f]{16}$")

    @classmethod
    def _recover_legacy_identity(
        cls, data: dict[str, Any], raw_id: str, path: Path
    ) -> tuple[str, str | None]:
        """Recover canonical id from legacy schema 1/2 manifests.

        A schema-2 manifest written by the old explicit-distribution code may
        have stored the hashed storage key as ``id``.  Detect and recover the
        canonical prefix when metadata confirms it.
        """
        metadata = data.get("metadata") or {}
        cache_id = metadata.get("cache_id")
        has_selected = "selected_distribution" in metadata
        match = cls._LEGACY_DIST_PATTERN.match(raw_id)
        if match and has_selected and cache_id == raw_id:
            canonical = match.group(1)
            return canonical, raw_id
        return raw_id, None

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
