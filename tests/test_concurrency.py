from __future__ import annotations

import hashlib
import multiprocessing
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch

import pytest

import onnxvoice.store as store_module
from onnxvoice.catalog import CatalogClient
from onnxvoice.errors import IntegrityError, LockError
from onnxvoice.store import AssetStore, FileLock
from onnxvoice.types import Artifact, AssetProgress, CatalogItem


def _install_worker(cache: str, source: str, result_queue) -> None:
    path = Path(source)
    item = CatalogItem(
        system="test",
        id="shared",
        kind="model",
        artifacts=(
            Artifact(
                "model",
                "voice.onnx",
                path.as_uri(),
                path.stat().st_size,
                hashlib.sha256(path.read_bytes()).hexdigest(),
            ),
        ),
    )
    try:
        result_queue.put(AssetStore(cache).install(item).ref)
    except Exception as exc:  # pragma: no cover - assertion reports child failures
        result_queue.put(f"error: {exc!r}")


def test_concurrent_installers_converge(tmp_path):
    source = tmp_path / "voice.onnx"
    source.write_bytes(b"same model")
    cache = tmp_path / "cache"
    # Explicit spawn coverage also runs on POSIX, approximating Windows process startup.
    context = multiprocessing.get_context("spawn")
    result_queue = context.Queue()
    processes = [
        context.Process(target=_install_worker, args=(str(cache), str(source), result_queue))
        for _ in range(2)
    ]
    for process in processes:
        process.start()
    for process in processes:
        process.join(timeout=10)
        assert process.exitcode == 0

    assert [result_queue.get(timeout=2) for _ in processes] == ["test:shared", "test:shared"]
    store = AssetStore(cache)
    assert store.get("test", "shared").artifact("model").path.read_bytes() == b"same model"
    assert list((cache / "installs" / "test").glob(".shared-*")) == []


def test_progress_events_are_ordered_and_failure_cleans_staging(tmp_path):
    source = tmp_path / "voice.onnx"
    source.write_bytes(b"model")
    events: list[AssetProgress] = []
    store = AssetStore(tmp_path / "cache")
    item = CatalogItem(
        system="test",
        id="broken",
        kind="model",
        artifacts=(Artifact("model", "voice.onnx", source.as_uri(), size=99),),
    )

    with pytest.raises(IntegrityError):
        store.install(item, progress=events.append)

    assert [event.phase for event in events] == [
        "install_started",
        "download_started",
        "download_progress",
        "verify_started",
        "install_failed",
    ]
    assert list((tmp_path / "cache" / "installs" / "test").glob(".broken-*")) == []


def _hold_lock_worker(path: str, acquired, release) -> None:
    with FileLock(path, timeout=5, poll_interval=0.01):
        acquired.set()
        release.wait(timeout=5)


def _crash_while_holding_lock(path: str, acquired) -> None:
    lock = FileLock(path)
    lock.acquire()
    acquired.set()
    os._exit(0)


def _fork_lock_contender(path: str, started, acquired) -> None:
    started.set()
    with FileLock(path, timeout=3, poll_interval=0.01):
        acquired.set()


def test_file_lock_serializes_threads(tmp_path):
    path = tmp_path / "thread.lock"
    counter_lock = threading.Lock()
    inside = 0
    max_inside = 0

    def worker():
        nonlocal inside, max_inside
        for _ in range(10):
            with FileLock(path, timeout=3):
                with counter_lock:
                    inside += 1
                    max_inside = max(max_inside, inside)
                time.sleep(0.001)
                with counter_lock:
                    inside -= 1

    with ThreadPoolExecutor(max_workers=6) as executor:
        list(executor.map(lambda _: worker(), range(6)))

    assert max_inside == 1
    assert path.is_file()  # Advisory lock files are stable coordination files.


def test_file_lock_releases_local_mutex_after_open_failure(tmp_path, monkeypatch):
    path = tmp_path / "open-failure.lock"
    real_open = os.open

    def fail_lock_open(file, flags, *args, **kwargs):
        if Path(file) == path:
            raise PermissionError("synthetic lock-file open failure")
        return real_open(file, flags, *args, **kwargs)

    monkeypatch.setattr(store_module.os, "open", fail_lock_open)
    with pytest.raises(LockError, match="Could not acquire lock"):
        FileLock(path).acquire()
    monkeypatch.undo()

    with FileLock(path):
        pass
    assert path.is_file()


def test_file_lock_timeout_uses_one_budget_across_local_and_os_wait(tmp_path):
    context = multiprocessing.get_context("spawn")
    path = str(tmp_path / "one-budget.lock")
    owner_acquired, release_owner = context.Event(), context.Event()
    owner = context.Process(target=_hold_lock_worker, args=(path, owner_acquired, release_owner))
    owner.start()
    assert owner_acquired.wait(3)

    first_waiting = threading.Event()
    original_try_os_lock = store_module._try_os_lock

    def observe_first_wait(fd):
        if threading.current_thread().name == "first-lock-contender":
            first_waiting.set()
        return original_try_os_lock(fd)

    results: dict[str, float] = {}

    def contender(name: str, timeout: float) -> None:
        started = time.monotonic()
        with pytest.raises(LockError):
            FileLock(path, timeout=timeout, poll_interval=0.01).acquire()
        results[name] = time.monotonic() - started

    first = threading.Thread(name="first-lock-contender", target=contender, args=("first", 0.5))
    second = threading.Thread(target=contender, args=("second", 0.9))
    try:
        with patch.object(store_module, "_try_os_lock", side_effect=observe_first_wait):
            first.start()
            assert first_waiting.wait(2)
            second.start()
            first.join(2)
            second.join(2)
        assert not first.is_alive()
        assert not second.is_alive()
        assert 0.45 <= results["first"] < 0.8
        # The second thread spends the first contender's wait under the same
        # timeout budget rather than receiving another full timeout afterward.
        assert 0.75 <= results["second"] < 1.15
    finally:
        release_owner.set()
        owner.join(3)
        if owner.is_alive():
            owner.terminate()
            owner.join(2)
    assert owner.exitcode == 0


def test_file_lock_does_not_break_a_live_owner_after_stale_after(tmp_path):
    context = multiprocessing.get_context("spawn")
    path = str(tmp_path / "old-live-owner.lock")
    owner_acquired, release_owner = context.Event(), context.Event()
    owner = context.Process(target=_hold_lock_worker, args=(path, owner_acquired, release_owner))
    owner.start()
    assert owner_acquired.wait(3)
    try:
        with pytest.raises(LockError, match="Timed out"):
            FileLock(path, timeout=0.15, poll_interval=0.01, stale_after=0).acquire()
        assert owner.is_alive()
    finally:
        release_owner.set()
        owner.join(3)
        if owner.is_alive():
            owner.terminate()
            owner.join(2)
    assert owner.exitcode == 0
    with FileLock(path, timeout=1):
        pass


def test_file_lock_released_when_spawned_owner_exits(tmp_path):
    context = multiprocessing.get_context("spawn")
    path = str(tmp_path / "crashed-owner.lock")
    acquired = context.Event()
    owner = context.Process(target=_crash_while_holding_lock, args=(path, acquired))
    owner.start()
    assert acquired.wait(3)
    owner.join(3)
    assert owner.exitcode == 0

    with FileLock(path, timeout=1):
        pass


def test_file_lock_fork_child_does_not_inherit_parent_lock_state(tmp_path):
    if "fork" not in multiprocessing.get_all_start_methods():
        pytest.skip("fork start method is unavailable")
    context = multiprocessing.get_context("fork")
    path = str(tmp_path / "fork.lock")
    started, acquired = context.Event(), context.Event()
    child = context.Process(target=_fork_lock_contender, args=(str(path), started, acquired))

    with FileLock(path):
        child.start()
        assert started.wait(3)
        time.sleep(0.1)
        assert not acquired.is_set()
    assert acquired.wait(3), "child inherited a local mutex or lock-file handle"
    child.join(3)
    assert child.exitcode == 0


class _CountingCatalogHandler(BaseHTTPRequestHandler):
    request_count = 0
    request_lock = threading.Lock()
    payload = b"{}"

    def do_GET(self):
        with type(self).request_lock:
            type(self).request_count += 1
        time.sleep(0.1)
        self.send_response(200)
        self.send_header("Content-Length", str(len(type(self).payload)))
        self.end_headers()
        self.wfile.write(type(self).payload)

    def log_message(self, format, *args):
        pass


def _catalog_spawn_worker(cache_root, source, refresh, barrier, result_queue):
    client = CatalogClient(
        cache_dir=cache_root,
        sources={"test": source},
        ttl_seconds=3600,
    )
    original_generation = CatalogClient._cache_generation
    initial_snapshot = True

    def synchronized_generation(path):
        nonlocal initial_snapshot
        generation = original_generation(path)
        if initial_snapshot:
            initial_snapshot = False
            barrier.wait(timeout=15)
        return generation

    try:
        with patch.object(
            CatalogClient, "_cache_generation", staticmethod(synchronized_generation)
        ):
            result_queue.put(("ok", client.load_raw("test", refresh=refresh)))
    except Exception as exc:  # pragma: no cover - surfaced by the parent assertion
        result_queue.put(("error", repr(exc)))
        raise


@pytest.mark.parametrize("refresh", [False, True], ids=["cache-miss", "refresh"])
def test_catalog_concurrent_loads_coalesce_across_threads(tmp_path, refresh):
    cache_root = tmp_path / "thread-catalog-cache"
    client = CatalogClient(cache_dir=cache_root, sources={"test": "counted-source"})
    cache_path = client._cache_path("test")
    original_payload = b'{"generation":"old"}'
    expected_payload = b'{"generation":"new"}'
    if refresh:
        cache_path.write_bytes(original_payload)
        os.utime(cache_path, ns=(1_000_000_000, 1_000_000_000))

    count = 0
    count_lock = threading.Lock()
    source_started = threading.Event()
    allow_source = threading.Event()

    def read_source(_client, _source):
        nonlocal count
        with count_lock:
            count += 1
        source_started.set()
        assert allow_source.wait(5)
        return expected_payload

    original_generation = CatalogClient._cache_generation
    barrier = threading.Barrier(6)
    thread_state = threading.local()

    def synchronized_generation(path):
        generation = original_generation(path)
        if not getattr(thread_state, "snapshotted", False):
            thread_state.snapshotted = True
            barrier.wait(timeout=5)
        return generation

    with (
        patch.object(CatalogClient, "_read_source", read_source),
        patch.object(CatalogClient, "_cache_generation", staticmethod(synchronized_generation)),
        ThreadPoolExecutor(max_workers=6) as executor,
    ):
        futures = [executor.submit(client.load_raw, "test", refresh=refresh) for _ in range(6)]
        assert source_started.wait(5)
        if refresh:
            assert cache_path.read_bytes() == original_payload
        else:
            assert not cache_path.exists()
        allow_source.set()
        results = [future.result(timeout=5) for future in futures]
    assert count == 1
    assert results == [{"generation": "new"}] * 6
    assert cache_path.read_bytes() == expected_payload
    assert not cache_path.with_suffix(".json.tmp").exists()


@pytest.mark.parametrize("refresh", [False, True], ids=["cache-miss", "refresh"])
def test_catalog_concurrent_loads_coalesce_across_spawned_processes(tmp_path, refresh):
    cache_root = tmp_path / "spawn-catalog-cache"
    cache_path = cache_root / "catalogs" / "test.json"
    original_payload = b'{"generation":"old"}'
    expected_payload = b'{"generation":"new"}'
    if refresh:
        cache_path.parent.mkdir(parents=True)
        cache_path.write_bytes(original_payload)
        os.utime(cache_path, ns=(1_000_000_000, 1_000_000_000))

    _CountingCatalogHandler.request_count = 0
    _CountingCatalogHandler.payload = expected_payload
    server = ThreadingHTTPServer(("127.0.0.1", 0), _CountingCatalogHandler)
    server.daemon_threads = True
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()
    source = f"http://127.0.0.1:{server.server_port}/catalog.json"

    context = multiprocessing.get_context("spawn")
    barrier = context.Barrier(4)
    result_queue = context.Queue()
    processes = [
        context.Process(
            target=_catalog_spawn_worker,
            args=(str(cache_root), source, refresh, barrier, result_queue),
        )
        for _ in range(4)
    ]
    try:
        for process in processes:
            process.start()
        results = [result_queue.get(timeout=20) for _ in processes]
        for process in processes:
            process.join(timeout=20)
            assert process.exitcode == 0
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
                process.join(timeout=2)
        server.shutdown()
        server.server_close()
        server_thread.join(timeout=2)

    assert results == [("ok", {"generation": "new"})] * 4
    assert _CountingCatalogHandler.request_count == 1
    assert cache_path.read_bytes() == expected_payload
    assert not cache_path.with_suffix(".json.tmp").exists()
