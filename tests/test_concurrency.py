from __future__ import annotations

import hashlib
import multiprocessing
from pathlib import Path

import pytest

from onnxvoice.errors import IntegrityError
from onnxvoice.store import AssetStore
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
    context = multiprocessing.get_context("fork")
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
