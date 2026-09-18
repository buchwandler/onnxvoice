from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from onnxvoice.store import AssetStore
from onnxvoice.types import Artifact, AssetProgress, CatalogItem


def _make_item(
    system: str = "test",
    item_id: str = "one",
    kind: str = "model",
    filename: str = "model.onnx",
    payload: bytes = b"model-bytes",
    role: str = "model",
) -> tuple[CatalogItem, Path]:
    """Create a CatalogItem and write the payload to a temporary source file."""
    source = Path(f"/tmp/{filename}")
    source.write_bytes(payload)
    sha = hashlib.sha256(payload).hexdigest()
    item = CatalogItem(
        system=system,
        id=item_id,
        kind=kind,
        artifacts=(
            Artifact(
                role=role,
                filename=filename,
                url=source.as_uri(),
                size=len(payload),
                sha256=sha,
            ),
        ),
    )
    return item, source


def test_fresh_install_events(tmp_path):
    """Test that a fresh install emits the correct event sequence."""
    item, source = _make_item()
    store = AssetStore(tmp_path / "cache")
    events: list[AssetProgress] = []

    def capture(event: AssetProgress) -> None:
        events.append(event)

    store.install(item, progress=capture)

    # Verify event sequence
    # Verify event sequence (download_progress may occur for file:// URLs)
    phases = [e.phase for e in events]
    assert phases[0] == "install_started"
    assert phases[1] == "download_started"
    # May have download_progress events
    # Find the index of verify_started
    verify_start_idx = phases.index("verify_started")
    assert phases[verify_start_idx] == "verify_started"
    assert phases[verify_start_idx + 1] == "verify_completed"
    assert phases[verify_start_idx + 2] == "download_completed"
    assert phases[verify_start_idx + 3] == "artifact_installed"
    assert phases[-1] == "install_completed"

    # Verify install_started
    install_started = next(e for e in events if e.phase == "install_started")
    assert install_started.ref == item.ref
    assert install_started.target is not None
    assert install_started.target.endswith("/test/one")

    # Verify download_started
    download_started = next(e for e in events if e.phase == "download_started")
    assert download_started.ref == item.ref
    assert download_started.artifact == "model.onnx"
    assert download_started.role == "model"
    assert download_started.total == len(b"model-bytes")
    assert download_started.target is not None
    assert download_started.target.endswith("/test/one/model.onnx")

    # Verify verify_started
    verify_started = next(e for e in events if e.phase == "verify_started")
    assert verify_started.ref == item.ref
    assert verify_started.artifact == "model.onnx"
    assert verify_started.role == "model"
    assert verify_started.target is not None

    # Verify verify_completed
    verify_completed = next(e for e in events if e.phase == "verify_completed")
    assert verify_completed.ref == item.ref
    assert verify_completed.artifact == "model.onnx"
    assert verify_completed.role == "model"
    assert verify_completed.target is not None

    # Verify download_completed
    download_completed = next(e for e in events if e.phase == "download_completed")
    assert download_completed.ref == item.ref
    assert download_completed.artifact == "model.onnx"
    assert download_completed.role == "model"
    assert download_completed.completed == len(b"model-bytes")
    assert download_completed.total == len(b"model-bytes")
    assert download_completed.target is not None

    # Verify artifact_installed
    artifact_installed = next(e for e in events if e.phase == "artifact_installed")
    assert artifact_installed.ref == item.ref
    assert artifact_installed.artifact == "model.onnx"
    assert artifact_installed.role == "model"
    assert artifact_installed.target is not None
    assert artifact_installed.target.endswith("/test/one/model.onnx")

    # Verify install_completed
    install_completed = next(e for e in events if e.phase == "install_completed")
    assert install_completed.ref == item.ref
    assert install_completed.target is not None
    assert install_completed.target.endswith("/test/one")

    # Verify the installation exists
    assert Path(install_completed.target).exists()


def test_already_installed_no_download_events(tmp_path):
    """Test that already-installed runs emit install_started/install_completed without download events."""
    item, source = _make_item()
    store = AssetStore(tmp_path / "cache")

    # First install
    store.install(item)

    # Second install (already installed)
    events: list[AssetProgress] = []

    def capture(event: AssetProgress) -> None:
        events.append(event)

    store.install(item, progress=capture)

    # Verify no download events
    phases = [e.phase for e in events]
    assert "download_started" not in phases
    assert "download_progress" not in phases
    assert "download_completed" not in phases
    assert "verify_started" not in phases
    assert "verify_completed" not in phases
    assert "artifact_cached" not in phases
    assert "artifact_installed" not in phases

    # Verify install_started and install_completed are present
    assert "install_started" in phases
    assert "install_completed" in phases

    # Verify install_completed has "already installed" message
    install_completed = events[-1]
    assert install_completed.phase == "install_completed"
    assert install_completed.message == "already installed"
    assert install_completed.ref == item.ref
    assert install_completed.target is not None
    assert install_completed.target.endswith("/test/one")


def test_blob_cache_hit(tmp_path):
    """Test that blob cache hits emit artifact_cached with role/target metadata."""
    payload = b"model-bytes"
    sha = hashlib.sha256(payload).hexdigest()

    # Create two items with the same content
    source1 = tmp_path / "model1.onnx"
    source1.write_bytes(payload)
    item1 = CatalogItem(
        system="test",
        id="one",
        kind="model",
        artifacts=(
            Artifact(
                role="model",
                filename="model1.onnx",
                url=source1.as_uri(),
                size=len(payload),
                sha256=sha,
            ),
        ),
    )

    source2 = tmp_path / "model2.onnx"
    source2.write_bytes(payload)
    item2 = CatalogItem(
        system="test",
        id="two",
        kind="model",
        artifacts=(
            Artifact(
                role="model",
                filename="model2.onnx",
                url=source2.as_uri(),
                size=len(payload),
                sha256=sha,
            ),
        ),
    )

    store = AssetStore(tmp_path / "cache")

    # First install - downloads and caches
    store.install(item1)

    # Second install - should use blob cache
    events: list[AssetProgress] = []

    def capture(event: AssetProgress) -> None:
        events.append(event)

    store.install(item2, progress=capture)

    # Verify artifact_cached event exists
    cached_events = [e for e in events if e.phase == "artifact_cached"]
    assert len(cached_events) == 1

    cached = cached_events[0]
    assert cached.ref == item2.ref
    assert cached.artifact == "model2.onnx"
    assert cached.role == "model"
    assert cached.target is not None
    assert cached.target.endswith("/test/two/model2.onnx")
    assert cached.total == len(payload)

    # Verify no download events for the cached artifact
    phases = [e.phase for e in events]
    assert "download_started" not in phases
    assert "download_progress" not in phases
    assert "download_completed" not in phases


def test_verification_failure(tmp_path):
    """Test that verification failure emits install_failed and no install_completed."""
    # Create a file with invalid content
    source = tmp_path / "bad.onnx"
    source.write_bytes(b"bad-content")

    # Create item with wrong sha256
    item = CatalogItem(
        system="test",
        id="one",
        kind="model",
        artifacts=(
            Artifact(
                role="model",
                filename="bad.onnx",
                url=source.as_uri(),
                size=len(b"bad-content"),
                sha256="0" * 64,  # Wrong sha256
            ),
        ),
    )

    store = AssetStore(tmp_path / "cache")
    events: list[AssetProgress] = []

    def capture(event: AssetProgress) -> None:
        events.append(event)

    with pytest.raises(Exception):  # noqa: B017
        store.install(item, progress=capture)

    # Verify install_failed is emitted
    failed_events = [e for e in events if e.phase == "install_failed"]
    assert len(failed_events) == 1

    failed = failed_events[0]
    assert failed.ref == item.ref
    assert failed.target is not None

    # Verify no install_completed
    completed_events = [e for e in events if e.phase == "install_completed"]
    assert len(completed_events) == 0


def test_unknown_total(tmp_path):
    """Test that unknown sizes use total=None, not total=0."""
    payload = b"model-bytes"
    source = tmp_path / "model.onnx"
    source.write_bytes(payload)
    sha = hashlib.sha256(payload).hexdigest()

    # Create item with size=None (unknown)
    item = CatalogItem(
        system="test",
        id="one",
        kind="model",
        artifacts=(
            Artifact(
                role="model",
                filename="model.onnx",
                url=source.as_uri(),
                size=None,  # Unknown size
                sha256=sha,
            ),
        ),
    )

    store = AssetStore(tmp_path / "cache")
    events: list[AssetProgress] = []

    def capture(event: AssetProgress) -> None:
        events.append(event)

    store.install(item, progress=capture)

    # Find download_started event
    download_started = next(e for e in events if e.phase == "download_started")
    assert download_started.total is None  # Should be None, not 0

    # Find download_completed event
    download_completed = next(e for e in events if e.phase == "download_completed")
    assert download_completed.total is None  # Should be None, not 0


def test_multiple_artifacts(tmp_path):
    """Test that events are emitted correctly for multiple artifacts."""
    payload1 = b"model-bytes"
    payload2 = b"config-bytes"
    source1 = tmp_path / "model.onnx"
    source1.write_bytes(payload1)
    source2 = tmp_path / "config.json"
    source2.write_bytes(payload2)
    sha1 = hashlib.sha256(payload1).hexdigest()
    sha2 = hashlib.sha256(payload2).hexdigest()

    item = CatalogItem(
        system="test",
        id="one",
        kind="model",
        artifacts=(
            Artifact(
                role="model",
                filename="model.onnx",
                url=source1.as_uri(),
                size=len(payload1),
                sha256=sha1,
            ),
            Artifact(
                role="config",
                filename="config.json",
                url=source2.as_uri(),
                size=len(payload2),
                sha256=sha2,
            ),
        ),
    )

    store = AssetStore(tmp_path / "cache")
    events: list[AssetProgress] = []

    def capture(event: AssetProgress) -> None:
        events.append(event)

    store.install(item, progress=capture)

    # Verify we have events for both artifacts
    artifact_events = [e for e in events if e.artifact is not None]
    model_events = [e for e in artifact_events if e.artifact == "model.onnx"]
    config_events = [e for e in artifact_events if e.artifact == "config.json"]

    assert len(model_events) > 0
    assert len(config_events) > 0

    # Verify roles are correct
    for e in model_events:
        assert e.role == "model"
    for e in config_events:
        assert e.role == "config"
