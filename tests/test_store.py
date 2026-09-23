from __future__ import annotations

import hashlib
import json
import os

import pytest

from onnxvoice.errors import IntegrityError, OfflineError
from onnxvoice.store import AssetStore
from onnxvoice.types import Artifact, CatalogItem


def test_canonical_pocket_install_requires_integrity(tmp_path):
    source = tmp_path / "bundle.json"
    source.write_bytes(b"bundle")
    item = CatalogItem(
        system="pocket",
        id="bundle",
        kind="bundle",
        artifacts=(Artifact("bundle_metadata", source.name, source.as_uri()),),
        metadata={"canonical_catalog": True},
    )
    store = AssetStore(tmp_path / "cache")
    with pytest.raises(IntegrityError, match="no positive size"):
        store.install(item)


def test_store_deduplicates_content(tmp_path):
    payload = b"same-model-bytes"
    source_a = tmp_path / "a.onnx"
    source_b = tmp_path / "b.onnx"
    source_a.write_bytes(payload)
    source_b.write_bytes(payload)
    sha = hashlib.sha256(payload).hexdigest()

    store = AssetStore(tmp_path / "cache")
    first = CatalogItem(
        system="test",
        id="one",
        kind="model",
        artifacts=(Artifact("model", "a.onnx", source_a.as_uri(), len(payload), sha),),
    )
    second = CatalogItem(
        system="test",
        id="two",
        kind="model",
        artifacts=(Artifact("model", "b.onnx", source_b.as_uri(), len(payload), sha),),
    )

    store.install(first)
    store.install(second)

    blobs = [p for p in store.blobs.glob("*/*") if p.is_file()]
    assert len(blobs) == 1
    assert store.get("test", "one").artifact("model").path.read_bytes() == payload
    assert store.get("test", "two").artifact("model").path.read_bytes() == payload


def test_remove_then_gc(tmp_path):
    payload = b"model"
    source = tmp_path / "voice.onnx"
    source.write_bytes(payload)
    sha = hashlib.sha256(payload).hexdigest()
    item = CatalogItem(
        system="test",
        id="one",
        kind="model",
        artifacts=(Artifact("model", source.name, source.as_uri(), len(payload), sha),),
    )
    store = AssetStore(tmp_path / "cache")
    store.install(item)
    store.remove("test", "one")
    report = store.gc()
    assert report.removed_blobs == 1


def test_pocket_state_gc_respects_references_from_all_installed_variants(tmp_path):
    state_payload = b"pinned voice state"
    state_sha = hashlib.sha256(state_payload).hexdigest()
    source_record = {
        "name": "alba",
        "source": {
            "provider": "huggingface",
            "repository": "kyutai/pocket-tts",
            "revision": "d" * 40,
            "path": "languages/english_2026-04/embeddings/alba.safetensors",
        },
        "size": len(state_payload),
        "sha256": state_sha,
    }
    bundle_payload = b"bundle metadata"
    bundle_file = tmp_path / "bundle.json"
    bundle_file.write_bytes(bundle_payload)
    bundle_artifact = Artifact(
        "bundle_metadata",
        "bundle.json",
        bundle_file.as_uri(),
        len(bundle_payload),
        hashlib.sha256(bundle_payload).hexdigest(),
    )
    store = AssetStore(tmp_path / "cache")

    def item(cache_id: str) -> CatalogItem:
        return CatalogItem(
            system="pocket",
            id="english_2026-04",
            kind="bundle",
            artifacts=(bundle_artifact,),
            metadata={
                "cache_id": cache_id,
                "canonical_catalog": True,
                "predefined_voice_names": ["alba"],
                "voice_states": [source_record],
            },
        )

    store.install(item("english-profile-a"))
    store.install(item("english-profile-b"))
    key = {
        "system": "pocket",
        "asset_kind": "predefined_voice_state",
        "model_repo": "kyutai/pocket-tts",
        "model_revision": "d" * 40,
        "asset_path": "languages/english_2026-04/embeddings/alba.safetensors",
        "bundle_id": "english_2026-04",
        "voice_name": "alba",
        "expected_size": len(state_payload),
        "expected_sha256": state_sha,
    }
    digest = hashlib.sha256(
        json.dumps(key, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    cache_dir = store.root / "pocket-voice-states" / digest[:2] / digest
    cache_dir.mkdir(parents=True)
    state_path = cache_dir / "alba.safetensors"
    state_path.write_bytes(state_payload)
    (cache_dir / "alba.json").write_text(
        json.dumps({"key": key, "size": len(state_payload), "sha256": state_sha}),
        encoding="utf-8",
    )

    assert store.gc().removed_auxiliary_count == 0
    store.remove("pocket", "english-profile-a")
    assert store.gc().removed_auxiliary_count == 0
    store.remove("pocket", "english-profile-b")
    report = store.gc()
    assert report.removed_auxiliary_count == 1
    assert report.removed_bytes == len(bundle_payload)
    assert report.removed_auxiliary_bytes >= len(state_payload)
    assert not state_path.exists()


def test_install_falls_back_to_copy_when_os_link_is_unavailable(tmp_path, monkeypatch):
    payload = b"model"
    source = tmp_path / "voice.onnx"
    source.write_bytes(payload)
    sha = hashlib.sha256(payload).hexdigest()
    item = CatalogItem(
        system="test",
        id="copy-fallback",
        kind="model",
        artifacts=(Artifact("model", source.name, source.as_uri(), len(payload), sha),),
    )
    store = AssetStore(tmp_path / "cache")

    monkeypatch.delattr(os, "link", raising=False)

    installation = store.install(item)

    assert installation.artifact("model").path.read_bytes() == payload
    store.verify(installation)


@pytest.mark.parametrize(
    "error",
    [
        OSError("hard links unavailable"),
        NotImplementedError("hard links unavailable"),
    ],
)
def test_install_falls_back_when_hard_link_operation_is_unsupported(
    tmp_path,
    monkeypatch,
    error,
):
    payload = b"model"
    source = tmp_path / "voice.onnx"
    source.write_bytes(payload)
    sha = hashlib.sha256(payload).hexdigest()
    item = CatalogItem(
        system="test",
        id="copy-fallback-error",
        kind="model",
        artifacts=(Artifact("model", source.name, source.as_uri(), len(payload), sha),),
    )
    store = AssetStore(tmp_path / "cache")

    def fail_link(*_args, **_kwargs):
        raise error

    monkeypatch.setattr(os, "link", fail_link, raising=False)

    installation = store.install(item)

    assert installation.artifact("model").path.read_bytes() == payload
    store.verify(installation)


def test_new_install_persists_timing_metadata(tmp_path):
    payload = b"model-with-timing"
    source = tmp_path / "timed.onnx"
    source.write_bytes(payload)
    item = CatalogItem(
        system="kokoro",
        id="v1.0",
        kind="model",
        artifacts=(
            Artifact(
                "model",
                source.name,
                source.as_uri(),
                len(payload),
                hashlib.sha256(payload).hexdigest(),
            ),
        ),
        metadata={"runtime": {"timings_output": "durations"}},
    )
    store = AssetStore(tmp_path / "cache")

    store.install(item)
    reloaded = store.get("kokoro", "v1.0")

    assert reloaded.timing_output == "durations"


def _huggingface_artifact(payload: bytes, *, filename: str = "model.onnx") -> Artifact:
    return Artifact(
        role="flow_lm_main",
        filename=filename,
        url="https://huggingface.co/owner/repo/resolve/revision/model.onnx",
        size=len(payload),
        sha256=hashlib.sha256(payload).hexdigest(),
        metadata={
            "source": {
                "provider": "huggingface",
                "repository": "owner/repo",
                "revision": "a" * 40,
                "path": "onnx/bundle/model.onnx",
                "gated": False,
            }
        },
    )


def test_pocket_huggingface_artifact_uses_structured_source_and_integrity(
    tmp_path, monkeypatch
) -> None:
    import onnxvoice.store as store_module

    payload = b"expected model bytes"
    calls = []
    store = AssetStore(tmp_path / "cache")
    item = CatalogItem(
        system="pocket",
        id="bundle",
        kind="bundle",
        artifacts=(_huggingface_artifact(payload),),
        metadata={"canonical_catalog": True},
    )

    def download(source, *, local_dir, offline):
        calls.append((source, local_dir, offline))
        path = local_dir / source.path
        path.parent.mkdir(parents=True)
        path.write_bytes(payload)
        return path

    monkeypatch.setattr(store_module, "download_huggingface_file", download)
    monkeypatch.setattr(store, "_download", lambda *_args, **_kwargs: pytest.fail("used urllib"))
    events = []
    installed = store.install(item, progress=events.append)

    assert installed.artifact("flow_lm_main").path.read_bytes() == payload
    source, local_dir, offline = calls[0]
    assert source.repository == "owner/repo"
    assert source.revision == "a" * 40
    assert source.path == "onnx/bundle/model.onnx"
    assert local_dir.name.startswith("onnxvoice-download-")
    assert not local_dir.exists()
    assert offline is False
    assert [event.phase for event in events if event.phase.startswith("download_")] == [
        "download_started",
        "download_completed",
    ]


def test_pocket_huggingface_download_still_checks_checksum(tmp_path, monkeypatch) -> None:
    import onnxvoice.store as store_module

    store = AssetStore(tmp_path / "cache")
    payload = b"expected model bytes"
    item = CatalogItem(
        system="pocket",
        id="bad-digest",
        kind="bundle",
        artifacts=(_huggingface_artifact(payload[:-1] + b"!"),),
        metadata={"canonical_catalog": True},
    )

    def download(source, *, local_dir, offline):
        path = local_dir / source.path
        path.parent.mkdir(parents=True)
        path.write_bytes(payload)
        return path

    monkeypatch.setattr(store_module, "download_huggingface_file", download)
    with pytest.raises(IntegrityError):
        store.install(item)


def test_offline_pocket_cache_miss_does_not_invoke_huggingface(tmp_path, monkeypatch) -> None:
    import onnxvoice.store as store_module

    store = AssetStore(tmp_path / "cache", offline=True)
    payload = b"expected model bytes"
    item = CatalogItem(
        system="pocket",
        id="offline-miss",
        kind="bundle",
        artifacts=(_huggingface_artifact(payload),),
        metadata={"canonical_catalog": True},
    )
    monkeypatch.setattr(
        store_module,
        "download_huggingface_file",
        lambda *_args, **_kwargs: pytest.fail("offline cache miss attempted a Hub download"),
    )

    with pytest.raises(OfflineError):
        store.install(item)
