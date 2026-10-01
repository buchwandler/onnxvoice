from __future__ import annotations

import copy
import hashlib
import json

import pytest

import onnxvoice.catalog_tools.piper as piper
from onnxvoice.catalog import CatalogClient
from onnxvoice.catalog_tools.piper import CatalogError, build_catalog, verify_catalog
from onnxvoice.manager import OnnxVoice


def _artifact(path: str, payload: bytes) -> dict[str, object]:
    return {
        "size_bytes": len(payload),
        "md5_digest": hashlib.md5(payload, usedforsecurity=False).hexdigest(),
    }


def _upstream() -> dict[str, object]:
    return {
        "en_US-test-medium": {
            "key": "en_US-test-medium",
            "name": "Test",
            "language": {"code": "en_US", "family": "en"},
            "quality": "medium",
            "num_speakers": 2,
            "speaker_id_map": {"a": 0, "b": 1},
            "aliases": ["test-medium"],
            "files": {
                "en/en_US/test/medium/MODEL_CARD": _artifact("MODEL_CARD", b"card"),
                "en/en_US/test/medium/test.onnx": _artifact("test.onnx", b"model"),
                "en/en_US/test/medium/test.onnx.json": _artifact("test.onnx.json", b"config"),
            },
        }
    }


def test_build_catalog_pins_revision_and_is_deterministic():
    first = build_catalog(_upstream(), resolved_revision="a" * 40)
    second = build_catalog(_upstream(), resolved_revision="a" * 40)

    assert first == second
    verify_catalog(first)
    source = first["source"]
    assert source["revision"] == "a" * 40
    assert source["voice_count"] == 1
    assert all(
        "/resolve/" in artifact["url"]
        for artifact in first["voices"]["en_US-test-medium"]["artifacts"].values()
    )


def test_explicit_gender_round_trips_through_runtime_and_list_voices(tmp_path):
    voice_id = "en_US-test-medium"
    upstream = _upstream()
    upstream[voice_id]["gender"] = "female"
    catalog = build_catalog(upstream, resolved_revision="a" * 40)

    assert catalog["voices"][voice_id]["gender"] == "female"
    catalog_path = tmp_path / "piper.json"
    catalog_path.write_text(json.dumps(catalog), encoding="utf-8")

    client = CatalogClient(cache_dir=tmp_path / "cache", sources={"piper": str(catalog_path)})
    item = client.resolve(f"piper:{voice_id}")
    assert item.metadata["gender"] == "female"

    manager = OnnxVoice(
        cache_dir=tmp_path / "manager-cache",
        catalog_sources={"piper": str(catalog_path)},
        offline=True,
    )
    record = next(
        record for record in manager.list_voices(system="piper") if record.voice_id == voice_id
    )
    assert record.gender == "female"
    assert record.metadata.gender == "female"


def test_missing_gender_stays_unknown_through_catalog_and_runtime(tmp_path):
    voice_id = "en_US-test-medium"
    catalog = build_catalog(_upstream(), resolved_revision="b" * 40)
    assert "gender" not in catalog["voices"][voice_id]

    catalog_path = tmp_path / "piper.json"
    catalog_path.write_text(json.dumps(catalog), encoding="utf-8")
    manager = OnnxVoice(
        cache_dir=tmp_path / "cache",
        catalog_sources={"piper": str(catalog_path)},
        offline=True,
    )
    record = next(
        record for record in manager.list_voices(system="piper") if record.voice_id == voice_id
    )
    assert record.gender == "unknown"
    assert record.metadata.gender == "unknown"


def test_invalid_upstream_and_normalized_gender_are_rejected():
    voice_id = "en_US-test-medium"
    upstream = _upstream()
    upstream[voice_id]["gender"] = "Female"
    with pytest.raises(CatalogError, match="invalid gender"):
        build_catalog(upstream, resolved_revision="c" * 40)

    normalized = build_catalog(_upstream(), resolved_revision="d" * 40)
    normalized["voices"][voice_id]["gender"] = "Female"
    with pytest.raises(CatalogError, match="invalid gender"):
        verify_catalog(normalized)


def test_verify_rejects_unsafe_paths_aliases_and_duplicate_speaker_ids():
    catalog = build_catalog(_upstream(), resolved_revision="a" * 40)

    unsafe = copy.deepcopy(catalog)
    unsafe["voices"]["en_US-test-medium"]["artifacts"]["model"]["path"] = "../escape"
    with pytest.raises(CatalogError, match="unsafe path"):
        verify_catalog(unsafe)

    duplicate = copy.deepcopy(catalog)
    duplicate["voices"]["en_US-test-medium"]["speaker_id_map"] = {"a": 0, "b": 0}
    with pytest.raises(CatalogError, match="duplicate numeric speaker id"):
        verify_catalog(duplicate)

    ambiguous = copy.deepcopy(catalog)
    ambiguous["voices"]["other"] = copy.deepcopy(ambiguous["voices"]["en_US-test-medium"])
    ambiguous["voices"]["other"]["id"] = "other"
    ambiguous["voices"]["other"]["aliases"] = ["test-medium"]
    ambiguous["source"]["voice_count"] = 2
    with pytest.raises(CatalogError, match="ambiguous"):
        verify_catalog(ambiguous)


def test_resolve_revision_requires_exact_sha(monkeypatch):
    monkeypatch.setattr(piper, "_read_url", lambda url: b'{"sha":"main"}')

    with pytest.raises(CatalogError, match="exact 40-character"):
        piper.resolve_revision("example/repo", "main")


def test_catalog_source_output_is_json_compatible(tmp_path):
    catalog = build_catalog(_upstream(), resolved_revision="b" * 40)
    path = tmp_path / "catalog.json"
    path.write_text(json.dumps(catalog), encoding="utf-8")
    assert json.loads(path.read_text(encoding="utf-8"))["kind"] == "piper-voice-catalog"
