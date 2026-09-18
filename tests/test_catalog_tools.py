from __future__ import annotations

import copy
import hashlib
import json

import pytest

import onnxvoice.catalog_tools.piper as piper
from onnxvoice.catalog_tools.piper import CatalogError, build_catalog, verify_catalog


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
