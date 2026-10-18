from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest
from _kitten2_fixture import create_bundle

from onnxvoice.errors import RuntimeContractError
from onnxvoice.systems.kitten2_contract import load_kitten2_bundle


def _manifest(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_manifest(path: Path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(data, sort_keys=True), encoding="utf-8")


def test_validates_complete_nested_bundle_inventory_and_hashes(tmp_path: Path) -> None:
    bundle_path = create_bundle(tmp_path / "bundle")

    bundle = load_kitten2_bundle(bundle_path)

    assert bundle.root == bundle_path.parent
    assert bundle.manifest["default_voice"] == "Bruno"
    assert bundle.files["decoder/mel2wav/mel2wav.onnx.data"]
    assert bundle.files["lm/lm.ternary-exact-int8.onnx.data"]


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("format", "kitten-v0.8", "format"),
        ("sample_rate", 16000, "sample_rate"),
        ("default_voice", "Unknown", "default_voice"),
        ("default_preset", "unknown", "default_preset"),
    ],
)
def test_rejects_invalid_bundle_defaults(
    tmp_path: Path, field: str, value: Any, message: str
) -> None:
    bundle_path = create_bundle(tmp_path / "bundle")
    data = _manifest(bundle_path)
    data[field] = value
    _write_manifest(bundle_path, data)

    with pytest.raises(RuntimeContractError, match=message):
        load_kitten2_bundle(bundle_path)


def test_rejects_unsafe_file_path(tmp_path: Path) -> None:
    bundle_path = create_bundle(tmp_path / "bundle")
    data = _manifest(bundle_path)
    data["files"]["../escape"] = "0" * 64
    _write_manifest(bundle_path, data)

    with pytest.raises(RuntimeContractError, match="Unsafe bundle file path"):
        load_kitten2_bundle(bundle_path)


def test_rejects_missing_corrupt_and_undeclared_files(tmp_path: Path) -> None:
    bundle_path = create_bundle(tmp_path / "missing")
    (bundle_path.parent / "lm/lm.onnx").unlink()
    with pytest.raises(RuntimeContractError, match="inventory mismatch"):
        load_kitten2_bundle(bundle_path)

    bundle_path = create_bundle(tmp_path / "corrupt")
    (bundle_path.parent / "speaker/speaker.onnx").write_bytes(b"corrupt")
    with pytest.raises(RuntimeContractError, match="SHA-256 mismatch"):
        load_kitten2_bundle(bundle_path)

    bundle_path = create_bundle(tmp_path / "extra")
    (bundle_path.parent / "unlisted.bin").write_bytes(b"extra")
    with pytest.raises(RuntimeContractError, match="undeclared"):
        load_kitten2_bundle(bundle_path)


def test_rejects_invalid_token_map_presets_and_metadata(tmp_path: Path) -> None:
    bundle_path = create_bundle(tmp_path / "token-map")
    data = _manifest(bundle_path)
    data["token_map"]["speech_start_id"] = data["token_map"]["text_start_id"]
    _write_manifest(bundle_path, data)
    with pytest.raises(RuntimeContractError, match="token_map IDs must be distinct"):
        load_kitten2_bundle(bundle_path)

    bundle_path = create_bundle(tmp_path / "preset")
    data = _manifest(bundle_path)
    data["decode_presets"]["stable"]["temperature"] = float("nan")
    _write_manifest(bundle_path, data)
    with pytest.raises(RuntimeContractError, match="temperature"):
        load_kitten2_bundle(bundle_path)

    bundle_path = create_bundle(tmp_path / "metadata")
    metadata_path = bundle_path.parent / "speaker/speaker.meta.json"
    metadata_path.write_text("[]", encoding="utf-8")
    data = _manifest(bundle_path)
    data["files"]["speaker/speaker.meta.json"] = hashlib.sha256(
        metadata_path.read_bytes()
    ).hexdigest()
    _write_manifest(bundle_path, data)
    with pytest.raises(RuntimeContractError, match="must contain a JSON object"):
        load_kitten2_bundle(bundle_path)


def test_rejects_external_data_without_adjacent_graph(tmp_path: Path) -> None:
    bundle_path = create_bundle(tmp_path / "bundle")
    bundle = bundle_path.parent
    source = bundle / "lm/lm.ternary-exact-int8.onnx.data"
    target = bundle / "orphan.onnx.data"
    source.rename(target)
    data = _manifest(bundle_path)
    data["files"].pop("lm/lm.ternary-exact-int8.onnx.data")
    data["files"]["orphan.onnx.data"] = hashlib.sha256(target.read_bytes()).hexdigest()
    _write_manifest(bundle_path, data)

    with pytest.raises(RuntimeContractError, match="external ONNX data must be adjacent"):
        load_kitten2_bundle(bundle_path)
