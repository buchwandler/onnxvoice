from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from onnxvoice.cli import main
from onnxvoice.manager import OnnxVoice as Manager
from onnxvoice.systems import KittenAdapter
from onnxvoice.types import TensorSpec

_REPOSITORY = "KittenML/kitten-tts-nano-0.8-int8"
_REVISION = "a" * 40
_MODEL_ID = "nano-0.8-int8"
_PAYLOADS = {
    "model.onnx": b"local Kitten ONNX fixture",
    "voices.npz": b"local voice styles fixture",
}


class FakeRuntimeSession:
    input_names = ("input_ids", "style", "speed")
    input_specs = (
        TensorSpec("input_ids", "tensor(int64)", (1, "tokens")),
        TensorSpec("style", "tensor(float)", (1, 4)),
        TensorSpec("speed", "tensor(float)", (1,)),
    )

    def run(self, inputs: dict[str, np.ndarray]) -> list[np.ndarray]:
        assert set(inputs) == {"input_ids", "style", "speed"}
        return [np.arange(6000, dtype=np.float32)[None, :]]


def _catalog() -> dict[str, Any]:
    artifacts = []
    for role, filename, format_name in (
        ("model", "model.onnx", "onnx"),
        ("voices", "voices.npz", "npz"),
    ):
        payload = _PAYLOADS[filename]
        artifacts.append(
            {
                "role": role,
                "format": format_name,
                "filename": filename,
                "url": f"https://huggingface.co/{_REPOSITORY}/resolve/{_REVISION}/{filename}",
                "size": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        )
    return {
        "schema": 1,
        "kind": "kitten-onnx-model-catalog",
        "models": {
            _MODEL_ID: {
                "id": _MODEL_ID,
                "aliases": ["nano-int8"],
                "name": "Kitten TTS Nano",
                "version": "0.8",
                "language": "en-US",
                "sample_rate": 24000,
                "quality": "int8",
                "runtime": {"profile": "ONNX2"},
                "upstream": {
                    "provider": "huggingface",
                    "repository": _REPOSITORY,
                    "revision": _REVISION,
                    "license": "apache-2.0",
                },
                "artifacts": artifacts,
                "voice_aliases": {
                    "Bella": "expr-voice-2-f",
                    "Jasper": "expr-voice-2-m",
                },
                "speed_priors": {"expr-voice-2-f": 0.8},
                "metadata": {"parameters_millions": 15},
            }
        },
    }


def _write_catalog(path: Path) -> None:
    path.write_text(json.dumps(_catalog()), encoding="utf-8")


def test_managed_install_and_open_preserve_kitten_roles_and_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    catalog_path = tmp_path / "catalog.json"
    _write_catalog(catalog_path)
    downloads: list[str] = []

    def download(source, *, local_dir: Path, offline: bool) -> Path:
        assert not offline
        downloads.append(source.path)
        result = local_dir / Path(source.path).name
        result.write_bytes(_PAYLOADS[source.path])
        return result

    monkeypatch.setattr("onnxvoice.store.download_huggingface_file", download)
    monkeypatch.setattr(
        "onnxvoice.systems.kitten.OnnxSession", lambda *_args, **_kwargs: FakeRuntimeSession()
    )
    monkeypatch.setattr(
        np, "load", lambda *_args, **_kwargs: pytest.fail("OnnxVoice must not open voices.npz")
    )
    manager = Manager(
        cache_dir=tmp_path / "cache",
        catalog_sources={"kitten": str(catalog_path)},
    )

    installation = manager.install("kitten:nano-int8")
    assert downloads == ["model.onnx", "voices.npz"]
    assert installation.system == "kitten"
    assert installation.sample_rate == 24000
    assert installation.voices == ("Bella", "Jasper")
    assert {artifact.role for artifact in installation.artifacts} == {"model", "voices"}
    assert installation.require_artifact("model").path.read_bytes() == _PAYLOADS["model.onnx"]
    assert installation.require_artifact("voices").path.read_bytes() == _PAYLOADS["voices.npz"]
    assert installation.metadata["voice_aliases"] == {
        "Bella": "expr-voice-2-f",
        "Jasper": "expr-voice-2-m",
    }
    assert installation.metadata["speed_priors"] == {"expr-voice-2-f": 0.8}
    assert installation.metadata["runtime"] == {"profile": "ONNX2"}
    assert installation.metadata["source_revision"] == _REVISION
    assert installation.metadata["source_repository"] == _REPOSITORY

    runtime = manager.open(installation)
    try:
        assert isinstance(runtime, KittenAdapter)
        result = runtime.infer([0, 31, 10, 0], style=np.ones((1, 4), dtype=np.float32))
        assert result.sample_rate == 24000
        assert result.audio.shape == (1000,)
    finally:
        runtime.close()


def test_manager_and_cli_list_filter_show_public_kitten_voice_aliases(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    catalog_path = tmp_path / "catalog.json"
    _write_catalog(catalog_path)
    cache = tmp_path / "cache"
    manager = Manager(
        cache_dir=cache,
        catalog_sources={"kitten": str(catalog_path)},
    )

    records = manager.list_voices(system="kitten")
    assert [record.ref for record in records] == [
        "kitten:nano-0.8-int8/Bella",
        "kitten:nano-0.8-int8/Jasper",
    ]
    assert records[0].voice_id == "Bella"
    assert records[0].backing_ref == "kitten:nano-0.8-int8"
    assert records[0].locale == "en-US"
    assert records[0].languages == ("en-US",)
    assert records[0].gender == "unknown"
    assert manager.list_voices(system="kitten", language="en-US") == records
    assert manager.list_voices(system="kitten", language="fr") == []
    resolved = manager.resolve_voice("kitten:nano-int8/Bella")
    assert resolved.ref == "kitten:nano-0.8-int8/Bella"
    assert resolved.voice_id == "Bella"

    monkeypatch.setenv("ONNXVOICE_KITTEN_CATALOG", str(catalog_path))
    args = ["--cache-dir", str(cache), "voices"]
    assert main([*args, "list", "--system", "kitten"]) == 0
    table = capsys.readouterr().out
    assert "kitten:nano-0.8-int8/Bella" in table
    assert "Bella" in table
    assert "expr-voice-2-f" not in table

    assert main([*args, "list", "--system", "kitten", "--lang", "en-US", "--format", "json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert [item["voice_id"] for item in payload["items"]] == ["Bella", "Jasper"]
    assert all(item["gender"] == "unknown" for item in payload["items"])
    assert all(item["languages"] == ["en-US"] for item in payload["items"])

    assert main([*args, "show", "kitten:nano-int8/Bella", "--format", "json"]) == 0
    shown = json.loads(capsys.readouterr().out)
    assert shown["ref"] == "kitten:nano-0.8-int8/Bella"
    assert shown["voice_id"] == "Bella"
    assert shown["gender"] == "unknown"
