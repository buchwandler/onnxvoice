from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from io import BytesIO
from typing import Any
from urllib.error import URLError

import pytest

from onnxvoice.catalog_tools import kitten as kitten_tools
from onnxvoice.cli import main

_REPOSITORY = "KittenML/test-model"
_SEED_REVISION = "a" * 40
_REVISION = "d" * 40
_MODEL_ID = "nano-0.8-int8"


def _url(filename: str, revision: str = _SEED_REVISION) -> str:
    return f"https://huggingface.co/{_REPOSITORY}/resolve/{revision}/{filename}"


def _seed_catalog() -> dict[str, Any]:
    return {
        "schema": 1,
        "kind": "kitten-onnx-model-catalog",
        "models": {
            _MODEL_ID: {
                "id": _MODEL_ID,
                "aliases": ["nano-int8"],
                "name": "Old name",
                "version": "old",
                "language": "en-US",
                "sample_rate": 24000,
                "quality": "int8",
                "runtime": {"profile": "ONNX2"},
                "upstream": {
                    "provider": "huggingface",
                    "repository": _REPOSITORY,
                    "revision": _SEED_REVISION,
                    "license": "apache-2.0",
                },
                "artifacts": [
                    {
                        "role": "model",
                        "format": "onnx",
                        "filename": "old.onnx",
                        "url": _url("old.onnx"),
                        "size": 10,
                        "sha256": "b" * 64,
                    },
                    {
                        "role": "voices",
                        "format": "npz",
                        "filename": "voices.npz",
                        "url": _url("voices.npz"),
                        "size": 10,
                        "sha256": "c" * 64,
                    },
                ],
                "voice_aliases": {"Bella": "expr-voice-2-f", "Jasper": "expr-voice-2-m"},
                "speed_priors": {"expr-voice-2-f": 0.5},
                "metadata": {
                    "parameters_millions": 15,
                    "upstream_model": "kitten-tts-nano-0.8",
                },
            }
        },
    }


def _upstream_config() -> bytes:
    return json.dumps(
        {
            "name": "Kitten TTS Nano",
            "version": "0.8",
            "type": "ONNX2",
            "model": "kitten-tts-nano-0.8",
            "voices": "voices.npz",
            "model_file": "nano.onnx",
            "speed_priors": {"expr-voice-2-f": 0.8},
            "voice_aliases": {
                "Bella": "expr-voice-2-f",
                "Jasper": "expr-voice-2-m",
            },
        }
    ).encode()


class _Response:
    def __init__(self, payload: bytes, headers: Mapping[str, str] | None = None) -> None:
        self._stream = BytesIO(payload)
        self.headers = dict(headers or {})

    def read(self, size: int = -1) -> bytes:
        return self._stream.read(size)

    def __enter__(self) -> _Response:
        return self

    def __exit__(self, *args: object) -> None:
        return None


def _mock_huggingface(monkeypatch: pytest.MonkeyPatch, *, has_lfs: bool = True) -> list[str]:
    config = _upstream_config()
    model = b"fake onnx payload"
    voices = b"fake voices archive"
    files = {
        "nano.onnx": model,
        "voices.npz": voices,
    }
    tree: list[dict[str, Any]] = [
        {"path": "config.json", "size": len(config)},
    ]
    for filename, payload in files.items():
        entry: dict[str, Any] = {"path": filename, "size": len(payload)}
        if has_lfs:
            entry["lfs"] = {"oid": hashlib.sha256(payload).hexdigest()}
        tree.append(entry)

    requests: list[str] = []

    def fake_urlopen(request, timeout=60):
        url = request.full_url
        requests.append(url)
        if url.endswith("/revision/main"):
            return _Response(json.dumps({"sha": _REVISION}).encode())
        if url.endswith(f"/tree/{_REVISION}?recursive=true&expand=true"):
            return _Response(json.dumps(tree).encode())
        if url.endswith(f"/resolve/{_REVISION}/config.json"):
            return _Response(config)
        for filename, payload in files.items():
            if url.endswith(f"/resolve/{_REVISION}/{filename}"):
                return _Response(payload)
        raise AssertionError(f"Unexpected URL: {url}")

    monkeypatch.setattr(kitten_tools.urllib.request, "urlopen", fake_urlopen)
    return requests


def test_cli_catalog_build_uses_upstream_config_and_lfs_metadata(
    tmp_path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    requests = _mock_huggingface(monkeypatch)
    seed_path = tmp_path / "seed.json"
    catalog_path = tmp_path / "models.json"
    source_path = tmp_path / "source.json"
    seed_path.write_text(json.dumps(_seed_catalog()), encoding="utf-8")

    assert (
        main(
            [
                "catalog",
                "kitten",
                "build",
                "--seed-catalog",
                str(seed_path),
                "--output",
                str(catalog_path),
                "--source-output",
                str(source_path),
                "--revision",
                "main",
            ]
        )
        == 0
    )

    built = json.loads(catalog_path.read_text(encoding="utf-8"))
    model = built["models"][_MODEL_ID]
    source = json.loads(source_path.read_text(encoding="utf-8"))
    assert model["name"] == "Kitten TTS Nano"
    assert model["version"] == "0.8"
    assert model["runtime"] == {"profile": "ONNX2"}
    assert model["voice_aliases"] == {
        "Bella": "expr-voice-2-f",
        "Jasper": "expr-voice-2-m",
    }
    assert model["speed_priors"] == {"expr-voice-2-f": 0.8}
    assert model["metadata"]["upstream_model"] == "kitten-tts-nano-0.8"
    assert model["upstream"]["revision"] == _REVISION
    assert model["artifacts"][0]["sha256"] == hashlib.sha256(b"fake onnx payload").hexdigest()
    assert source["repositories"] == [
        {
            "repository": _REPOSITORY,
            "revision": _REVISION,
            "license": "apache-2.0",
        }
    ]
    assert len(requests) == 3
    assert "Wrote 1 Kitten models" in capsys.readouterr().out


def test_build_catalog_is_deterministic(monkeypatch: pytest.MonkeyPatch) -> None:
    requests = _mock_huggingface(monkeypatch)

    first = kitten_tools.build_catalog(_seed_catalog())
    second = kitten_tools.build_catalog(_seed_catalog())

    assert first == second
    assert kitten_tools.build_source(first) == kitten_tools.build_source(second)
    assert len(requests) == 6


def test_build_downloads_and_hashes_artifacts_without_lfs_oids(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requests = _mock_huggingface(monkeypatch, has_lfs=False)

    catalog = kitten_tools.build_catalog(_seed_catalog())
    model = catalog["models"][_MODEL_ID]

    assert model["artifacts"][0]["size"] == len(b"fake onnx payload")
    assert model["artifacts"][0]["sha256"] == hashlib.sha256(b"fake onnx payload").hexdigest()
    assert model["artifacts"][1]["sha256"] == hashlib.sha256(b"fake voices archive").hexdigest()
    assert len(requests) == 5


def test_cli_catalog_verify_is_offline_and_checks_source(tmp_path, monkeypatch, capsys) -> None:
    catalog = _seed_catalog()
    source = kitten_tools.build_source(catalog)
    source["snapshot_date"] = "2026-10-01"
    source["note"] = "A previous deterministic snapshot."
    catalog_path = tmp_path / "models.json"
    source_path = tmp_path / "source.json"
    catalog_path.write_text(json.dumps(catalog), encoding="utf-8")
    source_path.write_text(json.dumps(source), encoding="utf-8")

    def fail_if_network(*args, **kwargs):
        raise AssertionError("verify must not access the network")

    monkeypatch.setattr(kitten_tools.urllib.request, "urlopen", fail_if_network)
    assert (
        main(
            [
                "catalog",
                "kitten",
                "verify",
                "--catalog",
                str(catalog_path),
                "--source",
                str(source_path),
            ]
        )
        == 0
    )
    assert "Verified 1 Kitten models" in capsys.readouterr().out

    source["repositories"][0]["revision"] = "e" * 40
    source_path.write_text(json.dumps(source), encoding="utf-8")
    assert (
        main(
            [
                "catalog",
                "kitten",
                "verify",
                "--catalog",
                str(catalog_path),
                "--source",
                str(source_path),
            ]
        )
        == 2
    )
    assert "Source repositories do not match catalog provenance" in capsys.readouterr().err


def test_build_surfaces_upstream_network_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail_request(*args, **kwargs):
        raise URLError("offline")

    monkeypatch.setattr(kitten_tools.urllib.request, "urlopen", fail_request)
    with pytest.raises(kitten_tools.CatalogError, match="Unable to fetch"):
        kitten_tools.build_catalog(_seed_catalog())
