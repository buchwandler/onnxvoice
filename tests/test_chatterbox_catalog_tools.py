from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from onnxvoice import cli
from onnxvoice.catalog_tools import chatterbox

REVISION = "a" * 40
DIGEST = "b" * 64
COMPONENTS = ("speech_encoder", "embed_tokens", "language_model", "conditional_decoder")


def catalog_fixture() -> dict[str, Any]:
    repository = "owner/turbo"
    artifacts = []
    for component in COMPONENTS:
        path = f"onnx/{component}.onnx"
        artifacts.append(
            {
                "role": "graph",
                "component": component,
                "quality": "fp32",
                "filename": f"{component}.onnx",
                "source_path": path,
                "url": chatterbox.huggingface_resolve_url(repository, REVISION, path),
                "size": 4,
                "sha256": DIGEST,
            }
        )
    tokenizer_path = "tokenizer.json"
    artifacts.append(
        {
            "role": "tokenizer",
            "component": None,
            "quality": None,
            "filename": tokenizer_path,
            "source_path": tokenizer_path,
            "url": chatterbox.huggingface_resolve_url(repository, REVISION, tokenizer_path),
            "size": 4,
            "sha256": DIGEST,
        }
    )
    return {
        "schema": 1,
        "kind": "chatterbox-onnx-bundle-catalog",
        "bundles": {
            "turbo": {
                "id": "turbo",
                "system": "chatterbox",
                "aliases": [],
                "language": "en",
                "sample_rate": 24000,
                "default_profile": "fp32",
                "source": {
                    "provider": "huggingface",
                    "repository": repository,
                    "requested_revision": "main",
                    "revision": REVISION,
                },
                "frontend": {"tokenizer_role": "tokenizer", "tokenizer_contract": "turbo"},
                "runtime": {
                    "start_speech_token": 6561,
                    "stop_speech_token": 6562,
                    "silence_token": 4299,
                    "trailing_silence_tokens": 3,
                    "default_max_new_tokens": 1024,
                    "default_repetition_penalty": 1.2,
                },
                "profiles": {"fp32": dict.fromkeys(COMPONENTS, "fp32")},
                "artifacts": artifacts,
            }
        },
    }


def test_catalog_cli_build_and_verify_are_offline_for_generated_data(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    catalog = catalog_fixture()
    seed_path = tmp_path / "seed.json"
    seed_path.write_text("{}", encoding="utf-8")
    catalog_path = tmp_path / "bundles.json"
    source_path = tmp_path / "source.json"
    monkeypatch.setattr(chatterbox, "build_catalog", lambda seed: catalog)

    assert (
        cli.main(
            [
                "catalog",
                "chatterbox",
                "build",
                "--seed",
                str(seed_path),
                "--output",
                str(catalog_path),
                "--source-output",
                str(source_path),
            ]
        )
        == 0
    )
    assert "Wrote 1 Chatterbox bundles" in capsys.readouterr().out
    first_catalog = catalog_path.read_bytes()
    first_source = source_path.read_bytes()
    assert (
        cli.main(
            [
                "catalog",
                "chatterbox",
                "build",
                "--seed",
                str(seed_path),
                "--output",
                str(catalog_path),
                "--source-output",
                str(source_path),
            ]
        )
        == 0
    )
    assert catalog_path.read_bytes() == first_catalog
    assert source_path.read_bytes() == first_source

    assert (
        cli.main(
            [
                "catalog",
                "chatterbox",
                "verify",
                "--catalog",
                str(catalog_path),
                "--source",
                str(source_path),
            ]
        )
        == 0
    )
    assert "Verified 1 Chatterbox bundles" in capsys.readouterr().out

    invalid = json.loads(source_path.read_text(encoding="utf-8"))
    invalid["artifact_count"] += 1
    source_path.write_text(json.dumps(invalid), encoding="utf-8")
    assert (
        cli.main(
            [
                "catalog",
                "chatterbox",
                "verify",
                "--catalog",
                str(catalog_path),
                "--source",
                str(source_path),
            ]
        )
        == 2
    )
    assert "Source metadata does not match" in capsys.readouterr().err
