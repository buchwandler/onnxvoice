"""CLI tests for the Supertonic catalog tools."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from onnxvoice import catalog_tools, cli
from onnxvoice.catalog_tools import supertonic as supertonic_tools


def _catalog() -> dict:
    return {
        "schema": 1,
        "kind": "supertonic-onnx-bundle-catalog",
        "source": {
            "provider": "huggingface",
            "repository": supertonic_tools.DEFAULT_REPOSITORY,
            "requested_revision": "main",
            "revision": "a" * 40,
            "bundle_count": 1,
        },
        "bundles": {"supertonic-3": {"artifacts": []}},
    }


def test_catalog_tools_exports_supertonic_builder_and_verifier() -> None:
    assert catalog_tools.build_supertonic_catalog is supertonic_tools.build_catalog
    assert catalog_tools.verify_supertonic_catalog is supertonic_tools.verify_catalog
    assert catalog_tools.load_supertonic_catalog is supertonic_tools.load_catalog
    assert catalog_tools.supertonic_tools is supertonic_tools


def test_supertonic_build_writes_catalog_and_provenance(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    catalog = _catalog()
    calls = []

    def build_catalog(*, repository: str, revision: str) -> dict:
        calls.append((repository, revision))
        return catalog

    monkeypatch.setattr(cli.supertonic_catalog_tools, "build_catalog", build_catalog)
    monkeypatch.setattr(cli.supertonic_catalog_tools, "verify_catalog", lambda value: None)
    output = tmp_path / "catalog.json"
    source_output = tmp_path / "source.json"

    result = cli.main(
        [
            "catalog",
            "supertonic",
            "build",
            "--output",
            str(output),
            "--source-output",
            str(source_output),
            "--repository",
            "example/model",
            "--revision",
            "release-tag",
        ]
    )

    assert result == 0
    assert calls == [("example/model", "release-tag")]
    assert json.loads(output.read_text(encoding="utf-8")) == catalog
    assert json.loads(source_output.read_text(encoding="utf-8")) == {
        "schema": 1,
        **catalog["source"],
    }
    assert capsys.readouterr().out == f"Wrote 1 bundles to {output}\n"


def test_supertonic_defaults_and_help() -> None:
    args = cli.build_parser().parse_args(
        ["catalog", "supertonic", "build", "--output", "catalog.json"]
    )
    assert args.repository == supertonic_tools.DEFAULT_REPOSITORY
    assert args.revision == supertonic_tools.DEFAULT_REVISION

    with pytest.raises(SystemExit) as error:
        cli.main(["catalog", "supertonic", "--help"])
    assert error.value.code == 0


def test_supertonic_verify_routes_catalog_and_source(tmp_path: Path, monkeypatch, capsys) -> None:
    catalog = _catalog()
    calls = []

    def load_catalog(path: Path, source: Path | None = None) -> dict:
        calls.append((path, source))
        return catalog

    monkeypatch.setattr(cli.supertonic_catalog_tools, "load_catalog", load_catalog)
    source = tmp_path / "source.json"
    result = cli.main(
        [
            "catalog",
            "supertonic",
            "verify",
            "--catalog",
            str(tmp_path / "catalog.json"),
            "--source",
            str(source),
        ]
    )

    assert result == 0
    assert calls == [(tmp_path / "catalog.json", source)]
    assert capsys.readouterr().out == "Verified 1 Supertonic bundles\n"


def test_supertonic_verify_reports_catalog_errors(monkeypatch, capsys) -> None:
    def fail_load(path: Path, source: Path | None = None) -> dict:
        raise supertonic_tools.CatalogError("invalid Supertonic catalog")

    monkeypatch.setattr(cli.supertonic_catalog_tools, "load_catalog", fail_load)
    result = cli.main(["catalog", "supertonic", "verify", "--catalog", "catalog.json"])

    assert result == 2
    assert capsys.readouterr().err == "onnxvoice: invalid Supertonic catalog\n"
