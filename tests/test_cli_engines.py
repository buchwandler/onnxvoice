from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import replace
from typing import Any

import pytest

from onnxvoice import cli
from onnxvoice.types import CatalogItem


class FakeCatalog:
    def __init__(self) -> None:
        self.items = [
            CatalogItem(
                system="qwen3",
                id="base-0.6b",
                kind="bundle",
                artifacts=(),
                aliases=("base-small",),
                sample_rate=24_000,
                metadata={
                    "model_type": "base",
                    "profiles": {"fp32": {}, "int4": {}},
                    "runtime": {"model_type": "base", "layout": "qwen3-wavekat-v1"},
                },
            ),
            CatalogItem(
                system="pocket",
                id="pocket-clone",
                kind="bundle",
                artifacts=(),
                sample_rate=24_000,
                metadata={"supports_voice_cloning": True},
            ),
        ]
        self.resolve_calls: list[tuple[str, str | None, str | None]] = []

    def list(self, engine: str, refresh: bool = False) -> Sequence[CatalogItem]:
        del refresh
        return [item for item in self.items if item.system == engine]

    def resolve(
        self,
        ref: str,
        *,
        refresh: bool = False,
        quality: str | None = None,
        distribution: str | None = None,
    ) -> CatalogItem:
        del refresh
        self.resolve_calls.append((ref, quality, distribution))
        engine, target = ref.split(":", 1)
        item = next(item for item in self.list(engine) if item.id == target)
        metadata = dict(item.metadata)
        if quality is not None:
            metadata["selected_quality"] = quality
        return replace(item, metadata=metadata)


class FakeManager:
    def __init__(self) -> None:
        self.catalog = FakeCatalog()

    def installed(self, system: str | None = None) -> list[Any]:
        del system
        return []


def _patch_manager(monkeypatch: pytest.MonkeyPatch) -> tuple[FakeManager, list[Any]]:
    manager = FakeManager()
    seen_args: list[Any] = []

    def manager_factory(args: Any) -> FakeManager:
        seen_args.append(args)
        return manager

    monkeypatch.setattr(cli, "_manager", manager_factory)
    return manager, seen_args


def test_options_json_selectors_and_offline_mode_do_not_acquire_models(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    manager, seen_args = _patch_manager(monkeypatch)
    result = cli.main(
        [
            "--offline",
            "engines",
            "options",
            "--engine",
            "qwen3",
            "--target",
            "base-small",
            "--quality",
            "int4",
            "--format",
            "json",
        ]
    )
    assert result == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["target"] == "base-0.6b"
    assert payload["quality"] == "int4"
    assert payload["capabilities"]["voice_cloning"]["supported"] is True
    assert seen_args[0].offline is True
    assert manager.catalog.resolve_calls == [("qwen3:base-0.6b", "int4", None)]
    assert not hasattr(manager, "install")
    assert not hasattr(manager, "open")


def test_options_support_table_plain_and_tsv_renderers(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _patch_manager(monkeypatch)
    for output_format in ("table", "plain", "tsv"):
        result = cli.main(
            [
                "engines",
                "options",
                "--engine",
                "qwen3",
                "--target",
                "base-0.6b",
                "--format",
                output_format,
            ]
        )
        assert result == 0
        output = capsys.readouterr().out
        assert "voice_cloning" in output
        assert "base-0.6b" in output
        if output_format == "table":
            assert "SECTION" in output
        elif output_format == "plain":
            assert "SECTION" not in output
        else:
            assert output.splitlines()[0].startswith("SECTION\tNAME\tTYPE\tVALUE\tDETAIL")


def test_list_capability_json_and_unknown_selector_errors(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _patch_manager(monkeypatch)
    result = cli.main(
        [
            "engines",
            "list",
            "--engine",
            "pocket",
            "--capability",
            "voice-cloning",
            "--format",
            "json",
        ]
    )
    assert result == 0
    payload = json.loads(capsys.readouterr().out)
    assert [row["target"] for row in payload["rows"]] == ["pocket-clone"]

    result = cli.main(
        ["engines", "options", "--engine", "qwen3", "--target", "missing", "--format", "json"]
    )
    captured = capsys.readouterr()
    assert result == 2
    assert "Unknown target 'missing'" in captured.err
    assert "base-0.6b" in captured.err


def test_list_plain_and_tsv_are_rendered(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _patch_manager(monkeypatch)
    for output_format in ("plain", "tsv"):
        result = cli.main(["engines", "list", "--engine", "pocket", "--format", output_format])
        assert result == 0
        output = capsys.readouterr().out
        assert "pocket-clone" in output
        if output_format == "tsv":
            assert output.splitlines()[0].startswith("ENGINE\tTARGET\tREF")
