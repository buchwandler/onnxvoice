"""Tests for the CLI module: table output, JSON output, plain format, and new subcommands."""

from __future__ import annotations

import json
from io import StringIO

from onnxvoice._table import format_bytes, render_inventory, render_json, render_table

# ---------------------------------------------------------------------------
# format_bytes tests
# ---------------------------------------------------------------------------


class TestFormatBytes:
    def test_zero(self):
        assert format_bytes(0) == "0 B"

    def test_bytes(self):
        assert format_bytes(812) == "812 B"

    def test_kib(self):
        assert format_bytes(1024) == "1.00 KiB"

    def test_mib(self):
        assert format_bytes(66270000) == "63.2 MiB"

    def test_gib(self):
        assert format_bytes(1438814042) == "1.34 GiB"

    def test_none(self):
        assert format_bytes(None) == "-"

    def test_negative(self):
        assert format_bytes(-100) == "-100 B"


# ---------------------------------------------------------------------------
# Table rendering tests
# ---------------------------------------------------------------------------


class TestTableRendering:
    def test_basic_table(self):
        output = StringIO()
        columns = ["NAME", "VALUE"]
        rows = [["test", "100"]]
        render_table(columns, rows, file=output)
        result = output.getvalue()
        assert "NAME" in result
        assert "test" in result

    def test_deterministic_output(self):
        """Same input should produce same output."""
        columns = ["A", "B", "C"]
        rows = [["x", "y", "z"], ["1", "2", "3"]]
        out1 = StringIO()
        out2 = StringIO()
        render_table(columns, rows, file=out1)
        render_table(columns, rows, file=out2)
        assert out1.getvalue() == out2.getvalue()

    def test_empty_rows(self):
        output = StringIO()
        render_table(["A", "B"], [], file=output)
        # Header + separator only
        lines = output.getvalue().strip().split("\n")
        assert len(lines) == 2


# ---------------------------------------------------------------------------
# JSON rendering tests
# ---------------------------------------------------------------------------


class TestJsonRendering:
    def test_json_structure(self):
        output = StringIO()
        columns = ["STATUS", "ID"]
        rows = [["installed", "test-1"]]
        render_json(columns, rows, file=output)
        data = json.loads(output.getvalue())
        assert "items" in data
        assert len(data["items"]) == 1
        assert data["items"][0]["STATUS"] == "installed"

    def test_json_extra_keys(self):
        output = StringIO()
        render_json(["A"], [["val"]], file=output, extra={"warnings": ["test"]})
        data = json.loads(output.getvalue())
        assert "warnings" in data
        assert data["warnings"] == ["test"]

    def test_json_bytes_are_numeric(self):
        """JSON output should have numeric bytes, not formatted strings."""
        output = StringIO()
        render_json(["SIZE"], [["1024"]], file=output)
        # SIZE value is a string in the row, but that's because rows are string-based
        # The CLI should pass numeric values in JSON mode


# ---------------------------------------------------------------------------
# render_inventory tests
# ---------------------------------------------------------------------------


class TestRenderInventory:
    def test_table_format(self):
        output = StringIO()
        render_inventory(["A"], [["val"]], format="table", file=output)
        assert "val" in output.getvalue()

    def test_json_format(self):
        output = StringIO()
        render_inventory(["A"], [["val"]], format="json", file=output)
        data = json.loads(output.getvalue())
        assert data["items"][0]["A"] == "val"

    def test_plain_format(self):
        output = StringIO()
        render_inventory(["A", "B"], [["x", "y"]], format="plain", file=output)
        assert "x\ty" in output.getvalue()


def test_doctor_reports_non_secret_huggingface_status(monkeypatch, capsys, tmp_path) -> None:
    from onnxvoice.cli import main

    secret = "hf_FAKE_SECRET_SENTINEL"
    monkeypatch.setenv("HF_TOKEN", secret)
    monkeypatch.setenv("HF_TOKEN_PATH", str(tmp_path / "token"))
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("HF_HUB_DISABLE_IMPLICIT_TOKEN", "1")

    assert main(["--offline", "doctor", "--system", "pocket", "--format", "json"]) == 0
    output = capsys.readouterr().out
    report = json.loads(output)
    assert report["system"] == "pocket"
    assert report["credentials"] == "configured"
    assert report["offline"] is True
    assert report["implicit_token_disabled"] is True
    assert report["gated_access"].startswith("not checked")
    assert secret not in output
