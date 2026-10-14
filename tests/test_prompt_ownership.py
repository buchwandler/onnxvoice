from __future__ import annotations

import importlib.util

import pytest

import onnxvoice
from onnxvoice import errors
from onnxvoice.catalog_tools import pocket as pocket_catalog_tools
from onnxvoice.cli import build_parser
from onnxvoice.manager import OnnxVoice


def test_prompt_manager_and_package_api_are_absent(tmp_path) -> None:
    manager = OnnxVoice(cache_dir=tmp_path)

    assert not hasattr(manager, "voice_prompts")
    assert not hasattr(manager, "list_pocket_voice_" + "prompts")
    assert not hasattr(manager, "resolve_pocket_voice_" + "prompt")
    assert not hasattr(manager, "fetch_pocket_voice_" + "prompt")
    assert not any(name.startswith("PocketVoice" + "Prompt") for name in onnxvoice.__all__)
    assert not any(name.startswith("Voice" + "Prompt") for name in vars(errors))
    assert importlib.util.find_spec("onnxvoice.pocket_voice" + "_prompts") is None


def test_pocket_catalog_tooling_is_bundle_only() -> None:
    assert hasattr(pocket_catalog_tools, "build_catalog")
    assert hasattr(pocket_catalog_tools, "verify_catalog")
    assert not any("prompt" in name.casefold() for name in vars(pocket_catalog_tools))


def test_pocket_catalog_cli_is_bundle_only(capsys) -> None:
    with pytest.raises(SystemExit) as exit_info:
        build_parser().parse_args(["catalog", "pocket", "--help"])

    assert exit_info.value.code == 0
    help_text = capsys.readouterr().out
    assert "build" in help_text
    assert "verify" in help_text
    assert "prompts" not in help_text
