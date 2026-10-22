from __future__ import annotations

import ast
import builtins
import json
import os
from pathlib import Path
from typing import Any

import pytest

from onnxvoice.errors import OptionalDependencyError, RuntimeContractError
from onnxvoice.systems.kitten2_tokenizer import Kitten2Tokenizer

_FIXTURE = Path(__file__).parent / "fixtures/kitten2_tokenizer_golden.json"


def _golden_vectors() -> list[dict[str, Any]]:
    return json.loads(_FIXTURE.read_text(encoding="utf-8"))["vectors"]


def _tokenizer_json() -> Path:
    raw_bundle = os.environ.get("ONNXVOICE_KITTEN2_BUNDLE")
    if not raw_bundle:
        pytest.skip("set ONNXVOICE_KITTEN2_BUNDLE to a KittenTTS 2 bundle for tokenizer vectors")
    bundle_path = Path(raw_bundle)
    root = bundle_path.parent if bundle_path.name == "bundle.json" else bundle_path
    return root / "tokenizer" / "tokenizer.json"


def test_bundled_tokenizer_matches_transformers_golden_vectors() -> None:
    tokenizer = Kitten2Tokenizer(_tokenizer_json())

    for vector in _golden_vectors():
        assert tokenizer.encode(vector["text"]) == vector["ids"], vector["name"]


def test_tokenizer_requires_tokenizer_json(tmp_path: Path) -> None:
    with pytest.raises(RuntimeContractError, match="required tokenizer artifact is missing"):
        Kitten2Tokenizer(tmp_path / "tokenizer.json")


def test_tokenizer_reports_missing_optional_dependency(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tokenizer_json = tmp_path / "tokenizer.json"
    tokenizer_json.write_text("{}", encoding="utf-8")
    original_import = builtins.__import__

    def import_without_tokenizers(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "tokenizers":
            raise ImportError("tokenizers unavailable")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_without_tokenizers)
    with pytest.raises(OptionalDependencyError, match=r"onnxvoice\[kitten2\]"):
        Kitten2Tokenizer(tokenizer_json)


def test_kitten2_runtime_has_no_semantic_text_dependencies() -> None:
    repository = Path(__file__).parents[1]
    production = repository / "onnxvoice/systems"
    forbidden = {
        "kitten_text_processing",
        "transformers",
        "spokenform",
        "num2words",
        "nltk",
        "spacy",
        "phonemizer",
    }
    for path in production.glob("kitten2*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        imported: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module.split(".")[0])
        assert not imported & forbidden, path.name

    project = (repository / "pyproject.toml").read_text(encoding="utf-8").lower()
    assert "kitten-text-processing" not in project
    assert "transformers" not in project
    assert 'kitten2 = ["tokenizers>=0.22,<0.23"]' in project
