from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from onnxvoice.systems import kitten2_frontend as frontend_module
from onnxvoice.systems.kitten2_frontend import Kitten2Frontend
from onnxvoice.systems.kitten2_tokens import TokenMap


class RecordingTokenizer:
    def __init__(self) -> None:
        self.inputs: list[str] = []

    def encode(self, text: str) -> list[int]:
        self.inputs.append(text)
        return [17, 23]


def _frontend(tmp_path: Path) -> tuple[Kitten2Frontend, RecordingTokenizer]:
    frontend = Kitten2Frontend(tmp_path, {"tokenizer": "tokenizer"}, _token_map())
    tokenizer = RecordingTokenizer()
    frontend._tokenizer = tokenizer  # type: ignore[assignment]
    return frontend, tokenizer


def _token_map() -> TokenMap:
    return TokenMap(
        audio_id_base=100,
        num_audio_tokens=10,
        speech_start_id=1,
        speech_end_id=2,
        text_start_id=3,
        text_end_id=4,
        start_id=5,
        stop_id=6,
    )


@pytest.mark.parametrize(
    "text",
    [
        "Dr. Smith paid $12.50 on 10/12/2026.",
        "100 km, 5th Avenue, and 2 units.",
        "Café naïve こんにちは 🌍",
        "First line.\nSecond line.\n",
        "  preserve both sides exactly.  ",
    ],
)
def test_prepare_target_passes_caller_text_unchanged(tmp_path: Path, text: str) -> None:
    frontend, tokenizer = _frontend(tmp_path)

    target = frontend.prepare_target(text)

    assert target.text == text
    assert target.token_ids == (17, 23)
    assert not target.has_expression
    assert tokenizer.inputs == [text]


def test_prepare_target_preserves_expression_markup(tmp_path: Path) -> None:
    frontend, tokenizer = _frontend(tmp_path)
    text = "[angry] I told you (((never))) to open that door. <sigh>"

    target = frontend.prepare_target(text)

    assert target.text == text
    assert target.has_expression
    assert tokenizer.inputs == [text]


def test_prepare_target_validates_without_loading_tokenizer(tmp_path: Path) -> None:
    frontend = Kitten2Frontend(tmp_path, {"tokenizer": "tokenizer"}, _token_map())

    with pytest.raises(TypeError, match="text must be a string"):
        frontend.prepare_target(None)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="text must not be empty"):
        frontend.prepare_target(" \t\n")
    assert frontend._tokenizer is None


def test_voice_reference_transcript_reaches_tokenizer_unchanged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    transcript = "  Stored reference transcript.\n"
    arrays: dict[str, Any] = {
        "reference_tokens": np.array([0]),
        "speaker_embedding": np.zeros((1, 4), dtype=np.float32),
        "conditioning_prompt_token": np.array([1]),
        "conditioning_prompt_token_len": np.array([1]),
        "conditioning_prompt_feat": np.zeros((1, 1), dtype=np.float32),
        "conditioning_embedding": np.zeros((1, 4), dtype=np.float32),
    }
    reference = SimpleNamespace(metadata={"reference_transcript": transcript}, arrays=arrays)
    monkeypatch.setattr(frontend_module, "load_decoder_reference", lambda _path: reference)
    frontend, tokenizer = _frontend(tmp_path)
    frontend.manifest["voices"] = {"Bruno": "voices/Bruno.npz"}

    frontend.prepare_voice("Bruno")

    assert tokenizer.inputs == [transcript]


def test_emotion_control_is_tokenized_once_with_the_bundled_tokenizer(tmp_path: Path) -> None:
    frontend = Kitten2Frontend(
        tmp_path,
        {"tokenizer": "tokenizer", "generation": {"emotion_control": "{emo: 1}"}},
        _token_map(),
    )
    tokenizer = RecordingTokenizer()
    frontend._tokenizer = tokenizer  # type: ignore[assignment]

    assert frontend.emotion_control_ids == (17, 23)
    assert frontend.emotion_control_ids == (17, 23)
    assert tokenizer.inputs == ["{emo: 1}"]


def test_importing_base_package_does_not_require_tokenizers() -> None:
    code = """
import builtins
original_import = builtins.__import__
def import_without_tokenizers(name, *args, **kwargs):
    if name == "tokenizers" or name.startswith("tokenizers."):
        raise ImportError("tokenizers unavailable")
    return original_import(name, *args, **kwargs)
builtins.__import__ = import_without_tokenizers
import onnxvoice
assert onnxvoice is not None
"""
    repository = Path(__file__).parents[1]
    environment = os.environ.copy()
    subprocess.run([sys.executable, "-c", code], cwd=repository, env=environment, check=True)
