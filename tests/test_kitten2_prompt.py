from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from _kitten2_fixture import create_bundle

from onnxvoice import OnnxVoice
from onnxvoice.systems.kitten2_frontend import (
    PreparedTarget,
    PreparedVoice,
    build_generation_prompt,
)
from onnxvoice.systems.kitten2_generation import GenerationResult
from onnxvoice.systems.kitten2_tokens import TokenMap


@pytest.mark.parametrize(
    ("use_reference", "use_emotion", "expected"),
    [
        (True, True, [10, 11, 20, 21, 30, 40, 41, 50, 60]),
        (False, False, [1, 30, 40, 41, 50, 60]),
    ],
)
def test_generation_prompt_places_emotion_before_text_start(
    use_reference: bool, use_emotion: bool, expected: list[int]
) -> None:
    token_map = TokenMap(
        audio_id_base=100,
        num_audio_tokens=10,
        speech_start_id=60,
        speech_end_id=61,
        text_start_id=30,
        text_end_id=31,
        start_id=1,
        stop_id=2,
        final_seg_id=50,
    )

    prompt = build_generation_prompt(
        token_map,
        [40, 41],
        reference_prefix=[10, 11] if use_reference else None,
        emotion_ids=[20, 21] if use_emotion else None,
    )

    assert prompt == expected


@pytest.mark.parametrize(
    (
        "has_expression",
        "use_emotion",
        "configured_reference",
        "use_reference_prompt",
        "expected_emotion",
        "expected_reference",
    ),
    [
        (True, None, True, None, True, True),
        (False, None, True, None, False, True),
        (False, True, True, None, True, True),
        (True, False, True, None, False, True),
        (True, None, False, None, True, False),
        (True, None, False, True, True, True),
        (True, None, True, False, True, False),
    ],
)
def test_adapter_resolves_expression_and_reference_overrides(
    tmp_path: Path,
    has_expression: bool,
    use_emotion: bool | None,
    configured_reference: bool,
    use_reference_prompt: bool | None,
    expected_emotion: bool,
    expected_reference: bool,
) -> None:
    bundle_path = create_bundle(tmp_path / "bundle")

    manifest_path = bundle_path
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["generation"]["use_reference_prompt"] = configured_reference
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    runtime = OnnxVoice.open_local(system="kitten2", files={"bundle": bundle_path})

    class FrontendStub:
        emotion_control_reads = 0
        last_text: str | None = None

        def prepare_target(self, text: str) -> PreparedTarget:
            self.last_text = text
            return PreparedTarget(text=text, token_ids=(40, 41), has_expression=has_expression)

        @property
        def emotion_control_ids(self) -> tuple[int, ...]:
            self.emotion_control_reads += 1
            return (20, 21)

        def prepare_voice(self, voice: str) -> PreparedVoice:
            assert voice == "Bruno"
            return PreparedVoice(
                speaker_embedding=np.zeros((1, 4), dtype=np.float32),
                reference_prefix=[10, 11],
                conditioning={},
            )

    class LanguageModelStub:
        prompt: list[int] | None = None

        def generate(self, prompt: list[int], **_kwargs: Any) -> GenerationResult:
            self.prompt = prompt
            return GenerationResult(raw_ids=[100, 2], s3_ids=[0])

    class DecoderStub:
        def decode(
            self, _s3_ids: list[int], _conditioning: dict[str, np.ndarray], *, seed: int | None
        ) -> np.ndarray:
            return np.zeros(8, dtype=np.float32)

    frontend = FrontendStub()
    language_model = LanguageModelStub()
    runtime.frontend = frontend
    runtime._get_language_model = lambda: language_model
    runtime._get_decoder = lambda: DecoderStub()
    text = "  [joyful] unchanged target  "
    try:
        result = runtime.infer_text(
            text,
            use_emotion=use_emotion,
            use_reference_prompt=use_reference_prompt,
        )

        head = [10, 11] if expected_reference else [5]
        emotion = [20, 21] if expected_emotion else []
        assert language_model.prompt == [*head, *emotion, 3, 40, 41, 110, 1]
        assert frontend.last_text == text
        assert frontend.emotion_control_reads == int(expected_emotion)
        assert result.metadata["expression_conditioning"] is expected_emotion
    finally:
        runtime.close()
