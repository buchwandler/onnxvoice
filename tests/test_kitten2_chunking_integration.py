from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from _kitten2_fixture import create_bundle

from onnxvoice import OnnxVoice
from onnxvoice.systems.kitten2_expression import inspect_expression_markup
from onnxvoice.systems.kitten2_frontend import PreparedTarget, PreparedVoice
from onnxvoice.systems.kitten2_generation import GenerationResult


def test_adapter_chunks_unchanged_text_and_shares_request_rng(tmp_path: Path) -> None:
    bundle_path = create_bundle(tmp_path / "bundle")
    manifest = json.loads(bundle_path.read_text(encoding="utf-8"))
    manifest["generation"]["chunk_chars"] = 30
    manifest["generation"]["chunk_min_chars"] = 1
    bundle_path.write_text(json.dumps(manifest), encoding="utf-8")
    runtime = OnnxVoice.open_local(system="kitten2", files={"bundle": bundle_path})

    class FrontendStub:
        texts: list[str] = []

        @property
        def emotion_control_ids(self) -> tuple[int, ...]:
            return (20, 21)

        def prepare_target(self, text: str) -> PreparedTarget:
            self.texts.append(text)
            return PreparedTarget(
                text=text,
                token_ids=(len(text),),
                has_expression=inspect_expression_markup(text).has_expression,
            )

        def prepare_voice(self, _voice: str) -> PreparedVoice:
            return PreparedVoice(
                speaker_embedding=np.zeros((1, 4), dtype=np.float32),
                reference_prefix=[10, 11],
                conditioning={},
            )

    class LanguageModelStub:
        rngs: list[np.random.Generator] = []
        draws: list[float] = []
        prompts: list[list[int]] = []

        def generate(
            self, prompt: list[int], *, rng: np.random.Generator, **_kwargs: Any
        ) -> GenerationResult:
            self.rngs.append(rng)
            self.draws.append(float(rng.random()))
            self.prompts.append(prompt)
            return GenerationResult(raw_ids=[100, 2], s3_ids=[0])

    class DecoderStub:
        def decode(
            self, _s3_ids: list[int], _conditioning: dict[str, np.ndarray], *, seed: int | None
        ) -> np.ndarray:
            assert seed == 7
            return np.array([0.25, -0.25], dtype=np.float32)

    frontend = FrontendStub()
    language_model = LanguageModelStub()
    runtime.frontend = frontend
    runtime._get_language_model = lambda: language_model
    runtime._get_decoder = lambda: DecoderStub()
    text = "[joyful] We won today. Another sentence with <laugh> after the close."
    try:
        first = runtime.infer_text(text, seed=7)
        first_chunks = frontend.texts[:]
        first_draws = language_model.draws[:]
        first_rngs = language_model.rngs[:]
        first_prompts = language_model.prompts[:]

        assert first.metadata["chunk_count"] == len(first_chunks) > 1
        assert "".join(first_chunks) == text
        assert all(rng is first_rngs[0] for rng in first_rngs)
        assert first_draws == list(np.random.default_rng(7).random(len(first_chunks)))
        assert all(prompt[:4] == [10, 11, 20, 21] for prompt in first_prompts)
        assert all(prompt[4] == 3 for prompt in first_prompts)
        assert first.metadata["expression_conditioning"] is True

        frontend.texts.clear()
        language_model.draws.clear()
        language_model.rngs.clear()
        language_model.prompts.clear()
        second = runtime.infer_text(text, seed=7)

        assert second.metadata["chunk_count"] == first.metadata["chunk_count"]
        assert second.audio.tolist() == first.audio.tolist()
        assert language_model.draws == first_draws
        assert len({id(rng) for rng in language_model.rngs}) == 1
        assert language_model.rngs[0] is not first_rngs[0]
    finally:
        runtime.close()
