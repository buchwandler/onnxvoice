from pathlib import Path
from typing import Any

import numpy as np
import pytest

from onnxvoice import OnnxVoice
from onnxvoice.errors import RuntimeContractError
from onnxvoice.systems import get_adapter
from onnxvoice.systems.chatterbox import ChatterboxAdapter, ChatterboxVoiceState
from onnxvoice.types import Installation, InstalledArtifact, TensorSpec

_RUNTIME = {
    "start_speech_token": 1,
    "stop_speech_token": 2,
    "silence_token": 4,
    "trailing_silence_tokens": 3,
    "default_max_new_tokens": 4,
    "default_repetition_penalty": 1.2,
    "sample_rate": 24000,
    "profile": "fp32",
}


class FakeSession:
    instances: list[Any] = []
    lm_step = 0
    stop_after_first = True
    include_position_ids = True
    broken_cache_contract = False
    nan_logits = False

    def __init__(self, model: str | Path, **kwargs: Any) -> None:
        self.model = Path(model)
        self.component = self.model.stem
        self.calls: list[dict[str, np.ndarray]] = []
        self.closed = False
        self.input_specs, self.output_specs = self._contract()
        self.__class__.instances.append(self)

    def _contract(self) -> tuple[tuple[TensorSpec, ...], tuple[TensorSpec, ...]]:
        float32 = "tensor(float)"
        int64 = "tensor(int64)"
        float16 = "tensor(float16)"
        if self.component == "speech_encoder":
            return (
                (TensorSpec("audio_values", float32, (1, "samples")),),
                (
                    TensorSpec("cond_emb", float32, (1, 1, 2)),
                    TensorSpec("prompt_token", int64, (1, 2)),
                    TensorSpec("speaker_embeddings", float32, (1, 2)),
                    TensorSpec("speaker_features", float32, (1, 2)),
                ),
            )
        if self.component == "embed_tokens":
            return (
                (TensorSpec("input_ids", int64, (1, "sequence")),),
                (TensorSpec("inputs_embeds", float32, (1, "sequence", 2)),),
            )
        if self.component == "language_model":
            inputs = [
                TensorSpec("inputs_embeds", float32, (1, "sequence", 2)),
                TensorSpec("attention_mask", int64, (1, "total_sequence")),
            ]
            if self.include_position_ids:
                inputs.append(TensorSpec("position_ids", int64, (1, "sequence")))
            inputs.extend(
                (
                    TensorSpec("past_key_values.0.key", float16, (1, 2, "past_sequence", 2)),
                    TensorSpec("past_key_values.0.value", float16, (1, 2, "past_sequence", 2)),
                )
            )
            outputs = [TensorSpec("logits", float32, (1, "sequence", 5))]
            outputs.append(TensorSpec("present.0.key", float16, (1, 2, "total_sequence", 2)))
            if not self.broken_cache_contract:
                outputs.append(TensorSpec("present.0.value", float16, (1, 2, "total_sequence", 2)))
            return tuple(inputs), tuple(outputs)
        if self.component == "conditional_decoder":
            return (
                (
                    TensorSpec("speech_tokens", int64, (1, "tokens")),
                    TensorSpec("speaker_embeddings", float32, (1, 2)),
                    TensorSpec("speaker_features", float32, (1, 2)),
                ),
                (TensorSpec("waveform", float32, (1, 1, 4)),),
            )
        raise AssertionError(f"unexpected graph {self.component}")

    def run(self, feeds: dict[str, np.ndarray]) -> list[np.ndarray]:
        self.calls.append({name: np.array(value, copy=True) for name, value in feeds.items()})
        if self.component == "speech_encoder":
            return [
                np.ones((1, 1, 2), np.float32),
                np.asarray([[8, 9]], np.int64),
                np.ones((1, 2), np.float32),
                np.full((1, 2), 2, np.float32),
            ]
        if self.component == "embed_tokens":
            ids = feeds["input_ids"].astype(np.float32)
            return [np.repeat(ids[:, :, None], 2, axis=2)]
        if self.component == "language_model":
            total = feeds["past_key_values.0.key"].shape[2] + feeds["inputs_embeds"].shape[1]
            logits = np.zeros((1, feeds["inputs_embeds"].shape[1], 5), np.float32)
            if self.stop_after_first and self.lm_step > 0:
                logits[:, -1, 2] = 5
            else:
                logits[:, -1, 3] = 5
            if self.nan_logits:
                logits[0, -1, 3] = np.nan
            self.__class__.lm_step += 1
            key = np.zeros((1, 2, total, 2), np.float16)
            value = np.ones((1, 2, total, 2), np.float16)
            if self.broken_cache_contract:
                return [logits, key]
            return [logits, key, value]
        if self.component == "conditional_decoder":
            return [np.asarray([[[0.1, 0.2, 0.3, 0.4]]], np.float32)]
        raise AssertionError(f"unexpected graph {self.component}")

    def close(self) -> None:
        self.closed = True


def _adapter() -> ChatterboxAdapter:
    artifacts = tuple(
        InstalledArtifact(
            role="graph",
            filename=f"{name}.onnx",
            path=Path(f"/fake/{name}.onnx"),
            sha256="0" * 64,
            size=1,
            component=name,
        )
        for name in ("speech_encoder", "embed_tokens", "language_model", "conditional_decoder")
    )
    installation = Installation(
        system="chatterbox",
        id="chatterbox-nano-v2",
        kind="model",
        path=Path("/fake"),
        artifacts=artifacts,
        sample_rate=24000,
        metadata={"runtime": dict(_RUNTIME), "selected_quality": "fp32"},
    )
    return ChatterboxAdapter(installation)


def _reset_fake() -> None:
    FakeSession.instances = []
    FakeSession.lm_step = 0
    FakeSession.stop_after_first = True
    FakeSession.include_position_ids = True
    FakeSession.broken_cache_contract = False
    FakeSession.nan_logits = False


def _session(component: str) -> FakeSession:
    return next(session for session in FakeSession.instances if session.component == component)


def test_open_local_uses_generic_role_component_artifacts(tmp_path: Path) -> None:
    files = {}
    for component in ("speech_encoder", "embed_tokens", "language_model", "conditional_decoder"):
        path = tmp_path / f"{component}.onnx"
        path.touch()
        files[f"graph:{component}"] = path

    adapter = OnnxVoice.open_local(
        system="chatterbox",
        files=files,
        metadata={"runtime": dict(_RUNTIME), "selected_quality": "fp32"},
        sample_rate=24000,
    )

    assert isinstance(adapter, ChatterboxAdapter)
    assert {artifact.component for artifact in adapter.installation.artifacts} == {
        "speech_encoder",
        "embed_tokens",
        "language_model",
        "conditional_decoder",
    }


def test_chatterbox_adapter_is_registered() -> None:
    assert get_adapter("chatterbox") is ChatterboxAdapter


def test_voice_state_is_immutable_and_checks_prompt_tokens() -> None:
    state = ChatterboxVoiceState(
        np.ones((1, 1, 2), np.float32),
        np.asarray([[8, 9]], np.int64),
        np.ones((1, 2), np.float32),
        np.ones((1, 2), np.float32),
    )
    assert not state.cond_emb.flags.writeable
    with pytest.raises(ValueError):
        state.cond_emb[0, 0, 0] = 2
    with pytest.raises(RuntimeContractError, match="must be integers"):
        ChatterboxVoiceState(np.ones((1, 1, 2)), [1.5], np.ones((1, 2)), np.ones((1, 2)))


def test_prepare_voice_and_reusable_state_run_all_four_graphs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _reset_fake()
    monkeypatch.setattr("onnxvoice.systems.chatterbox.OnnxSession", FakeSession)
    adapter = _adapter()
    state = adapter.prepare_voice(np.arange(5, dtype=np.float32))
    assert _session("speech_encoder").calls[0]["audio_values"].shape == (1, 5)
    assert state.prompt_token.tolist() == [[8, 9]]
    first = adapter.infer([10, 11], voice_state=state, max_new_tokens=4)
    second = adapter.infer([12], voice_state=state, max_new_tokens=1)

    assert first.audio.tolist() == pytest.approx([0.1, 0.2, 0.3, 0.4])
    assert first.sample_rate == 24000
    assert first.metadata["generated_tokens"] == 2
    assert first.metadata["stopped_on_eos"] is True
    assert second.metadata["generated_tokens"] == 1
    assert len(FakeSession.instances) == 4
    assert len(_session("speech_encoder").calls) == 1
    assert {session.component for session in FakeSession.instances} == {
        "speech_encoder",
        "embed_tokens",
        "language_model",
        "conditional_decoder",
    }
    adapter.close()
    assert all(session.closed for session in FakeSession.instances)


def test_kv_dtype_optional_positions_and_decoder_silence(monkeypatch: pytest.MonkeyPatch) -> None:
    _reset_fake()
    FakeSession.include_position_ids = False
    monkeypatch.setattr("onnxvoice.systems.chatterbox.OnnxSession", FakeSession)
    adapter = _adapter()
    state = adapter.prepare_voice(np.ones(8, np.float32))
    result = adapter.infer([5, 6], voice_state=state, max_new_tokens=4)

    language_model = _session("language_model")
    assert language_model.calls[0]["past_key_values.0.key"].dtype == np.float16
    assert language_model.calls[0]["past_key_values.0.value"].dtype == np.float16
    assert "position_ids" not in language_model.calls[0]
    assert language_model.calls[0]["attention_mask"].shape == (1, 3)
    assert language_model.calls[1]["attention_mask"].shape == (1, 4)
    decoder_tokens = _session("conditional_decoder").calls[0]["speech_tokens"].tolist()[0]
    assert decoder_tokens == [8, 9, 3, 4, 4, 4]
    assert result.outputs["speech_tokens"].tolist() == [1, 3, 2]


def test_position_ids_follow_generation_positions(monkeypatch: pytest.MonkeyPatch) -> None:
    _reset_fake()
    monkeypatch.setattr("onnxvoice.systems.chatterbox.OnnxSession", FakeSession)
    adapter = _adapter()
    state = adapter.prepare_voice(np.ones(8, np.float32))
    adapter.infer([5, 6], voice_state=state, max_new_tokens=4)

    calls = _session("language_model").calls
    assert calls[0]["position_ids"].tolist() == [[0, 1, 2]]
    assert calls[1]["position_ids"].tolist() == [[3]]


def test_hard_generation_limit_appends_exactly_three_silence_tokens(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _reset_fake()
    FakeSession.stop_after_first = False
    monkeypatch.setattr("onnxvoice.systems.chatterbox.OnnxSession", FakeSession)
    adapter = _adapter()
    state = adapter.prepare_voice(np.ones(8, np.float32))
    result = adapter.infer([5], voice_state=state, max_new_tokens=2)

    assert result.metadata["generated_tokens"] == 2
    assert result.metadata["stopped_on_eos"] is False
    assert _session("conditional_decoder").calls[0]["speech_tokens"].tolist()[0] == [
        8,
        9,
        3,
        3,
        4,
        4,
        4,
    ]


def test_repetition_penalty_matches_greedy_formula() -> None:
    scores = ChatterboxAdapter._penalize(np.asarray([-2.0, 3.0, 1.0]), [0, 1, 1], 2.0)
    assert scores.tolist() == pytest.approx([-4.0, 1.5, 1.0])


def test_non_finite_logits_are_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    _reset_fake()
    FakeSession.nan_logits = True
    monkeypatch.setattr("onnxvoice.systems.chatterbox.OnnxSession", FakeSession)
    adapter = _adapter()
    state = adapter.prepare_voice(np.ones(8, np.float32))

    with pytest.raises(RuntimeContractError, match="non-finite logits"):
        adapter.infer([5], voice_state=state, max_new_tokens=2)


def test_bad_kv_contract_fails_before_generation(monkeypatch: pytest.MonkeyPatch) -> None:
    _reset_fake()
    FakeSession.broken_cache_contract = True
    monkeypatch.setattr("onnxvoice.systems.chatterbox.OnnxSession", FakeSession)
    adapter = _adapter()

    with pytest.raises(RuntimeContractError, match="past/present KV layouts do not match"):
        adapter._get_session("language_model")


def test_all_sessions_are_lazy_until_a_graph_is_used(monkeypatch: pytest.MonkeyPatch) -> None:
    _reset_fake()
    monkeypatch.setattr("onnxvoice.systems.chatterbox.OnnxSession", FakeSession)
    adapter = _adapter()
    assert FakeSession.instances == []
    adapter.prepare_voice(np.ones(4, np.float32))
    assert [session.component for session in FakeSession.instances] == ["speech_encoder"]
