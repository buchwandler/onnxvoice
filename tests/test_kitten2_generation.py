from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
from _kitten2_fixture import create_bundle

from onnxvoice.errors import CapabilityError
from onnxvoice.systems import kitten2 as kitten2_module
from onnxvoice.systems import kitten2_generation as kitten2_generation_module
from onnxvoice.systems.kitten2 import Kitten2Adapter
from onnxvoice.systems.kitten2_generation import Kitten2LanguageModel
from onnxvoice.systems.kitten2_sampling import SamplingSettings
from onnxvoice.systems.kitten2_tokens import TokenMap
from onnxvoice.types import Installation, SessionDiagnostic, TensorSpec


class FakeLanguageSession:
    input_specs = (
        TensorSpec("inputs_embeds", "tensor(float)", ()),
        TensorSpec("attention_mask", "tensor(int64)", ()),
        TensorSpec("position_ids", "tensor(int64)", ()),
    )

    def run(self, inputs: dict[str, np.ndarray]) -> list[np.ndarray]:
        sequence_length = inputs["inputs_embeds"].shape[1]
        logits = np.full((1, 120), -100.0, dtype=np.float32)
        if inputs["past_key_0"].shape[2] > 0:
            logits[:, 2] = 10.0
        else:
            logits[:, 100] = 0.0
            logits[:, 101] = 0.0
        cache = np.zeros((1, 1, sequence_length, 1), dtype=np.float32)
        return [logits, cache, cache]


class FakeSpeakerSession:
    def run(self, inputs: dict[str, np.ndarray]) -> list[np.ndarray]:
        return [np.zeros((1, 4), dtype=np.float32)]


def test_native_language_model_generates_deterministically(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    embeddings_path = tmp_path / "embeddings.npy"
    np.save(embeddings_path, np.zeros((120, 4), dtype=np.float32))
    metadata = {
        "runtime_profile": "test",
        "io_dtype": "float32",
        "input_mode": "inputs_embeds",
        "embedding_storage_dtype": "float32",
        "num_hidden_layers": 1,
        "num_key_value_heads": 1,
        "head_dim": 1,
    }
    token_map = TokenMap.from_config(
        {
            "token_map": {
                "audio_id_base": 100,
                "num_audio_tokens": 10,
                "speech_start_id": 1,
                "speech_end_id": 2,
                "text_start_id": 3,
                "text_end_id": 4,
                "start_id": 5,
                "stop_id": 6,
                "final_seg_id": 110,
            }
        }
    )
    model = Kitten2LanguageModel(
        FakeLanguageSession(),
        FakeSpeakerSession(),
        metadata,
        embeddings_path,
    )
    settings = SamplingSettings(
        temperature=1.0,
        top_k=0,
        top_p=1.0,
        repetition_penalty=1.0,
        token_run_penalty=0.0,
    )
    first = model.generate(
        [5, 3], token_map=token_map, max_new_tokens=4, settings=settings, seed=17
    )
    second = model.generate(
        [5, 3], token_map=token_map, max_new_tokens=4, settings=settings, seed=17
    )
    assert first == second
    assert first.raw_ids in ([100, 2], [101, 2])
    assert first.s3_ids == [first.raw_ids[0] - token_map.audio_id_base]

    with_rng = model.generate(
        [5, 3],
        token_map=token_map,
        max_new_tokens=4,
        settings=settings,
        rng=np.random.default_rng(17),
    )
    assert with_rng == first

    samples: list[float] = []

    def capture_sample(*args: Any, **_kwargs: Any) -> int:
        samples.append(args[4])
        return token_map.speech_end_id

    monkeypatch.setattr(kitten2_generation_module, "sample", capture_sample)
    shared_rng = np.random.default_rng(31)
    for _ in range(2):
        model.generate(
            [5, 3], token_map=token_map, max_new_tokens=4, settings=settings, rng=shared_rng
        )
    assert samples == list(np.random.default_rng(31).random(2))


def test_adapter_defers_sessions_and_propagates_runtime_options(
    tmp_path: Path, monkeypatch: Any
) -> None:
    bundle_path = create_bundle(tmp_path)
    installation = Installation(
        system="kitten2",
        id="kitten2-onnx-bundle-v1",
        kind="tts",
        path=bundle_path.parent,
        artifacts=(),
        sample_rate=24000,
        metadata={"runtime": {"layout": "kitten2-onnx-bundle-v1", "sample_rate": 24000}},
    )
    providers = ["CUDAExecutionProvider"]
    provider_options = [{"device_id": "0"}]
    session_options = object()
    created: list[dict[str, Any]] = []

    class SessionStub:
        def __init__(self, model_path: Path, **kwargs: Any) -> None:
            created.append({"model_path": model_path, **kwargs})
            self.model_path = model_path
            self.kwargs = kwargs
            self.close_count = 0

        def diagnostics(self) -> SessionDiagnostic:
            providers = tuple(self.kwargs["providers"])
            return SessionDiagnostic(
                component=self.kwargs["component"],
                model_path=self.model_path,
                providers_requested=providers,
                providers_active=providers,
                inputs=(),
                outputs=(),
            )

        def close(self) -> None:
            self.close_count += 1

    monkeypatch.setattr(kitten2_module, "OnnxSession", SessionStub)
    adapter = Kitten2Adapter(
        installation,
        providers=providers,
        provider_options=provider_options,
        session_options=session_options,
    )
    assert created == []
    sessions = [
        adapter._get_session(component) for component in ("lm", "speaker", "flow", "mel2wav")
    ]
    assert len(created) == 4
    assert [entry["component"] for entry in created] == [
        "lm",
        "speaker",
        "flow",
        "mel2wav",
    ]

    for entry in created:
        assert entry["providers"] is providers
        assert entry["provider_options"] is provider_options
        assert entry["session_options"] is session_options
    diagnostic = adapter.diagnostics()
    assert diagnostic.system == "kitten2"
    assert [entry.component for entry in diagnostic.sessions] == [
        "lm",
        "speaker",
        "flow",
        "mel2wav",
    ]
    assert all(
        entry.providers_active == ("CUDAExecutionProvider",) for entry in diagnostic.sessions
    )
    with pytest.raises(CapabilityError):
        adapter.infer([1])
    adapter.close()
    adapter.close()
    assert [session.close_count for session in sessions] == [1, 1, 1, 1]
