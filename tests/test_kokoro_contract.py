from __future__ import annotations

import numpy as np
import pytest

from onnxvoice.errors import CapabilityError, RuntimeContractError
from onnxvoice.systems.kokoro import KokoroAdapter
from onnxvoice.types import Installation, InstalledArtifact, TensorSpec


class ContractSession:
    input_names = ("input_ids", "ref_s", "speed")
    input_specs = (
        TensorSpec("input_ids", "tensor(int32)", (1, 6)),
        TensorSpec("ref_s", "tensor(float16)", (1, 3)),
        TensorSpec("speed", "tensor(float)", (1,)),
    )
    output_names = ("audio", "durations")

    def __init__(self):
        self.seen = None

    def run(self, inputs):
        self.seen = inputs
        return [np.array([[0.1, 0.2]], dtype=np.float16), np.array([1, 2], dtype=np.int32)]

    def close(self):
        pass


def _installation(tmp_path, *, split=False):
    model = tmp_path / "model.onnx"
    model.write_bytes(b"model")
    voices = tmp_path / "voices.npz"
    np.savez(voices, af_heart=np.arange(4 * 3, dtype=np.float32).reshape(4, 3))
    artifacts = [InstalledArtifact("model", model.name, model, "0" * 64, model.stat().st_size)]
    if split:
        second = tmp_path / "decoder.onnx"
        second.write_bytes(b"decoder")
        artifacts.append(InstalledArtifact("model", second.name, second, "1" * 64, second.stat().st_size))
    artifacts.append(InstalledArtifact("voices", voices.name, voices, "2" * 64, voices.stat().st_size))
    return Installation(
        "kokoro",
        "v1",
        "model",
        tmp_path,
        tuple(artifacts),
        sample_rate=24000,
        voices=("af_heart",),
        default_voice="af_heart",
    )


def test_kokoro_negotiates_dtypes_and_named_timing_outputs(tmp_path):
    adapter = KokoroAdapter(_installation(tmp_path))
    fake = ContractSession()
    adapter._session = fake  # type: ignore[assignment]

    result = adapter.infer([5, 6], voice="af_heart")

    assert fake.seen["input_ids"].dtype == np.int32
    assert fake.seen["ref_s"].dtype == np.float16
    assert result.audio.dtype == np.float32
    np.testing.assert_array_equal(result.timings, np.array([1, 2], dtype=np.int32))
    assert set(result.outputs) == {"durations"}


def test_kokoro_enforces_graph_token_limit(tmp_path):
    adapter = KokoroAdapter(_installation(tmp_path))
    fake = ContractSession()
    adapter._session = fake  # type: ignore[assignment]

    with pytest.raises(RuntimeContractError, match="maximum"):
        adapter.infer([1, 2, 3, 4, 5], voice="af_heart")


def test_kokoro_rejects_split_layout(tmp_path):
    adapter = KokoroAdapter(_installation(tmp_path, split=True))

    with pytest.raises(CapabilityError, match="Split"):
        _ = adapter.session
