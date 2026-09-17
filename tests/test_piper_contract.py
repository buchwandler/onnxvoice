from __future__ import annotations

import json

import numpy as np
import pytest

from onnxvoice.errors import RuntimeContractError
from onnxvoice.systems.piper import PiperAdapter
from onnxvoice.types import Installation, InstalledArtifact, TensorSpec


class PiperSession:
    input_names = ("input", "input_lengths", "scales", "sid")
    input_specs = (
        TensorSpec("input", "tensor(int32)", (1, "N")),
        TensorSpec("input_lengths", "tensor(int64)", (1,)),
        TensorSpec("scales", "tensor(float16)", (3,)),
        TensorSpec("sid", "tensor(int64)", (1,)),
    )

    def __init__(self):
        self.seen = None

    def run(self, inputs):
        self.seen = inputs
        return [np.array([[0.1, 0.2]], dtype=np.float16)]

    def close(self):
        pass


def test_piper_validates_speakers_and_graph_dtypes(tmp_path):
    model = tmp_path / "voice.onnx"
    model.write_bytes(b"model")
    config = tmp_path / "voice.json"
    config.write_text(
        json.dumps(
            {
                "num_speakers": 2,
                "speaker_id_map": {"alice": 0, "bob": 1},
                "audio": {"sample_rate": 16000},
            }
        )
    )
    installation = Installation(
        "piper",
        "voice",
        "voice",
        tmp_path,
        (
            InstalledArtifact("model", model.name, model, "0" * 64, model.stat().st_size),
            InstalledArtifact("config", config.name, config, "1" * 64, config.stat().st_size),
        ),
    )
    adapter = PiperAdapter(installation)
    fake = PiperSession()
    adapter._session = fake  # type: ignore[assignment]

    result = adapter.infer([1, 2], speaker="bob")

    assert result.sample_rate == 16000
    assert result.audio.dtype == np.float32
    assert fake.seen["input"].dtype == np.int32
    assert fake.seen["scales"].dtype == np.float16
    assert fake.seen["sid"].item() == 1
    assert adapter._config() is adapter._config()

    with pytest.raises(RuntimeContractError, match="Unknown Piper speaker"):
        adapter.infer([1], speaker="missing")
    with pytest.raises(RuntimeContractError, match="must be provided"):
        adapter.infer([1])
