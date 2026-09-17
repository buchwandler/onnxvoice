from __future__ import annotations

import json

import numpy as np

from onnxvoice.systems.kokoro import KokoroAdapter
from onnxvoice.systems.piper import PiperAdapter
from onnxvoice.types import Installation, InstalledArtifact


class FakeSession:
    def __init__(self, names, output):
        self.input_names = tuple(names)
        self.output = output
        self.seen = None

    def run(self, inputs):
        self.seen = inputs
        return [self.output]

    def close(self):
        pass


def artifact(tmp_path, role, filename):
    path = tmp_path / filename
    if not path.exists():
        path.write_bytes(b"x")
    return InstalledArtifact(role, filename, path, "0" * 64, path.stat().st_size)


def test_piper_adapter_builds_runtime_inputs(tmp_path):
    config = tmp_path / "voice.onnx.json"
    config.write_text(json.dumps({"audio": {"sample_rate": 22050}}), encoding="utf-8")
    model = tmp_path / "voice.onnx"
    model.write_bytes(b"x")
    installation = Installation(
        "piper",
        "voice",
        "voice",
        tmp_path,
        (
            InstalledArtifact("model", model.name, model, "0" * 64, 1),
            InstalledArtifact("config", config.name, config, "0" * 64, config.stat().st_size),
        ),
    )
    adapter = PiperAdapter(installation)
    fake = FakeSession(
        ("input", "input_lengths", "scales"), np.array([[[0.1, 0.2]]], dtype=np.float32)
    )
    adapter._session = fake  # type: ignore[assignment]
    result = adapter.infer([1, 2, 3])
    assert result.sample_rate == 22050
    assert fake.seen["input"].dtype == np.int64
    assert fake.seen["scales"].dtype == np.float32


def test_kokoro_adapter_selects_voice_row_by_token_length(tmp_path):
    model = tmp_path / "kokoro.onnx"
    model.write_bytes(b"x")
    voices = tmp_path / "voices.npz"
    np.savez(voices, af_heart=np.arange(4 * 3, dtype=np.float32).reshape(4, 3))
    installation = Installation(
        "kokoro",
        "v1",
        "model",
        tmp_path,
        (
            InstalledArtifact("model", model.name, model, "0" * 64, 1, "fp32"),
            InstalledArtifact("voices", voices.name, voices, "0" * 64, voices.stat().st_size),
        ),
        sample_rate=24000,
        voices=("af_heart",),
        default_voice="af_heart",
    )
    adapter = KokoroAdapter(installation)
    fake = FakeSession(("input_ids", "style", "speed"), np.array([[0.1, 0.2]], dtype=np.float32))
    adapter._session = fake  # type: ignore[assignment]
    result = adapter.infer([5, 6])
    assert result.sample_rate == 24000
    np.testing.assert_array_equal(fake.seen["input_ids"], np.array([[0, 5, 6, 0]]))
    np.testing.assert_array_equal(fake.seen["style"], np.array([[3, 4, 5]], dtype=np.float32))
