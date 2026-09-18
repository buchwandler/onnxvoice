from __future__ import annotations

import json

import numpy as np
import pytest

from onnxvoice.errors import CapabilityError
from onnxvoice.systems.kokoro import KokoroAdapter
from onnxvoice.types import Installation, InstalledArtifact


class SplitSession:
    def __init__(self, component: str) -> None:
        self.component = component
        self.input_names = {
            "prosody": ("input_ids", "style_dur", "speed"),
            "curves": ("en", "style_dur"),
            "decoder": ("asr", "f0_curve", "n_curve", "style_acou", "har"),
        }[component]
        self.output_names = {
            "prosody": ("pred_dur", "d", "t_en"),
            "curves": ("f0_curve", "n_curve"),
            "decoder": ("audio",),
        }[component]
        self.seen: dict[str, np.ndarray] | None = None
        self.closed = False

    def run(self, inputs: dict[str, np.ndarray]) -> list[np.ndarray]:
        self.seen = inputs
        if self.component == "prosody":
            return [
                np.array([1, 1], dtype=np.int64),
                np.ones((1, 2, 2), dtype=np.float32),
                np.ones((1, 3, 2), dtype=np.float32),
            ]
        if self.component == "curves":
            return [
                np.full((1, 2), 100.0, dtype=np.float32),
                np.zeros((1, 2), dtype=np.float32),
            ]
        return [np.zeros((1, 32), dtype=np.float64)]

    def close(self) -> None:
        self.closed = True


def _installation(tmp_path, *, include_source: bool = True) -> Installation:
    paths: dict[str, tuple[str, str, str | None]] = {}
    for component in ("prosody", "curves", "decoder"):
        path = tmp_path / f"{component}.onnx"
        path.write_bytes(b"model")
        paths[component] = ("model", path.name, component)
    config = tmp_path / "manifest.json"
    config.write_text(json.dumps({"max_tokens": 20}), encoding="utf-8")
    voices = tmp_path / "voices.npz"
    np.savez(voices, voice=np.zeros((1, 4), dtype=np.float32))
    paths["config"] = ("config", config.name, None)
    paths["voices"] = ("voices", voices.name, None)
    if include_source:
        source = tmp_path / "source-params.npz"
        np.savez(
            source,
            weight=np.ones((1, 9), dtype=np.float32),
            bias=np.zeros(1, dtype=np.float32),
            window=np.ones(20, dtype=np.float32),
        )
        paths["source_params"] = ("source_params", source.name, None)
    artifacts = tuple(
        InstalledArtifact(
            role=role,
            filename=filename,
            path=tmp_path / filename,
            sha256="0" * 64,
            size=(tmp_path / filename).stat().st_size,
            component=component,
        )
        for role, filename, component in paths.values()
    )
    return Installation(
        "kokoro",
        "thai",
        "model",
        tmp_path,
        artifacts,
        sample_rate=24000,
        metadata={
            "runtime": {
                "layout": "split-onnx-v1",
                "style_dimensions": {"acoustic": 2, "duration": 2},
            }
        },
    )


def test_split_runtime_executes_graph_and_is_seeded(tmp_path) -> None:
    sessions: dict[str, SplitSession] = {}

    def factory(_path, component):
        sessions[component] = SplitSession(component)
        return sessions[component]

    adapter = KokoroAdapter(_installation(tmp_path), session_factory=factory)
    first = adapter.infer([1, 2], style=np.arange(4), speed=1.0, seed=7)
    second = adapter.infer([1, 2], style=np.arange(4), speed=1.0, seed=7)

    np.testing.assert_array_equal(first.audio, second.audio)
    np.testing.assert_array_equal(first.timings, np.array([1, 1]))
    assert first.audio.dtype == np.float32
    assert first.audio.ndim == 1
    np.testing.assert_array_equal(
        sessions["prosody"].seen["input_ids"], np.array([[0, 1, 2, 0]], dtype=np.int64)
    )
    assert set(first.outputs) == {"pred_dur", "f0_curve", "n_curve"}
    adapter.close()
    adapter.close()
    assert all(session.closed for session in sessions.values())


def test_split_runtime_requires_source_parameters(tmp_path) -> None:
    with pytest.raises(CapabilityError, match="source parameters"):
        KokoroAdapter(_installation(tmp_path, include_source=False)).infer(
            [1], style=np.zeros(4)
        )
