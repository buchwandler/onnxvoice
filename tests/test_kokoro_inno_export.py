from __future__ import annotations

import hashlib
import os
from pathlib import Path

import numpy as np
import pytest

from onnxvoice.systems.kokoro_inno import (
    InnoVoiceTuner,
    blend_weights,
    dense_blend_weights,
    head_stats,
    speaker_fbank,
    tilt,
)
from onnxvoice.types import Installation, InstalledArtifact


@pytest.mark.integration
@pytest.mark.resource_heavy
def test_exported_inno_graph_matches_upstream_voicepack(tmp_path, monkeypatch):
    checkpoint_value = os.environ.get("ONNXVOICE_INNO_CHECKPOINT")
    if not checkpoint_value or not Path(checkpoint_value).is_file():
        pytest.skip("set ONNXVOICE_INNO_CHECKPOINT to local upstream v0.2.0 model.safetensors")

    torch = pytest.importorskip("torch")
    pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")
    pytest.importorskip("scipy")
    pytest.importorskip("parselmouth")
    upstream = pytest.importorskip("inno_kokoro.enroll")
    upstream_prosody = pytest.importorskip("inno_kokoro.prosody")
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")

    from tools.export_kokoro_inno import export_inno_voicepack

    checkpoint = Path(checkpoint_value)
    graph_path = tmp_path / "inno_voicepack.onnx"
    metadata_path = tmp_path / "inno_tuner.npz"
    export_inno_voicepack(checkpoint, graph_path, metadata_path)
    assert graph_path.is_file()
    assert metadata_path.is_file()

    base_path = tmp_path / "base.onnx"
    base_path.write_bytes(b"test base model")

    def artifact(role: str, path: Path, component: str | None = None) -> InstalledArtifact:
        content = path.read_bytes()
        return InstalledArtifact(
            role=role,
            filename=path.name,
            path=path,
            sha256=hashlib.sha256(content).hexdigest(),
            size=len(content),
            component=component,
        )

    installation = Installation(
        system="kokoro",
        id="inno-export-parity",
        kind="model",
        path=tmp_path,
        artifacts=(
            artifact("model", base_path),
            artifact("model", graph_path, "inno_voicepack"),
            artifact("metadata", metadata_path, "inno_tuner"),
        ),
        metadata={
            "runtime": {
                "layout": "single-onnx-v1",
                "voice_enrollers": [
                    {
                        "id": "inno-v0.2",
                        "kind": "kokoro-voicepack-tuner",
                        "input": "reference-audio",
                        "transcript_required": False,
                        "min_seconds": 3.0,
                        "recommended_seconds": 5.0,
                        "max_seconds": 30.0,
                        "output": {
                            "format": "kokoro-voicepack-v1",
                            "shape": [510, 1, 256],
                            "dtype": "float32",
                        },
                        "model_component": "inno_voicepack",
                        "metadata_component": "inno_tuner",
                    }
                ],
            }
        },
    )
    runtime = InnoVoiceTuner(installation, providers=["CPUExecutionProvider"])

    fixture_path = Path(__file__).parent / "fixtures" / "inno_fbank.npz"
    with np.load(fixture_path, allow_pickle=False) as fixture:
        waveform = fixture["prosody_waveform"]
        sample_rate = int(fixture["sample_rate"])
    waveform_tensor = torch.from_numpy(waveform)
    upstream_tuner = upstream.Tuner(str(checkpoint), device="cpu")
    expected_pack, expected_weights = upstream.enroll(waveform_tensor, sample_rate, upstream_tuner)

    fmax = upstream_prosody.ceiling(waveform_tensor, sample_rate)
    target = np.asarray(
        [
            *upstream_prosody.stats(waveform_tensor, sample_rate, fmax)[:2],
            upstream_prosody.rate(waveform_tensor, sample_rate),
        ],
        dtype=np.float64,
    )
    upstream_weights = upstream.blend_weights(upstream_tuner, target)
    assert upstream_weights == expected_weights
    host_weights = blend_weights(runtime, target)
    assert host_weights == upstream_weights

    voicepack = runtime.infer_graph(
        speaker_fbank(waveform),
        (tilt(waveform, sample_rate) - runtime.tilt_mean) / runtime.tilt_sd,
        np.asarray([head_stats(waveform, sample_rate, fmax=fmax)], dtype=np.float32),
        dense_blend_weights(runtime.names, host_weights),
    )
    difference = np.abs(voicepack - expected_pack.detach().cpu().numpy())
    assert difference.shape == (510, 1, 256)
    assert float(difference.max()) < 2.0e-6
    enrolled = runtime.enroll(waveform, sample_rate=sample_rate)
    assert enrolled.fingerprint
    assert enrolled.data.shape == (510, 1, 256)
    assert float(np.abs(enrolled.data - expected_pack.detach().cpu().numpy()).max()) < 2.0e-6
    runtime.close()
