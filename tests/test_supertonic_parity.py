from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest

from onnxvoice import OnnxVoice

PINNED_REVISION = "aafc6e32416a594460b32413efc49d7fe4ce6d46"
WAVEFORM_RTOL = 1e-4
WAVEFORM_ATOL = 1e-5
MODEL_COMPONENTS = ("duration_predictor", "text_encoder", "vector_estimator", "vocoder")


@pytest.mark.integration
@pytest.mark.resource_heavy
def test_supertonic3_matches_archived_supertonic_py() -> None:
    """Compare the pinned bundle to archived supertonic-py on CPU.

    Set ONNXVOICE_SUPERTONIC3_BUNDLE to a local copy of the model snapshot and
    ONNXVOICE_SUPERTONIC3_REVISION to its revision. Waveforms are compared with
    rtol=1e-4 and atol=1e-5 to allow minor CPU execution differences.
    """
    bundle_value = os.environ.get("ONNXVOICE_SUPERTONIC3_BUNDLE")
    if not bundle_value:
        pytest.skip("set ONNXVOICE_SUPERTONIC3_BUNDLE to run pinned model parity")
    if os.environ.get("ONNXVOICE_SUPERTONIC3_REVISION") != PINNED_REVISION:
        pytest.skip(f"parity requires model revision {PINNED_REVISION}")

    root = Path(bundle_value)
    model_paths = {name: root / "onnx" / f"{name}.onnx" for name in MODEL_COMPONENTS}
    style_path = root / "voice_styles" / "M1.json"
    config_path = root / "onnx" / "tts.json"
    indexer_path = root / "onnx" / "unicode_indexer.json"
    required = [*model_paths.values(), style_path, config_path, indexer_path]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        pytest.fail("pinned Supertonic-3 bundle is incomplete: " + ", ".join(missing))

    ort = pytest.importorskip("onnxruntime")
    upstream = pytest.importorskip("supertonic.core")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    with style_path.open(encoding="utf-8") as stream:
        style_data = json.load(stream)
    style_ttl = np.asarray(style_data["style_ttl"]["data"], dtype=np.float32).reshape(
        style_data["style_ttl"]["dims"]
    )
    style_dp = np.asarray(style_data["style_dp"]["data"], dtype=np.float32).reshape(
        style_data["style_dp"]["dims"]
    )
    processor = upstream.UnicodeProcessor(str(indexer_path))
    text = "Supertonic local runtime parity check."
    token_ids, text_mask = processor([text], lang="en")
    steps = 5
    speed = 1.05
    seed = 1234
    style = upstream.Style(style_ttl, style_dp)
    sessions = [
        ort.InferenceSession(str(model_paths[component]), providers=["CPUExecutionProvider"])
        for component in MODEL_COMPONENTS
    ]
    reference_runtime = upstream.Supertonic(config, processor, *sessions)

    np.random.seed(seed)
    reference_audio, reference_duration = reference_runtime(
        [text], style, total_step=steps, speed=speed, lang="en"
    )
    runtime = OnnxVoice.open_local(
        system="supertonic",
        files={
            "config": config_path,
            "unicode_indexer": indexer_path,
            **model_paths,
            "voice_style:M1": style_path,
        },
        sample_rate=int(config["ae"]["sample_rate"]),
        runtime={"layout": "supertonic-3-v1"},
        providers="CPUExecutionProvider",
    )
    try:
        result = runtime.infer(
            token_ids[0].tolist(),
            text_mask=text_mask,
            style_ttl=style_ttl,
            style_dp=style_dp,
            steps=steps,
            speed=speed,
            seed=seed,
        )
        expected = np.asarray(reference_audio, dtype=np.float32).squeeze()
        assert config["ae"]["sample_rate"] == 44100
        assert result.sample_rate == 44100
        np.testing.assert_allclose(result.timings, reference_duration, rtol=0, atol=1e-7)
        np.testing.assert_allclose(result.audio, expected, rtol=WAVEFORM_RTOL, atol=WAVEFORM_ATOL)
    finally:
        runtime.close()
