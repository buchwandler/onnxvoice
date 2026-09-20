"""Optional smoke test for one real pinned Pocket bundle.

Set ONNXVOICE_POCKET_BUNDLE_DIR to a checked-out bundle directory containing
bundle.json and the artifact files to enable this test. It is intentionally
skipped in environments without the external model fixture.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest

from onnxvoice import OnnxVoice


def test_real_pinned_pocket_bundle_round_trip() -> None:
    root_value = os.environ.get("ONNXVOICE_POCKET_BUNDLE_DIR")
    if not root_value:
        pytest.skip("ONNXVOICE_POCKET_BUNDLE_DIR is not configured")
    root = Path(root_value)
    bundle_path = root / "bundle.json"
    if not bundle_path.is_file():
        pytest.skip(f"Pocket bundle metadata is missing: {bundle_path}")
    try:
        metadata = json.loads(bundle_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        pytest.fail(f"Unable to read real Pocket bundle metadata: {exc}")
    if not isinstance(metadata, dict):
        pytest.fail("Pocket bundle metadata must be an object")

    files: dict[str, Path] = {"bundle_metadata": bundle_path}
    for role, info in (metadata.get("artifacts") or {}).items():
        filename = info.get("filename") if isinstance(info, dict) else info
        if isinstance(filename, str):
            path = root / filename
            if path.is_file():
                files[role] = path
    required = {
        "bundle_metadata",
        "bos_conditioning",
        "mimi_encoder",
        "text_conditioner",
        "flow_lm_main",
        "flow_lm_flow",
        "mimi_decoder",
    }
    missing = sorted(required - files.keys())
    if missing:
        pytest.skip(f"Real Pocket bundle is incomplete; missing {', '.join(missing)}")

    runtime = OnnxVoice.open_local(
        system="pocket",
        files=files,
        metadata=metadata,
        sample_rate=int(metadata.get("sample_rate", 24000)),
        providers="CPUExecutionProvider",
    )
    try:
        voice = runtime.prepare_voice(np.zeros(24000, dtype=np.float32), sample_rate=24000)
        result = runtime.infer(
            [1, 2, 3], voice_state=voice, temperature=0.7, lsd_steps=1, max_frames=2
        )
        assert voice.embeddings.ndim == 3
        assert result.audio.ndim == 1
        assert result.audio.size > 0
        assert np.isfinite(result.audio).all()
        assert np.any(result.audio != 0)
        assert result.sample_rate == 24000
    finally:
        runtime.close()
