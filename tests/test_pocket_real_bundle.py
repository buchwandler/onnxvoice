"""Opt-in repeated-run diagnostics for a real pinned Pocket bundle.

Set ONNXVOICE_POCKET_BUNDLE_DIR to a checked-out bundle directory containing
bundle.json and its artifacts to enable three low-level smoke attempts. Set
ONNXVOICE_POCKET_DIAGNOSTICS_PATH to persist the generation and duration report.
The test uses token IDs and reference audio; it is not a text-semantic regression.
"""

from __future__ import annotations

import json
import os
import time
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
        assert voice.embeddings is not None
        assert voice.embeddings.ndim == 3

        attempts: list[dict[str, float | int | bool | None]] = []
        for _ in range(3):
            started = time.perf_counter()
            result = runtime.infer(
                [1, 2, 3],
                voice_state=voice,
                temperature=0.7,
                lsd_steps=1,
                max_frames=64,
                frames_after_eos=8,
            )
            duration_seconds = time.perf_counter() - started
            assert result.audio.ndim == 1
            assert result.audio.size > 0
            assert np.isfinite(result.audio).all()
            assert np.any(result.audio != 0)
            assert result.sample_rate == 24000

            frames_generated = result.metadata["frames_generated"]
            eos_step = result.metadata["eos_step"]
            assert isinstance(frames_generated, int) and frames_generated > 0
            assert eos_step is None or eos_step >= 6
            assert result.metadata["eos_detected"] is (eos_step is not None)
            attempts.append(
                {
                    "frames_generated": frames_generated,
                    "eos_step": eos_step,
                    "duration_seconds": duration_seconds,
                    "eos_observed": result.metadata["eos_observed"],
                }
            )

        report = {
            "bundle": metadata.get("bundle_name", root.name),
            "voice": "reference_audio_smoke",
            "token_ids": [1, 2, 3],
            "attempts": attempts,
        }
        rendered_report = json.dumps(report, indent=2, sort_keys=True)
        print(f"Pocket real-bundle diagnostics:\n{rendered_report}")
        report_path_value = os.environ.get("ONNXVOICE_POCKET_DIAGNOSTICS_PATH")
        if report_path_value:
            report_path = Path(report_path_value)
            report_path.parent.mkdir(parents=True, exist_ok=True)
            report_path.write_text(rendered_report + "\n", encoding="utf-8")
    finally:
        runtime.close()
