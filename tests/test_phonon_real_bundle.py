from __future__ import annotations

import importlib.util
import os
from collections.abc import Mapping
from pathlib import Path
from typing import NoReturn

import numpy as np
import pytest

from onnxvoice import OnnxVoice
from onnxvoice.types import CatalogItem

pytestmark = [pytest.mark.integration, pytest.mark.resource_heavy]

_RUN_ACCEPTANCE = os.environ.get("ONNXVOICE_PHONON_ACCEPTANCE") == "1"
_REQUIRE_ACCEPTANCE = os.environ.get("ONNXVOICE_REQUIRE_PHONON_ACCEPTANCE") == "1"


def _unavailable(message: str, *, required: bool = False) -> NoReturn:
    if required and _REQUIRE_ACCEPTANCE:
        pytest.fail(message)
    pytest.skip(message)


@pytest.mark.parametrize(
    ("model_key", "expected_license", "required"),
    [
        ("phonon-31m-en", "Apache-2.0", True),
        ("phonon-40m-en", "CC-BY-NC-4.0", False),
    ],
)
def test_phonon_real_bundle_managed_install_inference_and_offline_reopen(
    tmp_path: Path, model_key: str, expected_license: str, required: bool
) -> None:
    """Exercise only published fp32 targets when explicitly opted in."""
    if not _RUN_ACCEPTANCE:
        pytest.skip("set ONNXVOICE_PHONON_ACCEPTANCE=1 to download and test real Phonon bundles")
    if importlib.util.find_spec("onnxruntime") is None:
        _unavailable(
            "onnxruntime is required for the Phonon real-bundle acceptance check",
            required=required,
        )

    cache_dir = tmp_path / model_key
    manager = OnnxVoice(cache_dir=cache_dir)
    try:
        items = manager.list("phonon")
    except Exception as exc:
        _unavailable(f"could not load the published Phonon catalog: {exc}", required=required)

    item = next(
        (
            candidate
            for candidate in items
            if isinstance(candidate, CatalogItem) and candidate.id == model_key
        ),
        None,
    )
    if item is None:
        _unavailable(
            f"{model_key} is not currently published in the Phonon catalog", required=required
        )
    quality_profiles = item.metadata.get("quality_profiles", ())
    if "fp32" not in quality_profiles:
        _unavailable(f"{model_key} has no published fp32 bundle", required=required)

    installation = manager.install(f"phonon:{model_key}", quality="fp32")
    license_info = installation.metadata.get("license")
    assert isinstance(license_info, Mapping)
    assert license_info["spdx"] == expected_license
    assert installation.require_artifact("config").path.is_file()
    assert installation.require_artifact("attribution").path.is_file()
    assert {
        installation.require_artifact("model", component=component).path.name
        for component in ("text_conditioner", "flow_lm_main", "flow_lm_flow", "mimi_decoder")
    } == {
        "text_conditioner.onnx",
        "flow_lm_main.onnx",
        "flow_lm_flow.onnx",
        "mimi_decoder.onnx",
    }
    if expected_license == "CC-BY-NC-4.0":
        assert license_info["commercial_use"] is False
        assert installation.require_artifact("notice", component="model").path.is_file()
        assert installation.require_artifact("notice", component="voices").path.is_file()

    token_ids = (1, 2, 3, 4)
    generation = {
        "voice": installation.default_voice,
        "duration_delta": 0.0,
        "temperature": 0.8,
        "seed": 17,
        "eos_threshold": 4.0,
        "frames_after_eos": 1,
        "min_frames_before_eos": 2,
        "max_frames": 3,
    }
    with manager.open(installation, providers="cpu") as runtime:
        chunks = list(runtime.stream(token_ids, **generation))
        assert chunks
        assert len(chunks) <= generation["max_frames"]
        assert all(chunk.audio.dtype == np.float32 for chunk in chunks)
        assert all(chunk.audio.ndim == 1 and np.isfinite(chunk.audio).all() for chunk in chunks)
        assert all(chunk.sample_rate == installation.sample_rate for chunk in chunks)

    offline_manager = OnnxVoice(cache_dir=cache_dir, offline=True)
    with offline_manager.open(f"phonon:{model_key}", quality="fp32", providers="cpu") as runtime:
        result = runtime.infer(token_ids, **generation)
        assert result.sample_rate == installation.sample_rate
        assert result.audio.dtype == np.float32
        assert result.audio.ndim == 1 and result.audio.size > 0
        assert np.isfinite(result.audio).all()
        frame_samples = installation.metadata["runtime"]["frame_samples"]
        assert result.audio.size <= frame_samples * generation["max_frames"]
        assert result.metadata["quality"] == "fp32"
