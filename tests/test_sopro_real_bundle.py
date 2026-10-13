from __future__ import annotations

import json
import math
import os
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from onnxvoice.manager import OnnxVoice


def _json_fixture(name: str) -> dict[str, Any]:
    path = Path(__file__).parent / "fixtures" / "sopro" / name
    return json.loads(path.read_text(encoding="utf-8"))


def _reference_tone(reference_golden: dict[str, Any]) -> tuple[np.ndarray, int]:
    tone = reference_golden["tones"]["steady"]
    indices = np.arange(tone["samples"], dtype=np.float64)
    waveform = (
        0.08 * np.sin(2 * math.pi * 220 * indices / tone["sample_rate"])
        + 0.025 * np.sin(2 * math.pi * 791 * indices / tone["sample_rate"])
    ).astype(np.float32)
    return waveform, tone["sample_rate"]


def _assert_native_inference(
    runtime: Any,
    golden: dict[str, Any],
    reference_golden: dict[str, Any],
    installation_path: Path,
) -> tuple[Any, Any]:
    waveform, sample_rate = _reference_tone(reference_golden)
    voice = runtime.prepare_voice(waveform, sample_rate)
    assert (
        voice.semantic_tokens.tolist()
        == reference_golden["reference_graph"]["semantic_tokens"]["values"]
    )
    text = runtime.prepare_text(golden["text"], golden["language"])
    assert list(text.token_ids) == golden["text_ids"]
    result = runtime.infer(
        text.token_ids,
        voice_state=voice,
        language=golden["language"],
        temperature=0.0,
        top_p=1.0,
        top_k=0,
        max_seconds=golden["max_seconds"],
        min_seconds=golden["min_seconds"],
        seed=golden["seed"],
    )
    assert result.sample_rate == 24000
    assert result.audio.shape == (golden["audio_length"],)
    assert result.audio.dtype == np.float32
    assert result.audio.flags.c_contiguous
    assert np.isfinite(result.audio).all()
    np.testing.assert_allclose(
        [float(result.audio.min()), float(result.audio.max())],
        [golden["audio_min"], golden["audio_max"]],
        rtol=1.0e-5,
        atol=1.0e-5,
    )
    sample_positions = [0, 1, 2, 17, 255, 1024, 2048, 3839]
    np.testing.assert_allclose(
        result.audio[sample_positions], golden["audio_samples"], rtol=1.0e-5, atol=1.0e-5
    )
    assert result.metadata["profile"] == "wasm-uint8"
    assert result.metadata["semantic_tokens"] == len(golden["semantic_tokens"])
    assert result.metadata["steps"] == 2
    assert result.metadata["sample_rate"] == 24000
    json.dumps(result.metadata)

    diagnostics = runtime.diagnostics()
    expected_components = {
        "reference",
        "semantic_prefix",
        "semantic_core",
        "acoustic_condition",
        "acoustic_offline_2",
        "vocoder",
    }
    sessions = {session.component: session for session in diagnostics.sessions}
    assert set(sessions) == expected_components
    assert all(
        session.model_path.is_relative_to(installation_path) for session in sessions.values()
    )
    assert all(
        session.providers_active == ("CPUExecutionProvider",) for session in sessions.values()
    )
    return result, diagnostics


def test_pinned_native_bundle_managed_install_offline_reopen_and_infer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    if os.environ.get("ONNXVOICE_RUN_REAL_SOPRO_TESTS") != "1":
        pytest.skip("set ONNXVOICE_RUN_REAL_SOPRO_TESTS=1 to run the pinned native bundle gate")

    source_root = Path(
        os.environ.get(
            "ONNXVOICE_SOPRO_E2E_ROOT",
            str(Path.home() / ".cache" / "onnxvoice-sopro-parity"),
        )
    )
    required = [
        source_root / "manifest.json",
        source_root / "tokenizer.json",
        source_root / "model" / "common" / "reference.onnx",
        source_root / "model" / "common" / "weights-reference.bin",
        source_root / "model" / "common" / "semantic_prefix.onnx",
        source_root / "model" / "common" / "semantic_core.onnx",
        source_root / "model" / "common" / "weights-semantic.bin",
        source_root / "model" / "wasm-uint8" / "acoustic_condition.onnx",
        source_root / "model" / "wasm-uint8" / "acoustic_offline_2.onnx",
        source_root / "model" / "wasm-uint8" / "vocoder.onnx",
        source_root / "model" / "wasm-uint8" / "weights-condition.bin",
        source_root / "model" / "wasm-uint8" / "weights-acoustic.bin",
        source_root / "model" / "wasm-uint8" / "weights-vocoder.bin",
    ]
    missing = [path for path in required if not path.is_file()]
    if missing:
        pytest.fail(f"native Sopro gate cache is incomplete: {missing}")

    repository_root = Path(__file__).resolve().parents[1]
    catalog_value = os.environ.get("ONNXVOICE_SOPRO_CATALOG")
    catalog_path = (
        Path(catalog_value)
        if catalog_value
        else repository_root.parent / "sopro-onnx-bundles" / "catalog" / "bundles.json"
    )
    if not catalog_path.is_file():
        pytest.fail(f"pinned Sopro catalog is unavailable: {catalog_path}")

    golden = _json_fixture("offline_pipeline_golden.json")
    reference_golden = _json_fixture("reference_golden.json")
    pinned_manifest = json.loads((source_root / "manifest.json").read_text(encoding="utf-8"))
    assert pinned_manifest["revision"] == golden["model_revision"]
    assert golden["profile"] == "wasm-uint8"

    import onnxvoice.store as store_module

    original_download = store_module.AssetStore._download
    downloaded_from_network: list[str] = []

    def cache_first_download(self: Any, source: str, destination: Path, **kwargs: Any) -> None:
        artifact_path = Path(kwargs["artifact"])
        candidates = (source_root / artifact_path, source_root / "model" / artifact_path)
        local_source = next((path for path in candidates if path.is_file()), None)
        destination = Path(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if local_source is not None:
            shutil.copyfile(local_source, destination)
            return
        downloaded_from_network.append(str(artifact_path))
        original_download(self, source, destination, **kwargs)

    monkeypatch.setattr(store_module.AssetStore, "_download", cache_first_download)
    cache_dir = tmp_path / "managed-cache"
    catalog_sources = {"sopro": str(catalog_path)}
    manager = OnnxVoice(cache_dir=cache_dir, catalog_sources=catalog_sources)
    installation = manager.install("sopro:v2-turbo", quality="wasm-uint8")
    manager.store.verify(installation)
    assert installation.ref == "sopro:v2-turbo"
    assert installation.metadata["manifest_revision"] == golden["model_revision"]
    assert installation.artifact("graph", component="reference").filename == "common/reference.onnx"
    assert installation.artifact("graph", component="acoustic_offline_2").filename == (
        "wasm-uint8/acoustic_offline_2.onnx"
    )
    assert installation.artifact("shard", component="semantic").filename == (
        "common/weights-semantic.bin"
    )

    first_runtime = manager.open(installation, provider="cpu")
    second_runtime = None
    try:
        first_result, first_diagnostics = _assert_native_inference(
            first_runtime, golden, reference_golden, installation.path
        )
        assert first_diagnostics.ref == installation.ref
        first_runtime.close()
        first_runtime.close()

        def offline_download(*_args: Any, **_kwargs: Any) -> None:
            pytest.fail("offline reopen attempted to download a Sopro artifact")

        monkeypatch.setattr(store_module.AssetStore, "_download", offline_download)
        offline_manager = OnnxVoice(
            cache_dir=cache_dir,
            catalog_sources=catalog_sources,
            offline=True,
        )
        reopened = offline_manager.install("sopro:turbo", quality="wasm-uint8")
        offline_manager.store.verify(reopened)
        assert reopened.path == installation.path
        second_runtime = offline_manager.open(reopened, provider="cpu")
        second_result, second_diagnostics = _assert_native_inference(
            second_runtime, golden, reference_golden, reopened.path
        )
        assert second_diagnostics.ref == reopened.ref
        np.testing.assert_array_equal(second_result.audio, first_result.audio)
        second_runtime.close()
        second_runtime.close()
    finally:
        first_runtime.close()
        if second_runtime is not None:
            second_runtime.close()

    assert downloaded_from_network == [] or all(
        name.startswith("wasm-uint8/acoustic_stream") for name in downloaded_from_network
    )
