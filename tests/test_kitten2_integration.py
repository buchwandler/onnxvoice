from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from _kitten2_fixture import create_bundle

import onnxvoice.manager as manager_module
from onnxvoice import Kitten2Adapter, OnnxVoice, registered_systems
from onnxvoice.errors import RuntimeContractError
from onnxvoice.systems import get_adapter
from onnxvoice.systems.kitten2_frontend import PreparedTarget, PreparedVoice
from onnxvoice.systems.kitten2_generation import GenerationResult


def _catalog_for_bundle(bundle_path: Path) -> dict[str, Any]:
    bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
    artifacts = [
        {
            "role": "bundle",
            "path": "bundle.json",
            "sha256": hashlib.sha256(bundle_path.read_bytes()).hexdigest(),
        }
    ]
    artifacts.extend(
        {"role": "asset", "path": path, "sha256": digest}
        for path, digest in sorted(bundle["files"].items())
    )
    return {
        "schema": 1,
        "kind": "kitten2-onnx-bundle-catalog",
        "bundles": {
            "kitten-tts-v2": {
                "id": "kitten-tts-v2",
                "aliases": ["v2"],
                "name": "KittenTTS 2 ONNX",
                "layout": "kitten2-onnx-bundle-v1",
                "sample_rate": 24000,
                "default_voice": "Bruno",
                "voices": ["Bruno"],
                "source": {
                    "provider": "huggingface",
                    "repository": "example/kitten2-bundle",
                    "revision": "a" * 40,
                    "license": "Apache-2.0",
                },
                "runtime": {"bundle_format": "kitten2-onnx-bundle-v1"},
                "metadata": {"default_preset": "stable"},
                "artifacts": artifacts,
            }
        },
    }


def _forbidden(*_args: Any, **_kwargs: Any) -> None:
    pytest.fail("local Kitten2 opening must not access the catalog or asset store")


def test_kitten2_is_registered_and_exported() -> None:
    assert "kitten2" in registered_systems()
    assert Kitten2Adapter.system == "kitten2"
    assert get_adapter("kitten2") is Kitten2Adapter


def test_local_open_uses_native_adapter_without_creating_sessions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundle_path = create_bundle(tmp_path / "bundle")
    monkeypatch.setattr(manager_module.AssetStore, "__init__", _forbidden)
    monkeypatch.setattr(manager_module.CatalogClient, "__init__", _forbidden)

    runtime = OnnxVoice.open_local(system="kitten2", files={"bundle": bundle_path})
    try:
        assert isinstance(runtime, Kitten2Adapter)
        assert runtime.voices == ("Bruno",)
        assert runtime.default_voice == "Bruno"
        assert runtime.default_preset == "stable"
        assert runtime.sample_rate == 24000
        assert set(runtime.decode_presets) == {"stable", "expressive"}
        with pytest.raises(TypeError):
            runtime.decode_presets["stable"]["top_k"] = 1
        assert runtime._sessions == {}
        assert runtime.diagnostics().sessions == ()
        with pytest.raises(ValueError, match="Unknown decode preset"):
            runtime.infer_text("Hello", preset="missing")
        with pytest.raises(ValueError, match="Unknown voice"):
            runtime.infer_text("Hello", voice="not-a-voice")
        assert runtime._sessions == {}
    finally:
        runtime.close()


def test_single_file_import_rejects_kitten2(tmp_path: Path) -> None:
    manager = OnnxVoice(cache_dir=tmp_path / "cache")
    model = tmp_path / "lm.onnx"
    model.write_bytes(b"not a bundle")
    with pytest.raises(ValueError, match="complete bundle"):
        manager.import_model(system="kitten2", item_id="bundle", model=model)


def test_managed_install_preserves_nested_bundle_files_and_opens_natively(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundle_path = create_bundle(tmp_path / "upstream")
    catalog_path = tmp_path / "catalog.json"
    catalog_path.write_text(json.dumps(_catalog_for_bundle(bundle_path)), encoding="utf-8")
    manager = OnnxVoice(
        cache_dir=tmp_path / "cache",
        catalog_sources={"kitten2": str(catalog_path)},
    )

    def materialize(
        _store: Any, artifact: Any, *, ref: str, progress: Any, target: str
    ) -> tuple[Path, str]:
        del ref, progress, target
        return bundle_path.parent / artifact.filename, artifact.sha256

    monkeypatch.setattr(manager_module.AssetStore, "_materialize_artifact", materialize)
    installation = manager.install("kitten2:kitten-tts-v2")
    installed_names = {artifact.filename for artifact in installation.artifacts}
    assert "decoder/mel2wav/mel2wav.onnx.data" in installed_names
    assert "decoder/mel2wav/mel2wav.onnx" in installed_names
    graph = installation.path / "decoder/mel2wav/mel2wav.onnx"
    external_data = installation.path / "decoder/mel2wav/mel2wav.onnx.data"
    assert graph.parent == external_data.parent
    assert graph.is_file() and external_data.is_file()

    runtime = manager.open(installation)
    try:
        assert isinstance(runtime, Kitten2Adapter)
        assert runtime.bundle.root == installation.path
        assert runtime._sessions == {}
    finally:
        runtime.close()


def test_infer_text_returns_finite_mono_float32_at_bundle_sample_rate(
    tmp_path: Path,
) -> None:
    bundle_path = create_bundle(tmp_path / "bundle")
    runtime = OnnxVoice.open_local(system="kitten2", files={"bundle": bundle_path})

    class FrontendStub:
        def prepare_voice(self, voice: str) -> PreparedVoice:
            assert voice == "Bruno"
            return PreparedVoice(
                speaker_embedding=np.zeros((1, 4), dtype=np.float32),
                reference_prefix=[11, 12],
                conditioning={},
            )

        def prepare_target(self, text: str) -> PreparedTarget:
            return PreparedTarget(text=text, token_ids=(3, 4), has_expression=False)

    class LanguageModelStub:
        def generate(self, prompt: list[int], **kwargs: Any) -> GenerationResult:
            assert prompt
            assert isinstance(kwargs["rng"], np.random.Generator)
            assert kwargs["max_new_tokens"] == 12
            return GenerationResult(raw_ids=[100, 2], s3_ids=[0])

    class DecoderStub:
        def decode(
            self, s3_ids: list[int], conditioning: dict[str, np.ndarray], *, seed: int | None
        ) -> np.ndarray:
            assert s3_ids == [0]
            assert conditioning == {}
            assert seed == 23
            return np.array([[0.0, 0.25, -0.25]], dtype=np.float64)

    runtime.frontend = FrontendStub()
    runtime._get_language_model = lambda: LanguageModelStub()
    runtime._get_decoder = lambda: DecoderStub()
    try:
        result = runtime.infer_text("Hello", seed=23, max_new_tokens=12)
        assert result.sample_rate == 24000
        assert result.audio.dtype == np.float32
        assert result.audio.ndim == 1
        assert np.all(np.isfinite(result.audio))
        assert result.metadata["voice"] == "Bruno"
        assert result.metadata["preset"] == "stable"
        assert result.metadata["expression_conditioning"] is False
        assert result.metadata["chunk_count"] == 1

        class NonFiniteDecoder:
            def decode(self, *_args: Any, **_kwargs: Any) -> np.ndarray:
                return np.array([np.nan], dtype=np.float32)

        runtime._get_decoder = lambda: NonFiniteDecoder()
        with pytest.raises(RuntimeContractError, match="NaN or infinite"):
            runtime.infer_text("Hello", seed=23, max_new_tokens=12)
    finally:
        runtime.close()
