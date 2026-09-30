from __future__ import annotations

from pathlib import Path

import pytest

from onnxvoice import OnnxVoice, registered_systems
from onnxvoice.systems import SupertonicAdapter

MODEL_COMPONENTS = (
    "duration_predictor",
    "text_encoder",
    "vector_estimator",
    "vocoder",
)


def _bundle_files(root: Path) -> dict[str, Path]:
    files = {
        "config": "onnx/tts.json",
        "unicode_indexer": "onnx/unicode_indexer.json",
        **{component: f"onnx/{component}.onnx" for component in MODEL_COMPONENTS},
        **{
            f"voice_style:{name}": f"voice_styles/{name}.json"
            for name in ("F1", "F2", "F3", "F4", "F5", "M1", "M2", "M3", "M4", "M5")
        },
    }
    result = {}
    for key, relative_path in files.items():
        path = root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}", encoding="utf-8")
        result[key] = path
    return result


def test_supertonic_adapter_is_registered_and_exported() -> None:
    assert "supertonic" in registered_systems()
    assert SupertonicAdapter.system == "supertonic"


def test_open_local_maps_supertonic_artifacts_without_catalog_or_store(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def forbidden(*args: object, **kwargs: object) -> None:
        pytest.fail("local Supertonic opening must not access the catalog or asset store")

    monkeypatch.setattr("onnxvoice.manager.CatalogClient", forbidden)
    monkeypatch.setattr("onnxvoice.manager.AssetStore", forbidden)
    files = _bundle_files(tmp_path)

    runtime = OnnxVoice.open_local(
        system="supertonic",
        files=files,
        sample_rate=44100,
        artifact_metadata={"duration_predictor": {"source": "local"}},
    )
    try:
        installation = runtime.installation
        assert installation.sample_rate == 44100
        assert installation.artifact("config").path == files["config"].resolve()
        assert installation.artifact("unicode_indexer").path == files["unicode_indexer"].resolve()
        for component in MODEL_COMPONENTS:
            artifact = installation.artifact("model", component=component)
            assert artifact.path == files[component].resolve()
        assert installation.artifact("model", component="duration_predictor").metadata == {
            "source": "local"
        }
        for name in ("F1", "F2", "F3", "F4", "F5", "M1", "M2", "M3", "M4", "M5"):
            assert (
                installation.artifact("voice_style", component=name).path
                == files[f"voice_style:{name}"].resolve()
            )
    finally:
        runtime.close()


def test_open_local_rejects_empty_supertonic_voice_style_component(tmp_path: Path) -> None:
    style = tmp_path / "style.json"
    style.write_text("{}", encoding="utf-8")

    with pytest.raises(ValueError, match="voice_style keys must include a component name"):
        OnnxVoice.open_local(system="supertonic", files={"voice_style:": style})
