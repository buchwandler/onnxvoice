from __future__ import annotations

import importlib.util
import inspect
import os
from pathlib import Path
from typing import Any

_COMPAT_ENABLED = os.environ.get("ONNXVOICE_PHONONSYNTH_COMPAT") == "1"
_SYNTHESIS_ENABLED = os.environ.get("ONNXVOICE_PHONONSYNTH_ACCEPTANCE") == "1"


def _run_contract_smoke(*, require_wheel: bool = False) -> tuple[Any, ...]:
    import phononsynth

    import onnxvoice
    from onnxvoice.systems import get_adapter

    if phononsynth.__version__ != "0.1.0":
        raise AssertionError(f"expected phononsynth==0.1.0, got {phononsynth.__version__}")
    if require_wheel:
        if onnxvoice.__version__ != "0.2.16":
            raise AssertionError(
                f"expected candidate onnxvoice==0.2.16, got {onnxvoice.__version__}"
            )
        source_package = Path(__file__).resolve().parents[1] / "onnxvoice"
        loaded_package = Path(onnxvoice.__file__).resolve()
        if loaded_package.is_relative_to(source_package.resolve()):
            raise AssertionError(
                "ONNXVoice was imported from the checkout; run this smoke outside the source tree "
                "after installing the candidate wheel"
            )

    manager_parameters = inspect.signature(onnxvoice.OnnxVoice).parameters
    assert {"cache_dir", "catalog_sources", "offline"} <= set(manager_parameters)
    for method, required in (
        (onnxvoice.OnnxVoice.install, {"quality", "refresh", "force", "progress"}),
        (onnxvoice.OnnxVoice.open, {"providers", "provider_options", "session_options"}),
        (
            onnxvoice.OnnxVoice.open_local,
            {"system", "artifacts", "providers", "provider_options", "session_options"},
        ),
    ):
        assert required <= set(inspect.signature(method).parameters)
    assert get_adapter("phonon") is not None

    catalog_url = os.environ.get("ONNXVOICE_PHONONSYNTH_CATALOG_URL")
    models = phononsynth.discover_models(catalog_url=catalog_url)
    target = next((model for model in models if model.id == "phonon-31m-en"), None)
    assert target is not None, "phonon-31m-en was not discovered from the configured catalog"
    assert "q4" in target.quality_choices
    return tuple(models)


def test_released_phononsynth_contract_against_candidate_onnxvoice() -> None:
    import pytest

    if not _COMPAT_ENABLED:
        pytest.skip("set ONNXVOICE_PHONONSYNTH_COMPAT=1 in release validation")
    try:
        _run_contract_smoke(require_wheel=os.environ.get("ONNXVOICE_REQUIRE_WHEEL") == "1")
    except ImportError as exc:
        pytest.fail(f"release smoke requires phononsynth==0.1.0 and ONNXVoice: {exc}")


def test_released_phononsynth_q4_text_synthesis(tmp_path: Path) -> None:
    import pytest

    if not _SYNTHESIS_ENABLED:
        pytest.skip("set ONNXVOICE_PHONONSYNTH_ACCEPTANCE=1 for artifact-backed synthesis")
    try:
        import phononsynth
    except ImportError as exc:
        pytest.fail(f"artifact-backed smoke requires phononsynth==0.1.0: {exc}")
    if phononsynth.__version__ != "0.1.0":
        pytest.fail(f"expected phononsynth==0.1.0, got {phononsynth.__version__}")
    if os.environ.get("ONNXVOICE_REQUIRE_WHEEL") == "1":
        _run_contract_smoke(require_wheel=True)
    if importlib.util.find_spec("onnxruntime") is None:
        pytest.fail("onnxruntime is required for the PhononSynth q4 acceptance gate")

    import numpy as np

    models = phononsynth.discover_models(
        catalog_url=os.environ.get("ONNXVOICE_PHONONSYNTH_CATALOG_URL")
    )
    selected = next((model for model in models if model.id == "phonon-31m-en"), None)
    if selected is None or "q4" not in selected.quality_choices:
        pytest.fail("the configured catalog must publish phonon-31m-en/q4")

    cache_dir = (
        Path(os.environ["ONNXVOICE_PHONONSYNTH_CACHE_DIR"])
        if "ONNXVOICE_PHONONSYNTH_CACHE_DIR" in os.environ
        else tmp_path / "phononsynth-cache"
    )
    with phononsynth.PhononVoice.from_pretrained(
        selected.id,
        quality="q4",
        cache_dir=cache_dir,
        providers="cpu",
    ) as voice:
        result = voice.synthesize_prepared("Hello.", seed=1)
        assert result.audio.ndim == 1
        assert result.audio.size > 0
        assert np.isfinite(result.audio).all()
        assert result.sample_rate > 0
        assert result.voice == voice.default_voice
        assert result.metadata["model_id"] == selected.id
        assert result.metadata["selected_quality"] == "q4"


if __name__ == "__main__":
    _run_contract_smoke(require_wheel=os.environ.get("ONNXVOICE_REQUIRE_WHEEL") == "1")
