from __future__ import annotations

import os

import pytest

from onnxvoice import NotInstalledError, OnnxVoice

_REQUIRE_REAL = os.environ.get("ONNXVOICE_REQUIRE_PIPER_REAL") == "1"


@pytest.mark.integration
@pytest.mark.resource_heavy
def test_managed_amy_medium_reports_variable_sequence_axis(monkeypatch):
    """Inspect the actual managed Piper graph without executing acoustic inference."""
    try:
        import onnxruntime  # noqa: F401
    except ImportError:
        message = "onnxruntime is required for the Piper graph integration check"
        if _REQUIRE_REAL:
            pytest.fail(message)
        pytest.skip("onnxruntime is not installed")

    manager = OnnxVoice(
        cache_dir=os.environ.get("ONNXVOICE_TEST_CACHE"),
        offline=True,
    )
    try:
        installation = manager.resolve("piper:en_US-amy-medium")
    except NotInstalledError:
        message = "piper:en_US-amy-medium is not installed in the offline OnnxVoice store"
        if _REQUIRE_REAL:
            pytest.fail(message)
        pytest.skip(message)

    adapter = manager.open(installation, providers="cpu")
    try:
        session = adapter.session
        input_spec = next(spec for spec in session.input_specs if spec.name == "input")
        assert input_spec.shape is not None

        run_called = False

        def fail_if_inferred(_inputs):
            nonlocal run_called
            run_called = True
            raise AssertionError("sequence contract inspection must not execute inference")

        monkeypatch.setattr(session, "run", fail_if_inferred)
        contract = adapter.phoneme_sequence_contract

        assert contract.kind == "variable"
        assert contract.axis == 1
        assert contract.source == "onnx.input.shape"
        assert run_called is False
    finally:
        adapter.close()
