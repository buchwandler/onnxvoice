from __future__ import annotations

import importlib.util

import pytest


@pytest.mark.skipif(
    importlib.util.find_spec("onnxruntime") is None,
    reason="onnxruntime is an optional dependency",
)
def test_cpu_onnxruntime_provider_is_available():
    import onnxruntime

    assert "CPUExecutionProvider" in onnxruntime.get_available_providers()
