def test_package_all_exports_exist():
    import onnxvoice

    for name in onnxvoice.__all__:
        assert hasattr(onnxvoice, name), name


def test_public_runtime_api_exports() -> None:
    import onnxvoice
    from onnxvoice import runtime

    expected = (
        "OnnxRuntimeStatus",
        "RuntimeDistributionInfo",
        "ProviderPreference",
        "runtime_status",
        "get_default_providers",
        "set_default_providers",
        "clear_default_providers",
    )
    for name in expected:
        assert name in onnxvoice.__all__
        assert hasattr(onnxvoice, name)

    assert onnxvoice.runtime_status is runtime.runtime_status
    assert onnxvoice.get_default_providers is runtime.get_default_providers
    assert onnxvoice.set_default_providers is runtime.set_default_providers
    assert onnxvoice.clear_default_providers is runtime.clear_default_providers
