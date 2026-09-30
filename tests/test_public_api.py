def test_package_all_exports_exist():
    import onnxvoice

    for name in onnxvoice.__all__:
        assert hasattr(onnxvoice, name), name
