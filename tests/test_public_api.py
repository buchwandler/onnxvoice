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


def test_cache_management_api_is_public_and_local(tmp_path, monkeypatch) -> None:
    import hashlib

    import onnxvoice
    from onnxvoice.manager import OnnxVoice
    from onnxvoice.types import Artifact, CacheUsage, CatalogItem, GcPlan, GcReport, RemovalPlan

    expected = (
        "CacheFilesystemUsage",
        "CacheFileUsage",
        "CachePathUsage",
        "CacheScan",
        "CacheUsage",
        "InstallationStorageUsage",
        "GcCandidate",
        "GcPlan",
        "GcReport",
        "RemovalPlan",
        "cache_usage",
        "plan_remove",
        "plan_cache_gc",
        "gc_cache",
    )
    for name in expected:
        assert name in onnxvoice.__all__
        assert hasattr(onnxvoice, name)

    manager = OnnxVoice(cache_dir=tmp_path / "cache", offline=True)
    content = b"public-cache-api"
    digest = hashlib.sha256(content).hexdigest()
    orphan = manager.store.blobs / digest[:2] / digest
    orphan.parent.mkdir(parents=True)
    orphan.write_bytes(content)

    source = tmp_path / "managed.onnx"
    source.write_bytes(b"managed-installation")
    managed_digest = hashlib.sha256(source.read_bytes()).hexdigest()
    manager.store.install(
        CatalogItem(
            system="test",
            id="managed",
            kind="model",
            artifacts=(
                Artifact(
                    "model",
                    source.name,
                    source.as_uri(),
                    source.stat().st_size,
                    managed_digest,
                ),
            ),
        )
    )

    assert isinstance(manager.cache_usage(), CacheUsage)
    gc_plan = manager.plan_cache_gc()
    assert isinstance(gc_plan, GcPlan)
    assert any(candidate.path == orphan for candidate in gc_plan.candidates)
    removal_plan = manager.plan_remove(("test:managed",))
    assert isinstance(removal_plan, RemovalPlan)
    assert removal_plan.installations[0].ref == "test:managed"

    monkeypatch.setattr(onnxvoice, "_default", manager)
    assert isinstance(onnxvoice.cache_usage(), CacheUsage)
    assert isinstance(onnxvoice.plan_cache_gc(), GcPlan)
    assert isinstance(onnxvoice.plan_remove(("test:managed",)), RemovalPlan)
    report = onnxvoice.gc_cache()
    assert isinstance(report, GcReport)
    assert report.removed_blobs == 1
    assert not orphan.exists()
