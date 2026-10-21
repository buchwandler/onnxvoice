from __future__ import annotations

import json
from pathlib import Path

import pytest

import onnxvoice.runtime_config as runtime_config
from onnxvoice import runtime
from onnxvoice.errors import RuntimeContractError
from onnxvoice.types import OnnxRuntimeStatus, ProviderPreference, RuntimeDistributionInfo


def test_runtime_config_path_is_user_config_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(runtime_config, "user_config_path", lambda app: tmp_path / app)

    assert runtime_config.runtime_config_path() == tmp_path / "onnxvoice" / "runtime.json"


def test_runtime_config_round_trip_ordered_providers_and_atomic_replace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path = tmp_path / "config" / "runtime.json"
    real_replace = runtime_config.os.replace
    replacements: list[tuple[Path, Path]] = []

    def record_replace(source: str | Path, target: str | Path) -> None:
        replacements.append((Path(source), Path(target)))
        real_replace(source, target)

    monkeypatch.setattr(runtime_config.os, "replace", record_replace)
    saved_path = runtime_config.save_runtime_config(("cuda", "cpu"), config_path)

    assert saved_path == config_path
    assert runtime_config.load_runtime_config(config_path) == ("cuda", "cpu")
    assert json.loads(config_path.read_text(encoding="utf-8")) == {
        "schema": 1,
        "providers": ["cuda", "cpu"],
    }
    assert len(replacements) == 1
    assert replacements[0][0].parent == config_path.parent
    assert replacements[0][1] == config_path
    assert not replacements[0][0].exists()
    assert list(config_path.parent.iterdir()) == [config_path]


def test_runtime_config_missing_file_returns_none(tmp_path: Path) -> None:
    assert runtime_config.load_runtime_config(tmp_path / "missing.json") is None


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ("{", "Could not read"),
        ('{"schema": 2, "providers": ["cpu"]}', "schema"),
        ('{"schema": 1, "providers": []}', "non-empty"),
        ('{"schema": 1, "providers": [""]}', "non-empty strings"),
        ('{"schema": 1, "providers": [1]}', "non-empty strings"),
        ('{"providers": ["cpu"]}', "schema"),
    ],
)
def test_runtime_config_rejects_malformed_or_unsupported_values(
    tmp_path: Path, payload: str, message: str
) -> None:
    path = tmp_path / "runtime.json"
    path.write_text(payload, encoding="utf-8")

    with pytest.raises(RuntimeContractError, match=message):
        runtime_config.load_runtime_config(path)


def test_runtime_config_save_rejects_empty_policy(tmp_path: Path) -> None:
    with pytest.raises(RuntimeContractError, match="non-empty list"):
        runtime_config.save_runtime_config((), tmp_path / "runtime.json")


def test_runtime_config_reset_removes_malformed_file(tmp_path: Path) -> None:
    path = tmp_path / "runtime.json"
    path.write_text("not json", encoding="utf-8")

    runtime_config.clear_runtime_config(path)

    assert not path.exists()
    runtime_config.clear_runtime_config(path)


def test_runtime_status_public_dataclasses_expose_stable_fields() -> None:
    distribution = RuntimeDistributionInfo("onnxruntime", "1.24.1", Path("/ort"), True)
    preference = ProviderPreference(("cuda",), "user_config")
    status = OnnxRuntimeStatus(
        importable=True,
        module_version="1.24.1",
        module_path=Path("/ort/onnxruntime/__init__.py"),
        build_device="GPU",
        build_info="build info",
        distributions=(distribution,),
        imported_distribution="onnxruntime-gpu",
        distribution_state="resolved",
        available_providers=("CUDAExecutionProvider", "CPUExecutionProvider"),
        configured_providers=preference.requested,
        provider_source=preference.source,
        resolved_providers=("CUDAExecutionProvider",),
    )

    assert status.imported_distribution == "onnxruntime-gpu"
    assert status.distributions == (distribution,)
    assert preference.source == "user_config"


class FakeDistribution:
    def __init__(self, name: str, module_path: Path | None = None) -> None:
        self.metadata = {"Name": name}
        self.version = "1.24.1"
        self.files = [Path("onnxruntime/__init__.py")] if module_path is not None else []
        self.module_path = module_path

    def locate_file(self, file: str | Path) -> Path:
        if Path(str(file)).name == "__init__.py" and self.module_path is not None:
            return self.module_path
        return Path("/site-packages")


def _patch_distributions(
    monkeypatch: pytest.MonkeyPatch,
    distributions: dict[str, FakeDistribution],
    names: tuple[str, ...],
) -> None:
    monkeypatch.setattr(
        runtime.metadata,
        "packages_distributions",
        lambda: {"onnxruntime": list(names)},
    )

    def get_distribution(name: str) -> FakeDistribution:
        if name not in distributions:
            raise runtime.metadata.PackageNotFoundError(name)
        return distributions[name]

    monkeypatch.setattr(runtime.metadata, "distribution", get_distribution)


def test_provider_preference_precedence(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(runtime, "load_runtime_config", lambda: ("openvino", "cpu"))
    monkeypatch.setattr(
        runtime.importlib,
        "import_module",
        lambda name: pytest.fail("provider preference must not import ONNX Runtime"),
    )

    assert runtime.provider_preference(
        "cuda",
        environment={
            "ONNXVOICE_PROVIDERS": "cpu",
            "ONNXVOICE_PROVIDER": "openvino",
        },
    ) == ProviderPreference(("cuda",), "explicit")
    assert runtime.get_default_providers(
        environment={
            "ONNXVOICE_PROVIDERS": "cuda,cpu",
            "ONNXVOICE_PROVIDER": "openvino",
        }
    ) == ProviderPreference(("cuda", "cpu"), "ONNXVOICE_PROVIDERS")
    assert runtime.get_default_providers(
        environment={"ONNXVOICE_PROVIDER": "openvino"}
    ) == ProviderPreference(("openvino",), "ONNXVOICE_PROVIDER")
    assert runtime.get_default_providers(environment={}) == ProviderPreference(
        ("openvino", "cpu"), "user_config"
    )
    monkeypatch.setattr(runtime, "load_runtime_config", lambda: None)
    assert runtime.get_default_providers(environment={}) == ProviderPreference(("cpu",), "default")


def test_provider_resolution_supports_auto_and_unknown_reported_providers() -> None:
    assert runtime.resolve_providers(
        "auto",
        available=("CustomExecutionProvider", "CPUExecutionProvider"),
    ) == ("CPUExecutionProvider",)
    assert runtime.resolve_providers(
        "CustomExecutionProvider", available=("CustomExecutionProvider",)
    ) == ("CustomExecutionProvider",)
    with pytest.raises(RuntimeContractError, match="unavailable"):
        runtime.resolve_providers("custom", available=("CPUExecutionProvider",))


def test_runtime_status_reports_imported_module_and_distribution_owner(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    module_path = tmp_path / "site-packages" / "onnxruntime" / "__init__.py"
    fake_ort = type(
        "FakeOrt",
        (),
        {
            "__file__": str(module_path),
            "__version__": "1.24.1",
            "get_build_info": staticmethod(lambda: "fake build"),
            "get_device": staticmethod(lambda: "GPU"),
            "get_available_providers": staticmethod(
                lambda: ["CUDAExecutionProvider", "CPUExecutionProvider"]
            ),
        },
    )()
    distribution = FakeDistribution("onnxruntime-gpu", module_path)
    _patch_distributions(monkeypatch, {"onnxruntime-gpu": distribution}, ("onnxruntime-gpu",))
    monkeypatch.setattr(runtime.importlib, "import_module", lambda name: fake_ort)
    monkeypatch.setattr(runtime, "load_runtime_config", lambda: None)

    status = runtime.runtime_status(environment={"ONNXVOICE_PROVIDER": "cuda"})

    assert status.importable
    assert status.module_version == "1.24.1"
    assert status.module_path == module_path.resolve()
    assert status.build_device == "GPU"
    assert status.build_info == "fake build"
    assert status.imported_distribution == "onnxruntime-gpu"
    assert status.distribution_state == "resolved"
    assert status.available_providers == ("CUDAExecutionProvider", "CPUExecutionProvider")
    assert status.configured_providers == ("cuda",)
    assert status.provider_source == "ONNXVOICE_PROVIDER"
    assert status.resolved_providers == ("CUDAExecutionProvider",)


def test_runtime_status_discovers_non_builtin_distribution_names(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    module_path = tmp_path / "custom" / "onnxruntime" / "__init__.py"
    fake_ort = type(
        "FakeOrt",
        (),
        {
            "__file__": str(module_path),
            "__version__": "custom-build",
            "get_available_providers": staticmethod(lambda: ["CustomExecutionProvider"]),
        },
    )()
    custom_distribution = FakeDistribution("onnxruntime-custom", module_path)
    _patch_distributions(
        monkeypatch, {"onnxruntime-custom": custom_distribution}, ("onnxruntime-custom",)
    )
    monkeypatch.setattr(runtime.importlib, "import_module", lambda name: fake_ort)
    monkeypatch.setattr(runtime, "load_runtime_config", lambda: ("CustomExecutionProvider",))

    status = runtime.runtime_status(environment={})

    assert status.distributions[0].name == "onnxruntime-custom"
    assert status.imported_distribution == "onnxruntime-custom"
    assert status.distribution_state == "resolved"
    assert status.resolved_providers == ("CustomExecutionProvider",)


def test_runtime_status_reports_ambiguous_distribution_ownership(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    module_path = tmp_path / "onnxruntime" / "__init__.py"
    fake_ort = type(
        "FakeOrt",
        (),
        {
            "__file__": str(module_path),
            "get_available_providers": staticmethod(lambda: ["CPUExecutionProvider"]),
        },
    )()
    distributions = {
        name: FakeDistribution(name, module_path) for name in ("onnxruntime", "onnxruntime-gpu")
    }
    _patch_distributions(monkeypatch, distributions, tuple(distributions))
    monkeypatch.setattr(runtime.importlib, "import_module", lambda name: fake_ort)
    monkeypatch.setattr(runtime, "load_runtime_config", lambda: None)

    status = runtime.runtime_status(environment={})

    assert status.distribution_state == "ambiguous"
    assert status.imported_distribution is None
    assert {item.name for item in status.distributions if item.owns_import} == set(distributions)


def test_runtime_status_handles_vendor_build_and_optional_methods(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    fake_ort = type(
        "FakeOrt",
        (),
        {
            "__file__": str(tmp_path / "vendor" / "onnxruntime" / "__init__.py"),
            "get_available_providers": staticmethod(lambda: ["CPUExecutionProvider"]),
        },
    )()
    _patch_distributions(monkeypatch, {}, ())
    monkeypatch.setattr(runtime.importlib, "import_module", lambda name: fake_ort)
    monkeypatch.setattr(runtime, "load_runtime_config", lambda: None)

    status = runtime.runtime_status(environment={})

    assert status.importable
    assert status.distribution_state == "vendor"
    assert status.module_version is None
    assert status.build_device is None
    assert status.resolved_providers == ("CPUExecutionProvider",)


def test_runtime_status_distinguishes_missing_and_failed_imports(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(runtime, "load_runtime_config", lambda: None)
    monkeypatch.setattr(runtime.metadata, "packages_distributions", lambda: {})
    monkeypatch.setattr(
        runtime.metadata,
        "distribution",
        lambda name: (_ for _ in ()).throw(runtime.metadata.PackageNotFoundError(name)),
    )
    monkeypatch.setattr(
        runtime.importlib,
        "import_module",
        lambda name: (_ for _ in ()).throw(ImportError("native library missing")),
    )

    status = runtime.runtime_status(environment={})

    assert not status.importable
    assert status.distribution_state == "not_installed"
    assert status.import_error == "ImportError: native library missing"
    assert status.configured_providers == ("cpu",)
    assert status.resolved_providers == ()


def test_runtime_status_reports_installed_metadata_when_import_fails(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _patch_distributions(
        monkeypatch,
        {"onnxruntime-gpu": FakeDistribution("onnxruntime-gpu")},
        ("onnxruntime-gpu",),
    )
    monkeypatch.setattr(runtime, "load_runtime_config", lambda: None)
    monkeypatch.setattr(
        runtime.importlib,
        "import_module",
        lambda name: (_ for _ in ()).throw(OSError("native load failure")),
    )

    status = runtime.runtime_status(environment={})

    assert not status.importable
    assert status.distribution_state == "import_failed"
    assert status.distributions[0].name == "onnxruntime-gpu"
    assert status.import_error == "OSError: native load failure"


def test_runtime_status_reports_malformed_saved_preference_without_crashing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    fake_ort = type(
        "FakeOrt",
        (),
        {
            "__file__": str(tmp_path / "vendor" / "onnxruntime" / "__init__.py"),
            "get_available_providers": staticmethod(lambda: ["CPUExecutionProvider"]),
        },
    )()
    _patch_distributions(monkeypatch, {}, ())
    monkeypatch.setattr(runtime.importlib, "import_module", lambda name: fake_ort)
    monkeypatch.setattr(
        runtime,
        "load_runtime_config",
        lambda: (_ for _ in ()).throw(RuntimeContractError("invalid saved preference")),
    )

    status = runtime.runtime_status(environment={})

    assert status.importable
    assert status.configured_providers == ()
    assert status.provider_source == "user_config"
    assert status.resolved_providers == ()
    assert status.resolution_error == "invalid saved preference"
