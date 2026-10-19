from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from _kitten2_fixture import create_bundle

import onnxvoice.manager as manager_module
from onnxvoice.errors import RuntimeContractError
from onnxvoice.manager import OnnxVoice
from onnxvoice.types import Installation


class RuntimeStub:
    def __init__(self, installation: Installation, **kwargs: Any) -> None:
        self.installation = installation
        self.kwargs = kwargs


def _forbidden(*_args: Any, **_kwargs: Any) -> None:
    pytest.fail("Kitten2 open_local must not access the catalog or asset store")


def test_open_local_uses_bundle_root_and_preserves_nested_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundle_path = create_bundle(tmp_path / "local-kit")
    adapter_args: dict[str, Any] = {}

    def make_adapter(system: str):
        assert system == "kitten2"

        def create(installation: Installation, **kwargs: Any) -> RuntimeStub:
            adapter_args["installation"] = installation
            adapter_args["kwargs"] = kwargs
            return RuntimeStub(installation, **kwargs)

        return create

    monkeypatch.setattr(manager_module.AssetStore, "__init__", _forbidden)
    monkeypatch.setattr(manager_module.CatalogClient, "__init__", _forbidden)
    monkeypatch.setattr(manager_module, "get_adapter", make_adapter)

    runtime = OnnxVoice.open_local(
        system="kitten2",
        files={"bundle": bundle_path},
        providers="cpu",
        provider_options=[{"device_id": "2"}],
        session_options=object(),
    )

    installation = runtime.installation
    filenames = {artifact.filename for artifact in installation.artifacts}
    assert installation.system == "kitten2"
    assert installation.path == bundle_path.parent
    assert installation.sample_rate == 24000
    assert installation.voices == ("Bruno",)
    assert installation.default_voice == "Bruno"
    assert installation.metadata["runtime"] == {
        "layout": "kitten2-onnx-bundle-v1",
        "sample_rate": 24000,
    }
    assert "decoder/mel2wav/mel2wav.onnx.data" in filenames
    graph = next(
        artifact
        for artifact in installation.artifacts
        if artifact.filename == "decoder/mel2wav/mel2wav.onnx"
    )
    external_data = next(
        artifact
        for artifact in installation.artifacts
        if artifact.filename == "decoder/mel2wav/mel2wav.onnx.data"
    )
    assert external_data.path == Path(str(graph.path) + ".data")
    assert all(
        artifact.path.is_relative_to(installation.path) for artifact in installation.artifacts
    )
    assert adapter_args["kwargs"]["providers"] == "cpu"
    assert adapter_args["kwargs"]["provider_options"] == [{"device_id": "2"}]


def test_open_local_rejects_missing_bundle_and_files_outside_root(tmp_path: Path) -> None:
    ordinary_file = tmp_path / "not-a-bundle.onnx"
    ordinary_file.write_bytes(b"model")
    with pytest.raises(RuntimeContractError, match="exactly one bundle.json"):
        OnnxVoice.open_local(system="kitten2", model=ordinary_file)

    bundle_path = create_bundle(tmp_path / "bundle")
    outside = tmp_path / "outside.bin"
    outside.write_bytes(b"outside")
    with pytest.raises(RuntimeContractError, match="outside the bundle root"):
        OnnxVoice.open_local(
            system="kitten2",
            files={"bundle": bundle_path, "extra": outside},
        )
