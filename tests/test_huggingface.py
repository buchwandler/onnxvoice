from __future__ import annotations

import importlib
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from onnxvoice.errors import (
    AssetAuthenticationError,
    AssetDownloadError,
    AssetNotFoundError,
    AssetPermissionError,
    OfflineError,
    OptionalDependencyError,
)
from onnxvoice.huggingface import HuggingFaceSource, download_huggingface_file


def _source() -> HuggingFaceSource:
    return HuggingFaceSource(
        repository="KevinAHM/pocket-tts-onnx",
        revision="a" * 40,
        path="onnx/english_2026-04/bundle.json",
    )


def _install_hub(monkeypatch, download) -> ModuleType:
    hub = ModuleType("huggingface_hub")
    hub.hf_hub_download = download
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)
    return hub


def test_source_rejects_unsafe_repository_paths() -> None:
    with pytest.raises(ValueError, match="Unsafe"):
        HuggingFaceSource("owner/repo", "revision", "../secret")
    with pytest.raises(ValueError, match="owner/repository"):
        HuggingFaceSource("not-a-repository", "revision", "model.onnx")


def test_missing_optional_dependency_has_install_instructions(monkeypatch, tmp_path: Path) -> None:
    original_import = importlib.import_module

    def import_without_hub(name: str, package: str | None = None):
        if name == "huggingface_hub":
            raise ModuleNotFoundError(name)
        return original_import(name, package)

    monkeypatch.setattr(importlib, "import_module", import_without_hub)
    with pytest.raises(OptionalDependencyError, match=r"onnxvoice\[pocket\]"):
        download_huggingface_file(_source(), local_dir=tmp_path, offline=False)


def test_download_uses_exact_pinned_source_and_temporary_local_dir(
    monkeypatch, tmp_path: Path
) -> None:
    source = _source()
    calls = []

    def download(**kwargs):
        calls.append(kwargs)
        path = Path(kwargs["local_dir"]) / kwargs["filename"]
        path.parent.mkdir(parents=True)
        path.write_bytes(b"bundle")
        return str(path)

    _install_hub(monkeypatch, download)
    result = download_huggingface_file(source, local_dir=tmp_path, offline=False)

    assert result.read_bytes() == b"bundle"
    assert calls == [
        {
            "repo_id": "KevinAHM/pocket-tts-onnx",
            "filename": "onnx/english_2026-04/bundle.json",
            "revision": "a" * 40,
            "local_dir": str(tmp_path),
            "local_files_only": False,
            "token": None,
            "library_name": "onnxvoice",
        }
    ]


def test_download_missing_returned_file_is_an_asset_error(monkeypatch, tmp_path: Path) -> None:
    _install_hub(monkeypatch, lambda **_kwargs: str(tmp_path / "missing"))
    with pytest.raises(AssetDownloadError, match="returned a missing file"):
        download_huggingface_file(_source(), local_dir=tmp_path, offline=False)


@pytest.mark.parametrize(
    ("status", "error_type", "message"),
    [
        (401, AssetAuthenticationError, "hf auth login"),
        (403, AssetPermissionError, "accept or request access"),
        (404, AssetNotFoundError, "asset not found"),
    ],
)
def test_http_errors_map_without_exposing_exception_text(
    monkeypatch, tmp_path: Path, status: int, error_type: type[Exception], message: str
) -> None:
    secret = "hf_FAKE_SECRET_SENTINEL"

    def download(**_kwargs):
        error = RuntimeError(f"request failed with {secret}")
        error.response = SimpleNamespace(status_code=status)
        raise error

    _install_hub(monkeypatch, download)
    with pytest.raises(error_type) as raised:
        download_huggingface_file(_source(), local_dir=tmp_path, offline=False)
    assert message in str(raised.value).lower()
    assert secret not in str(raised.value)
    assert secret not in repr(raised.value)


def test_missing_revision_exception_maps_to_asset_not_found(monkeypatch, tmp_path: Path) -> None:
    class RevisionNotFoundError(Exception):
        pass

    def download(**_kwargs):
        raise RevisionNotFoundError("not found")

    _install_hub(monkeypatch, download)
    with pytest.raises(AssetNotFoundError, match="asset not found"):
        download_huggingface_file(_source(), local_dir=tmp_path, offline=False)


def test_offline_local_cache_miss_maps_to_offline_error(monkeypatch, tmp_path: Path) -> None:
    def download(**kwargs):
        assert kwargs["local_files_only"] is True
        raise RuntimeError("local cache miss")

    _install_hub(monkeypatch, download)
    with pytest.raises(OfflineError, match="while offline"):
        download_huggingface_file(_source(), local_dir=tmp_path, offline=True)
