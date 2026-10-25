from __future__ import annotations

import importlib
import io
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from urllib.error import URLError

import pytest

import onnxvoice.huggingface as huggingface
from onnxvoice.errors import (
    AssetAuthenticationError,
    AssetDownloadError,
    AssetNotFoundError,
    AssetPermissionError,
    AssetTransportError,
    OfflineError,
    OptionalDependencyError,
)
from onnxvoice.huggingface import (
    HuggingFaceSource,
    download_huggingface_file,
    huggingface_diagnostics,
)


def _source() -> HuggingFaceSource:
    return HuggingFaceSource(
        repository="KevinAHM/pocket-tts-onnx",
        revision="a" * 40,
        path="onnx/english_2026-04/bundle.json",
    )


class _StatusRuntimeError(RuntimeError):
    response: SimpleNamespace

    def __init__(self, message: str, status: int) -> None:
        super().__init__(message)
        self.response = SimpleNamespace(status_code=status)


class _StatusValueError(ValueError):
    response: SimpleNamespace

    def __init__(self, message: str, status: int) -> None:
        super().__init__(message)
        self.response = SimpleNamespace(status_code=status)


def _install_hub(monkeypatch, download) -> ModuleType:
    hub = ModuleType("huggingface_hub")
    hub.__dict__["hf_hub_download"] = download
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
            raise ModuleNotFoundError("No module named 'huggingface_hub'", name=name)
        return original_import(name, package)

    monkeypatch.setattr(importlib, "import_module", import_without_hub)
    with pytest.raises(OptionalDependencyError, match=r"onnxvoice\[huggingface\]") as raised:
        download_huggingface_file(_source(), local_dir=tmp_path, offline=False)
    assert "pocket" not in str(raised.value).lower()


def test_nested_hub_import_error_is_not_reported_as_missing_dependency(
    monkeypatch, tmp_path: Path
) -> None:
    original_import = importlib.import_module
    nested_error = ImportError("failed to import an installed Hub runtime dependency")

    def import_with_nested_error(name: str, package: str | None = None):
        if name == "huggingface_hub":
            raise nested_error
        return original_import(name, package)

    monkeypatch.setattr(importlib, "import_module", import_with_nested_error)
    with pytest.raises(AssetDownloadError) as raised:
        download_huggingface_file(_source(), local_dir=tmp_path, offline=False)

    assert raised.value.__cause__ is nested_error
    assert "onnxvoice[huggingface]" not in str(raised.value)
    assert "pocket" not in str(raised.value).lower()


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
        error = _StatusRuntimeError(f"request failed with {secret}", status)
        raise error

    _install_hub(monkeypatch, download)
    with pytest.raises(error_type) as raised:
        download_huggingface_file(_source(), local_dir=tmp_path, offline=False)
    assert message in str(raised.value).lower()
    assert secret not in str(raised.value)
    assert secret not in repr(raised.value)
    assert "Pocket asset" not in str(raised.value)


def test_missing_revision_exception_maps_to_asset_not_found(monkeypatch, tmp_path: Path) -> None:
    class RevisionNotFoundError(Exception):
        pass

    def download(**_kwargs):
        raise RevisionNotFoundError("not found")

    _install_hub(monkeypatch, download)
    with pytest.raises(AssetNotFoundError, match="asset not found"):
        download_huggingface_file(_source(), local_dir=tmp_path, offline=False)


def test_diagnostics_report_hub_import_failure_accurately(monkeypatch) -> None:
    original_import = importlib.import_module

    def import_with_runtime_failure(name: str, package: str | None = None):
        if name == "huggingface_hub":
            raise ImportError("nested Hub runtime import failure")
        return original_import(name, package)

    monkeypatch.setattr(importlib, "import_module", import_with_runtime_failure)
    report = huggingface_diagnostics()

    assert report["huggingface_hub"] == "import failed (ImportError)"


def test_offline_local_cache_miss_maps_to_offline_error(monkeypatch, tmp_path: Path) -> None:
    def download(**kwargs):
        assert kwargs["local_files_only"] is True
        raise RuntimeError("local cache miss")

    _install_hub(monkeypatch, download)
    with pytest.raises(OfflineError, match="while offline"):
        download_huggingface_file(_source(), local_dir=tmp_path, offline=True)


_XET_ERROR_MESSAGE = (
    "Xet storage is enabled for this repository, but the hf_xet package is not installed. "
    "Please install hf_xet to use this transport."
)


def _raise_xet_error(**_kwargs) -> None:
    raise ValueError(_XET_ERROR_MESSAGE)


def test_missing_xet_transport_falls_back_to_exact_pinned_https_file(
    monkeypatch, tmp_path: Path
) -> None:
    source = _source()
    hub_calls = []
    url_calls = []
    content = b"fallback asset" * 7000

    class Response(io.BytesIO):
        def read(self, size: int | None = -1) -> bytes:
            url_calls.append(("read", size))
            return super().read(size)

    def download(**kwargs):
        hub_calls.append(kwargs)
        raise ValueError(_XET_ERROR_MESSAGE)

    def open_url(request, *, timeout):
        url_calls.append((request, timeout))
        return Response(content)

    _install_hub(monkeypatch, download)
    monkeypatch.setattr(huggingface, "urlopen", open_url)

    result = download_huggingface_file(source, local_dir=tmp_path, offline=False)

    assert result.read_bytes() == content
    assert result.is_file()
    assert result.resolve().is_relative_to(tmp_path.resolve())
    assert result == tmp_path / source.path
    assert hub_calls[0]["repo_id"] == source.repository
    assert hub_calls[0]["revision"] == source.revision
    assert hub_calls[0]["filename"] == source.path
    request, timeout = url_calls[0]
    assert request.full_url == (
        f"https://huggingface.co/KevinAHM/pocket-tts-onnx/resolve/{source.revision}/{source.path}"
    )
    assert timeout == 30
    assert request.get_header("Authorization") is None
    assert all(size == 64 * 1024 for kind, size in url_calls[1:] if kind == "read")


def test_gated_source_does_not_attempt_unauthenticated_xet_fallback(
    monkeypatch, tmp_path: Path
) -> None:
    source = HuggingFaceSource(
        repository=_source().repository,
        revision=_source().revision,
        path=_source().path,
        gated=True,
    )
    _install_hub(monkeypatch, _raise_xet_error)
    monkeypatch.setattr(
        huggingface, "urlopen", lambda *_args, **_kwargs: pytest.fail("unexpected HTTPS fallback")
    )

    with pytest.raises(AssetTransportError) as raised:
        download_huggingface_file(source, local_dir=tmp_path, offline=False)

    assert raised.value.stage == "huggingface_download"
    assert raised.value.native_error_type == "ValueError"
    assert raised.value.fallback_attempted is False
    assert raised.value.fallback_failed is False
    assert "gated" in str(raised.value).lower()
    assert "hf_xet" in str(raised.value)


@pytest.mark.parametrize(("offline", "offline_env"), [(True, None), (False, "1")])
def test_offline_mode_never_attempts_xet_fallback(
    monkeypatch, tmp_path: Path, offline: bool, offline_env: str | None
) -> None:
    if offline_env is None:
        monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    else:
        monkeypatch.setenv("HF_HUB_OFFLINE", offline_env)
    _install_hub(monkeypatch, _raise_xet_error)
    monkeypatch.setattr(
        huggingface, "urlopen", lambda *_args, **_kwargs: pytest.fail("unexpected HTTPS fallback")
    )

    with pytest.raises(OfflineError) as raised:
        download_huggingface_file(_source(), local_dir=tmp_path, offline=offline)

    assert "hf_xet" in str(raised.value)
    assert "fallback attempted=no" in str(raised.value)
    assert "failed=no" in str(raised.value)


@pytest.mark.parametrize(
    ("status", "error_type"),
    [(401, AssetAuthenticationError), (403, AssetPermissionError), (404, AssetNotFoundError)],
)
def test_xet_message_does_not_override_hub_access_or_not_found_errors(
    monkeypatch, tmp_path: Path, status: int, error_type: type[Exception]
) -> None:
    def download(**_kwargs):
        error = _StatusValueError(_XET_ERROR_MESSAGE, status)
        raise error

    _install_hub(monkeypatch, download)
    monkeypatch.setattr(
        huggingface, "urlopen", lambda *_args, **_kwargs: pytest.fail("unexpected HTTPS fallback")
    )

    with pytest.raises(error_type):
        download_huggingface_file(_source(), local_dir=tmp_path, offline=False)


def test_unrelated_value_error_is_not_classified_as_xet_or_leaked(
    monkeypatch, tmp_path: Path
) -> None:
    secret = "hf_UNRELATED_ERROR_SECRET"
    _install_hub(
        monkeypatch,
        lambda **_kwargs: (_ for _ in ()).throw(ValueError(f"unrelated {secret}")),
    )
    monkeypatch.setattr(
        huggingface, "urlopen", lambda *_args, **_kwargs: pytest.fail("unexpected HTTPS fallback")
    )

    with pytest.raises(AssetDownloadError) as raised:
        download_huggingface_file(_source(), local_dir=tmp_path, offline=False)

    assert not isinstance(raised.value, AssetTransportError)
    assert "ValueError" in str(raised.value)
    assert secret not in str(raised.value)
    assert secret not in repr(raised.value)


def test_xet_fallback_failure_has_safe_structured_diagnostics(monkeypatch, tmp_path: Path) -> None:
    secret = "hf_SIGNED_URL_QUERY_SECRET"
    _install_hub(monkeypatch, _raise_xet_error)

    def fail_urlopen(_request, *, timeout):
        assert timeout == 30
        raise URLError(f"https://cdn.example/file?token={secret}")

    monkeypatch.setattr(huggingface, "urlopen", fail_urlopen)

    with pytest.raises(AssetTransportError) as raised:
        download_huggingface_file(_source(), local_dir=tmp_path, offline=False)

    assert raised.value.stage == "xet_fallback"
    assert raised.value.native_error_type == "URLError"
    assert raised.value.fallback_attempted is True
    assert raised.value.fallback_failed is True
    assert "Hugging Face asset" in str(raised.value)
    assert "pinned HTTPS request could not retrieve" in str(raised.value)
    assert secret not in str(raised.value)
    assert secret not in repr(raised.value)


def test_xet_fallback_rejects_symlink_escape_before_request(monkeypatch, tmp_path: Path) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    local_dir = tmp_path / "local"
    local_dir.mkdir()
    try:
        (local_dir / "onnx").symlink_to(outside, target_is_directory=True)
    except (NotImplementedError, OSError):
        pytest.skip("directory symlinks are not supported")
    _install_hub(monkeypatch, _raise_xet_error)
    monkeypatch.setattr(
        huggingface, "urlopen", lambda *_args, **_kwargs: pytest.fail("unexpected HTTPS fallback")
    )

    with pytest.raises(AssetTransportError) as raised:
        download_huggingface_file(_source(), local_dir=local_dir, offline=False)

    assert raised.value.stage == "xet_fallback"
    assert not list(outside.iterdir())
