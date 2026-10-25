from __future__ import annotations

import importlib
import io
import os
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from urllib.error import URLError
from urllib.request import Request

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


def test_missing_xet_retries_hub_with_runtime_switch_and_exact_arguments(
    monkeypatch, tmp_path: Path, caplog
) -> None:
    caplog.set_level("INFO", logger="onnxvoice.huggingface")
    constants = ModuleType("huggingface_hub.constants")
    constants.__dict__["HF_HUB_DISABLE_XET"] = False
    monkeypatch.setitem(sys.modules, "huggingface_hub.constants", constants)
    calls = []
    source = _source()

    def download(**kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            raise ValueError(_XET_ERROR_MESSAGE)
        assert constants.__dict__["HF_HUB_DISABLE_XET"] is True
        path = Path(kwargs["local_dir"]) / kwargs["filename"]
        path.parent.mkdir(parents=True)
        path.write_bytes(b"Hub HTTP retry")
        return str(path)

    _install_hub(monkeypatch, download)
    monkeypatch.setattr(
        huggingface, "urlopen", lambda *_args, **_kwargs: pytest.fail("unexpected HTTPS fallback")
    )

    result = download_huggingface_file(source, local_dir=tmp_path, offline=False)

    expected = {
        "repo_id": source.repository,
        "filename": source.path,
        "revision": source.revision,
        "local_dir": str(tmp_path),
        "local_files_only": False,
        "token": None,
        "library_name": "onnxvoice",
    }
    assert result.read_bytes() == b"Hub HTTP retry"
    assert calls == [expected, expected]
    assert os.environ["HF_HUB_DISABLE_XET"] == "1"
    assert constants.__dict__["HF_HUB_DISABLE_XET"] is True
    records = [record for record in caplog.records if hasattr(record, "recovery_stage")]
    assert len(records) == 1
    assert records[0].recovery_stage == "huggingface_download"
    assert records[0].retry_attempted is True
    assert records[0].retry_outcome == "succeeded"
    assert records[0].direct_fallback_attempted is False
    assert records[0].direct_fallback_outcome == "not_attempted"


def test_missing_xet_retry_works_when_constant_is_on_loaded_hub_module(
    monkeypatch, tmp_path: Path
) -> None:
    constants = SimpleNamespace(HF_HUB_DISABLE_XET=False)
    hub_calls = []

    def download(**kwargs):
        hub_calls.append(kwargs)
        if len(hub_calls) == 1:
            raise ValueError(_XET_ERROR_MESSAGE)
        assert constants.__dict__["HF_HUB_DISABLE_XET"] is True
        path = Path(kwargs["local_dir"]) / kwargs["filename"]
        path.parent.mkdir(parents=True)
        path.write_bytes(b"loaded constants")
        return str(path)

    hub = _install_hub(monkeypatch, download)
    hub.__dict__["constants"] = constants

    result = download_huggingface_file(_source(), local_dir=tmp_path, offline=False)

    assert result.read_bytes() == b"loaded constants"
    assert len(hub_calls) == 2
    assert hub_calls[0] == hub_calls[1]
    assert constants.__dict__["HF_HUB_DISABLE_XET"] is True


def test_missing_xet_transport_falls_back_to_exact_pinned_https_file(
    monkeypatch, tmp_path: Path
) -> None:
    source = _source()
    hub_calls = []
    url_calls: list[tuple[Request, int] | tuple[str, int | None]] = []
    content = b"fallback asset" * 7000

    class Response(io.BytesIO):
        def read(self, size: int | None = -1) -> bytes:
            url_calls.append(("read", size))
            return super().read(size)

    def download(**kwargs):
        hub_calls.append(kwargs)
        if len(hub_calls) == 1:
            raise ValueError(_XET_ERROR_MESSAGE)
        raise RuntimeError("HTTP-mode retry unavailable")

    def open_url(request: Request, *, timeout: int) -> Response:
        url_calls.append((request, timeout))
        return Response(content)

    hub = _install_hub(monkeypatch, download)
    hub.__dict__["constants"] = ModuleType("huggingface_hub.constants")
    monkeypatch.setattr(huggingface, "urlopen", open_url)

    result = download_huggingface_file(source, local_dir=tmp_path, offline=False)

    expected = {
        "repo_id": source.repository,
        "filename": source.path,
        "revision": source.revision,
        "local_dir": str(tmp_path),
        "local_files_only": False,
        "token": None,
        "library_name": "onnxvoice",
    }
    assert result.read_bytes() == content
    assert result.is_file()
    assert result.resolve().is_relative_to(tmp_path.resolve())
    assert result == tmp_path / source.path
    assert hub_calls == [expected, expected]
    assert os.environ["HF_HUB_DISABLE_XET"] == "1"
    assert "HF_HUB_DISABLE_XET" not in hub.__dict__["constants"].__dict__
    request, timeout = url_calls[0]
    assert isinstance(request, Request)
    assert request.full_url == (
        f"https://huggingface.co/KevinAHM/pocket-tts-onnx/resolve/{source.revision}/{source.path}"
    )
    assert timeout == 30
    assert request.get_header("Authorization") is None
    assert all(size == 64 * 1024 for kind, size in url_calls[1:] if kind == "read")


def test_missing_file_from_hub_retry_uses_https_fallback(monkeypatch, tmp_path: Path) -> None:
    calls = []
    source = _source()
    content = b"retried pinned file"

    def download(**kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            raise ValueError(_XET_ERROR_MESSAGE)
        return str(tmp_path / "missing-retry-result")

    _install_hub(monkeypatch, download)
    monkeypatch.setattr(huggingface, "urlopen", lambda *_args, **_kwargs: io.BytesIO(content))

    result = download_huggingface_file(source, local_dir=tmp_path, offline=False)

    assert len(calls) == 2
    assert calls[0] == calls[1]
    assert result == tmp_path / source.path
    assert result.read_bytes() == content


def test_gated_source_does_not_attempt_unauthenticated_xet_fallback(
    monkeypatch, tmp_path: Path
) -> None:
    source = HuggingFaceSource(
        repository=_source().repository,
        revision=_source().revision,
        path=_source().path,
        gated=True,
    )
    calls = []

    def download(**kwargs):
        calls.append(kwargs)
        raise ValueError(_XET_ERROR_MESSAGE)

    _install_hub(monkeypatch, download)
    monkeypatch.setattr(
        huggingface, "urlopen", lambda *_args, **_kwargs: pytest.fail("unexpected HTTPS fallback")
    )

    with pytest.raises(AssetTransportError) as raised:
        download_huggingface_file(source, local_dir=tmp_path, offline=False)

    assert len(calls) == 2
    assert calls[0] == calls[1]
    assert raised.value.stage == "huggingface_download"
    assert raised.value.native_error_type == "ValueError"
    assert raised.value.initial_error_type == "ValueError"
    assert raised.value.retry_attempted is True
    assert raised.value.retry_failed is True
    assert raised.value.retry_error_type == "ValueError"
    assert raised.value.fallback_attempted is False
    assert raised.value.fallback_failed is False
    assert raised.value.retry_outcome == "failed"
    assert raised.value.fallback_outcome == "not_attempted"
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
    calls = []

    def download(**kwargs):
        calls.append(kwargs)
        raise ValueError(_XET_ERROR_MESSAGE)

    _install_hub(monkeypatch, download)
    monkeypatch.setattr(
        huggingface, "urlopen", lambda *_args, **_kwargs: pytest.fail("unexpected HTTPS fallback")
    )

    with pytest.raises(OfflineError) as raised:
        download_huggingface_file(_source(), local_dir=tmp_path, offline=offline)

    assert len(calls) == 2
    assert calls[0] == calls[1]
    assert calls[0]["local_files_only"] is offline
    assert "hf_xet" in str(raised.value)
    assert "retry attempted=yes, failed=yes" in str(raised.value)
    assert "fallback attempted=no" in str(raised.value)
    assert raised.value.__dict__["retry_outcome"] == "failed"
    assert raised.value.__dict__["fallback_outcome"] == "not_attempted"
    assert "failed=no" in str(raised.value)


@pytest.mark.parametrize(
    ("status", "error_type"),
    [(401, AssetAuthenticationError), (403, AssetPermissionError), (404, AssetNotFoundError)],
)
def test_xet_message_does_not_override_hub_access_or_not_found_errors(
    monkeypatch, tmp_path: Path, status: int, error_type: type[Exception]
) -> None:
    calls = []
    monkeypatch.delenv("HF_HUB_DISABLE_XET", raising=False)

    def download(**kwargs):
        calls.append(kwargs)
        raise _StatusValueError(_XET_ERROR_MESSAGE, status)

    _install_hub(monkeypatch, download)
    monkeypatch.setattr(
        huggingface, "urlopen", lambda *_args, **_kwargs: pytest.fail("unexpected HTTPS fallback")
    )

    with pytest.raises(error_type):
        download_huggingface_file(_source(), local_dir=tmp_path, offline=False)

    assert len(calls) == 1
    assert "HF_HUB_DISABLE_XET" not in os.environ


@pytest.mark.parametrize(
    ("status", "error_type"),
    [(401, AssetAuthenticationError), (403, AssetPermissionError), (404, AssetNotFoundError)],
)
def test_retry_access_and_not_found_errors_are_not_bypassed(
    monkeypatch, tmp_path: Path, status: int, error_type: type[Exception]
) -> None:
    calls = []
    secret = "hf_RETRY_STATUS_SECRET"

    def download(**kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            raise ValueError(_XET_ERROR_MESSAGE)
        raise _StatusRuntimeError(f"private response {secret}", status)

    _install_hub(monkeypatch, download)
    monkeypatch.setattr(
        huggingface, "urlopen", lambda *_args, **_kwargs: pytest.fail("unexpected HTTPS fallback")
    )

    with pytest.raises(error_type) as raised:
        download_huggingface_file(_source(), local_dir=tmp_path, offline=False)

    assert len(calls) == 2
    assert calls[0] == calls[1]
    assert "HTTP-mode retry was attempted" in str(raised.value)
    assert "fallback was not attempted" in str(raised.value)
    assert raised.value.__dict__["retry_outcome"] == "failed"
    assert raised.value.__dict__["fallback_outcome"] == "not_attempted"
    assert secret not in str(raised.value)
    assert secret not in repr(raised.value)


def test_retry_revision_not_found_is_not_bypassed(monkeypatch, tmp_path: Path) -> None:
    calls = []

    class RevisionNotFoundError(Exception):
        pass

    def download(**kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            raise ValueError(_XET_ERROR_MESSAGE)
        raise RevisionNotFoundError("private revision detail")

    _install_hub(monkeypatch, download)
    monkeypatch.setattr(
        huggingface, "urlopen", lambda *_args, **_kwargs: pytest.fail("unexpected HTTPS fallback")
    )

    with pytest.raises(AssetNotFoundError) as raised:
        download_huggingface_file(_source(), local_dir=tmp_path, offline=False)

    assert len(calls) == 2
    assert "not found" in str(raised.value).lower()
    assert "direct HTTPS fallback was not attempted" in str(raised.value)


def test_unrelated_value_error_is_not_classified_as_xet_or_leaked(
    monkeypatch, tmp_path: Path
) -> None:
    secret = "hf_UNRELATED_ERROR_SECRET"
    calls = []
    monkeypatch.delenv("HF_HUB_DISABLE_XET", raising=False)

    def download(**kwargs):
        calls.append(kwargs)
        raise ValueError(f"unrelated {secret}")

    _install_hub(monkeypatch, download)
    monkeypatch.setattr(
        huggingface, "urlopen", lambda *_args, **_kwargs: pytest.fail("unexpected HTTPS fallback")
    )

    with pytest.raises(AssetDownloadError) as raised:
        download_huggingface_file(_source(), local_dir=tmp_path, offline=False)

    assert len(calls) == 1
    assert "HF_HUB_DISABLE_XET" not in os.environ
    assert not isinstance(raised.value, AssetTransportError)
    assert "ValueError" in str(raised.value)
    assert secret not in str(raised.value)
    assert secret not in repr(raised.value)


def test_xet_fallback_failure_has_safe_structured_diagnostics(
    monkeypatch, tmp_path: Path, caplog
) -> None:
    caplog.set_level("INFO", logger="onnxvoice.huggingface")
    retry_secret = "hf_RETRY_EXCEPTION_SECRET"
    fallback_secret = "hf_SIGNED_URL_QUERY_SECRET"
    fallback_calls = []
    hub_calls = []

    def download(**kwargs):
        hub_calls.append(kwargs)
        if len(hub_calls) == 1:
            raise ValueError(_XET_ERROR_MESSAGE)
        raise RuntimeError(f"private retry response {retry_secret}")

    _install_hub(monkeypatch, download)

    def fail_urlopen(_request, *, timeout):
        fallback_calls.append(_request)
        assert timeout == 30
        raise URLError(f"https://cdn.example/file?token={fallback_secret}")

    monkeypatch.setattr(huggingface, "urlopen", fail_urlopen)

    with pytest.raises(AssetTransportError) as raised:
        download_huggingface_file(_source(), local_dir=tmp_path, offline=False)

    assert len(hub_calls) == 2
    assert len(fallback_calls) == 1
    assert hub_calls[0] == hub_calls[1]
    assert raised.value.stage == "xet_fallback"
    assert raised.value.native_error_type == "URLError"
    assert raised.value.initial_error_type == "ValueError"
    assert raised.value.retry_attempted is True
    assert raised.value.retry_failed is True
    assert raised.value.retry_error_type == "RuntimeError"
    assert raised.value.runtime_switch_activated is False
    assert raised.value.fallback_attempted is True
    assert raised.value.fallback_failed is True
    assert raised.value.retry_outcome == "failed"
    assert raised.value.fallback_outcome == "failed"
    assert raised.value.fallback_error_type == "URLError"
    assert "Hugging Face asset" in str(raised.value)
    assert "HTTP-mode retry failed" in str(raised.value)
    assert "pinned HTTPS request could not retrieve" in str(raised.value)
    for secret in (retry_secret, fallback_secret):
        assert secret not in str(raised.value)
        assert secret not in repr(raised.value)
    assert raised.value.__cause__ is None
    records = [record for record in caplog.records if hasattr(record, "recovery_stage")]
    assert len(records) == 1
    assert records[0].initial_error_type == "ValueError"
    assert records[0].retry_outcome == "failed"
    assert records[0].retry_error_type == "RuntimeError"
    assert records[0].direct_fallback_attempted is True
    assert records[0].direct_fallback_outcome == "failed"
    assert records[0].direct_fallback_error_type == "URLError"
    assert retry_secret not in caplog.text
    assert fallback_secret not in caplog.text


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
