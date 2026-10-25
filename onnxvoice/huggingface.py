from __future__ import annotations

import importlib
import logging
import os
import re
import socket
import tempfile
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, TypedDict
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen

from .errors import (
    AssetAuthenticationError,
    AssetDownloadError,
    AssetNotFoundError,
    AssetPermissionError,
    AssetTransportError,
    OfflineError,
    OptionalDependencyError,
)
from .types import validate_relative_path

_REPOSITORY_RE = re.compile(r"^[^/\\\s]+/[^/\\\s]+$")
_LOGGER = logging.getLogger(__name__)


class _HubDownloadKwargs(TypedDict):
    repo_id: str
    filename: str
    revision: str
    local_dir: str
    local_files_only: bool
    token: None
    library_name: str


@dataclass(frozen=True, slots=True)
class HuggingFaceSource:
    repository: str
    revision: str
    path: str
    gated: bool = False

    def __post_init__(self) -> None:
        if (
            not isinstance(self.repository, str)
            or _REPOSITORY_RE.fullmatch(self.repository) is None
        ):
            raise ValueError("Hugging Face repository must have the form 'owner/repository'")
        if not isinstance(self.revision, str) or not self.revision:
            raise ValueError("Hugging Face revision must be a non-empty string")
        validate_relative_path(self.path, field_name="Hugging Face repository path")


def download_huggingface_file(
    source: HuggingFaceSource,
    *,
    local_dir: Path,
    offline: bool,
) -> Path:
    """Download one pinned Hub file into caller-owned temporary storage."""
    try:
        hub = importlib.import_module("huggingface_hub")
    except ModuleNotFoundError as exc:
        if exc.name == "huggingface_hub":
            raise OptionalDependencyError(
                "Hugging Face downloads require 'huggingface_hub'; "
                "install 'onnxvoice[huggingface]'."
            ) from exc
        raise AssetDownloadError(
            "The Hugging Face client could not be imported because a dependency is missing."
        ) from exc
    except ImportError as exc:
        raise AssetDownloadError(
            f"The Hugging Face client could not be imported ({type(exc).__name__})."
        ) from exc
    download_kwargs: _HubDownloadKwargs = {
        "repo_id": source.repository,
        "filename": source.path,
        "revision": source.revision,
        "local_dir": str(local_dir),
        "local_files_only": offline,
        "token": None,
        "library_name": "onnxvoice",
    }
    try:
        downloaded = hub.hf_hub_download(**download_kwargs)
    except Exception as exc:
        return _raise_huggingface_error(
            exc,
            source=source,
            offline=offline,
            local_dir=local_dir,
            hub=hub,
            download_kwargs=download_kwargs,
        )
    result = Path(downloaded)
    if not result.is_file():
        raise AssetDownloadError(
            f"Hugging Face returned a missing file for "
            f"{source.repository}@{source.revision}:{source.path}"
        )
    return result


def _raise_huggingface_error(
    exc: Exception,
    *,
    source: HuggingFaceSource,
    offline: bool,
    local_dir: Path,
    hub: ModuleType,
    download_kwargs: _HubDownloadKwargs,
) -> Path:
    classified = _classified_huggingface_error(exc, source=source, offline=offline)
    if classified is not None:
        raise classified from exc
    return _recover_missing_xet(
        exc,
        source=source,
        offline=offline,
        local_dir=local_dir,
        hub=hub,
        download_kwargs=download_kwargs,
    )


def _classified_huggingface_error(
    exc: Exception,
    *,
    source: HuggingFaceSource,
    offline: bool,
) -> Exception | None:
    error_types = _hub_error_types()
    status = _http_status(exc)
    label = f"{source.repository}@{source.revision}:{source.path}"

    if _is_instance(exc, error_types["gated"]) or status == 403:
        return AssetPermissionError(
            f"Hugging Face asset {label} requires access to a gated repository. "
            "Accept or request access on the repository page, authenticate with "
            "'hf auth login' or HF_TOKEN, verify with 'hf auth whoami', and retry."
        )
    if status == 401:
        return AssetAuthenticationError(
            f"Authentication is required to download Hugging Face asset {label}. "
            "Authenticate with 'hf auth login' or configure HF_TOKEN, then retry."
        )
    if _is_instance(exc, error_types["offline"]) or (
        offline and _is_instance(exc, error_types["local_entry"])
    ):
        return OfflineError(f"Hugging Face asset {label} is not available in the local cache")
    if _is_instance(exc, error_types["missing"]) or status == 404:
        return AssetNotFoundError(f"Hugging Face asset not found: {label}")

    error_name = type(exc).__name__
    if error_name == "GatedRepoError":
        return AssetPermissionError(
            f"Hugging Face asset {label} requires access to a gated repository. "
            "Accept or request access on the repository page, authenticate with "
            "'hf auth login' or HF_TOKEN, verify with 'hf auth whoami', and retry."
        )
    if error_name in {
        "RepositoryNotFoundError",
        "RevisionNotFoundError",
        "EntryNotFoundError",
        "RemoteEntryNotFoundError",
    }:
        return AssetNotFoundError(f"Hugging Face asset not found: {label}")
    if error_name == "LocalEntryNotFoundError":
        return OfflineError(f"Hugging Face asset {label} is not available in the local cache")
    if _is_missing_xet_transport_error(exc):
        return None
    if offline:
        return OfflineError(f"Hugging Face asset {label} could not be resolved while offline")
    return AssetDownloadError(f"Could not download Hugging Face asset {label} ({error_name})")


def _attach_xet_recovery_diagnostics(
    error: Exception,
    initial_error: Exception,
    retry_error: Exception,
    *,
    runtime_switch_activated: bool,
) -> None:
    error.__dict__.update(
        {
            "stage": "huggingface_download",
            "initial_error_type": type(initial_error).__name__,
            "native_error_type": type(retry_error).__name__,
            "retry_attempted": True,
            "retry_failed": True,
            "retry_error_type": type(retry_error).__name__,
            "retry_outcome": "failed",
            "runtime_switch_activated": runtime_switch_activated,
            "fallback_attempted": False,
            "fallback_failed": False,
            "fallback_error_type": None,
            "fallback_outcome": "not_attempted",
        }
    )


def _record_xet_recovery(
    initial_error: Exception,
    *,
    retry_outcome: str,
    retry_error_type: str | None,
    runtime_switch_activated: bool,
    fallback_attempted: bool,
    fallback_outcome: str,
    fallback_error_type: str | None = None,
) -> None:
    _LOGGER.info(
        "Hugging Face Xet recovery",
        extra={
            "recovery_stage": "huggingface_download",
            "initial_error_type": type(initial_error).__name__,
            "retry_attempted": True,
            "retry_outcome": retry_outcome,
            "retry_error_type": retry_error_type,
            "runtime_switch_activated": runtime_switch_activated,
            "direct_fallback_attempted": fallback_attempted,
            "direct_fallback_outcome": fallback_outcome,
            "direct_fallback_error_type": fallback_error_type,
        },
    )


def _recover_missing_xet(
    initial_error: Exception,
    *,
    source: HuggingFaceSource,
    offline: bool,
    local_dir: Path,
    hub: ModuleType,
    download_kwargs: _HubDownloadKwargs,
) -> Path:
    runtime_switch_activated = _disable_xet_for_current_process(hub)
    try:
        result = Path(hub.hf_hub_download(**download_kwargs))
        if not result.is_file():
            raise AssetDownloadError("Hugging Face HTTP-mode retry returned a missing file")
        _record_xet_recovery(
            initial_error,
            retry_outcome="succeeded",
            retry_error_type=None,
            runtime_switch_activated=runtime_switch_activated,
            fallback_attempted=False,
            fallback_outcome="not_attempted",
        )
        return result
    except Exception as exc:
        retry_error = exc

    classified = _classified_huggingface_error(retry_error, source=source, offline=offline)
    if isinstance(
        classified,
        (AssetAuthenticationError, AssetPermissionError, AssetNotFoundError, OfflineError),
    ):
        _record_xet_recovery(
            initial_error,
            retry_outcome="failed",
            retry_error_type=type(retry_error).__name__,
            runtime_switch_activated=runtime_switch_activated,
            fallback_attempted=False,
            fallback_outcome="not_attempted",
        )
        message = (
            f"The Hub Xet/hf_xet transport failed ({type(initial_error).__name__}); "
            f"the HTTP-mode retry was attempted and failed ({type(retry_error).__name__}); "
            "direct HTTPS fallback was not attempted. "
            f"{classified}"
        )
        recovered_error = type(classified)(message)
        _attach_xet_recovery_diagnostics(
            recovered_error,
            initial_error,
            retry_error,
            runtime_switch_activated=runtime_switch_activated,
        )
        raise recovered_error from None

    offline_mode = offline or _hub_offline_enabled()
    if offline_mode:
        _record_xet_recovery(
            initial_error,
            retry_outcome="failed",
            retry_error_type=type(retry_error).__name__,
            runtime_switch_activated=runtime_switch_activated,
            fallback_attempted=False,
            fallback_outcome="not_attempted",
        )
        offline_error = OfflineError(
            f"Could not download Hugging Face asset {source.repository}@{source.revision}:{source.path} "
            f"(huggingface_download, {type(retry_error).__name__}; retry attempted=yes, failed=yes; "
            f"fallback attempted=no, failed=no): The Hub Xet/hf_xet transport failed and the HTTP-mode "
            "retry was attempted. Offline mode is local-cache-only; no direct HTTPS request was made."
        )
        _attach_xet_recovery_diagnostics(
            offline_error,
            initial_error,
            retry_error,
            runtime_switch_activated=runtime_switch_activated,
        )
        raise offline_error from None

    if source.gated:
        _record_xet_recovery(
            initial_error,
            retry_outcome="failed",
            retry_error_type=type(retry_error).__name__,
            runtime_switch_activated=runtime_switch_activated,
            fallback_attempted=False,
            fallback_outcome="not_attempted",
        )
        raise _xet_retry_transport_error(
            source,
            initial_error,
            retry_error,
            runtime_switch_activated=runtime_switch_activated,
            fallback_attempted=False,
            fallback_error=None,
            summary=(
                "The source is marked gated, so no unauthenticated direct HTTPS fallback was "
                "attempted. Authenticate and retry."
            ),
        ) from None

    try:
        result = _download_xet_fallback(source, local_dir=local_dir)
    except Exception as fallback_error:
        _record_xet_recovery(
            initial_error,
            retry_outcome="failed",
            retry_error_type=type(retry_error).__name__,
            runtime_switch_activated=runtime_switch_activated,
            fallback_attempted=True,
            fallback_outcome="failed",
            fallback_error_type=type(fallback_error).__name__,
        )
        raise _xet_retry_transport_error(
            source,
            initial_error,
            retry_error,
            runtime_switch_activated=runtime_switch_activated,
            fallback_attempted=True,
            fallback_error=fallback_error,
            summary=_safe_fallback_summary(fallback_error),
        ) from None
    _record_xet_recovery(
        initial_error,
        retry_outcome="failed",
        retry_error_type=type(retry_error).__name__,
        runtime_switch_activated=runtime_switch_activated,
        fallback_attempted=True,
        fallback_outcome="succeeded",
    )
    return result


def _xet_retry_transport_error(
    source: HuggingFaceSource,
    initial_error: Exception,
    retry_error: Exception,
    *,
    runtime_switch_activated: bool,
    fallback_attempted: bool,
    fallback_error: Exception | None,
    summary: str,
) -> AssetTransportError:
    fallback_type = type(fallback_error).__name__ if fallback_error is not None else None
    initial_type = type(initial_error).__name__
    retry_type = type(retry_error).__name__
    label = f"{source.repository}@{source.revision}:{source.path}"
    fallback_state = "yes, failed" if fallback_error is not None else "no, not attempted"
    message = (
        f"Could not download Hugging Face asset {label} (xet_fallback, {retry_type}; "
        f"retry attempted=yes, failed=yes; direct HTTPS fallback attempted="
        f"{fallback_state}; initial Hub error={initial_type}): The Hub Xet/hf_xet transport failed, "
        f"the HTTP-mode retry failed, and {summary}"
    )
    if fallback_error is not None:
        stage = "xet_fallback"
        native_error_type = fallback_type or retry_type
    else:
        stage = "huggingface_download"
        native_error_type = retry_type
    return AssetTransportError(
        message,
        repository=source.repository,
        revision=source.revision,
        path=source.path,
        stage=stage,
        native_error_type=native_error_type,
        fallback_attempted=fallback_attempted,
        fallback_failed=fallback_error is not None,
        initial_error_type=initial_type,
        retry_attempted=True,
        retry_failed=True,
        retry_error_type=retry_type,
        runtime_switch_activated=runtime_switch_activated,
        fallback_error_type=fallback_type,
    )


def _disable_xet_for_current_process(hub: ModuleType) -> bool:
    """Disable Xet for this process and update Hub's loaded runtime switch if available."""
    os.environ["HF_HUB_DISABLE_XET"] = "1"
    try:
        constants = getattr(hub, "constants", None)
        if constants is None:
            constants = importlib.import_module("huggingface_hub.constants")
        if not hasattr(constants, "HF_HUB_DISABLE_XET"):
            return False
        constants.HF_HUB_DISABLE_XET = True
        return constants.HF_HUB_DISABLE_XET is True
    except Exception:
        return False


def _is_missing_xet_transport_error(exc: BaseException) -> bool:
    if not isinstance(exc, ValueError) and not type(exc).__module__.startswith("huggingface_hub."):
        return False
    if _http_status(exc) is not None:
        return False
    message = str(exc).casefold()
    if "xet" not in message or "hf_xet" not in message:
        return False
    return any(
        phrase in message
        for phrase in (
            "not installed",
            "not available",
            "unavailable",
            "missing",
            "not found",
            "could not import",
            "cannot import",
            "please install",
            "install hf_xet",
            "requires hf_xet",
        )
    )


def _hub_offline_enabled() -> bool:
    return os.environ.get("HF_HUB_OFFLINE", "").casefold() in {"1", "true", "yes"}


def _download_xet_fallback(source: HuggingFaceSource, *, local_dir: Path) -> Path:
    relative_path = validate_relative_path(source.path, field_name="Hugging Face repository path")
    root = Path(local_dir)
    root.mkdir(parents=True, exist_ok=True)
    root = root.resolve()

    parts = relative_path.split("/")
    parent = root
    for part in parts[:-1]:
        candidate = parent / part
        with suppress(FileExistsError):
            candidate.mkdir()
        resolved = candidate.resolve()
        if not resolved.is_relative_to(root) or not resolved.is_dir():
            raise ValueError("Hugging Face fallback path escapes local_dir")
        parent = resolved
    destination = parent / parts[-1]

    url = (
        "https://huggingface.co/"
        f"{quote(source.repository, safe='/')}/resolve/"
        f"{quote(source.revision, safe='')}/"
        f"{quote(relative_path, safe='/')}"
    )
    request = Request(url, headers={"User-Agent": "onnxvoice"})
    descriptor, temporary_name = tempfile.mkstemp(prefix=".onnxvoice-", dir=parent)
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as output, urlopen(request, timeout=30) as response:
            while chunk := response.read(64 * 1024):
                output.write(chunk)
        os.replace(temporary_path, destination)
        if not destination.is_file() or not destination.resolve().is_relative_to(root):
            raise OSError("Hugging Face fallback did not produce a file beneath local_dir")
        return destination
    finally:
        with suppress(FileNotFoundError):
            temporary_path.unlink()


def _safe_fallback_summary(exc: Exception) -> str:
    status = _http_status(exc)
    if status is None:
        code = getattr(exc, "code", None)
        status = code if isinstance(code, int) else None
    if status in {401, 403}:
        return "The pinned HTTPS request was denied by the server."
    if status == 404:
        return "The pinned HTTPS file was not found."
    if status is not None:
        return f"The pinned HTTPS request returned HTTP status {status}."
    reason = getattr(exc, "reason", None)
    if isinstance(exc, (TimeoutError, socket.timeout)) or isinstance(
        reason, (TimeoutError, socket.timeout)
    ):
        return "The pinned HTTPS request timed out."
    if isinstance(exc, (HTTPError, URLError)):
        return "The pinned HTTPS request could not retrieve the file."
    return "The pinned HTTPS fallback could not retrieve or write the file."


def _hub_error_types() -> dict[str, tuple[type[BaseException], ...]]:
    modules: list[ModuleType] = []
    for name in ("huggingface_hub.errors", "huggingface_hub.utils"):
        try:
            modules.append(importlib.import_module(name))
        except ImportError:
            continue

    names = {
        "gated": ("GatedRepoError",),
        "offline": ("OfflineModeIsEnabled",),
        "local_entry": ("LocalEntryNotFoundError",),
        "missing": (
            "RepositoryNotFoundError",
            "RevisionNotFoundError",
            "RemoteEntryNotFoundError",
            "EntryNotFoundError",
        ),
    }
    return {
        key: tuple(
            error_type
            for module in modules
            for name in type_names
            if isinstance((error_type := getattr(module, name, None)), type)
            and issubclass(error_type, BaseException)
        )
        for key, type_names in names.items()
    }


def _is_instance(exc: BaseException, classes: tuple[type[BaseException], ...]) -> bool:
    return bool(classes) and isinstance(exc, classes)


def _http_status(exc: BaseException) -> int | None:
    response: Any = getattr(exc, "response", None)
    for status in (
        getattr(response, "status_code", None),
        getattr(exc, "status_code", None),
        getattr(exc, "code", None),
    ):
        if isinstance(status, int):
            return status
    return None


def huggingface_diagnostics(*, offline: bool = False) -> dict[str, str | bool]:
    try:
        importlib.import_module("huggingface_hub")
    except ModuleNotFoundError as exc:
        hub_status = (
            "not installed"
            if exc.name == "huggingface_hub"
            else f"import failed ({type(exc).__name__})"
        )
    except ImportError as exc:
        hub_status = f"import failed ({type(exc).__name__})"
    else:
        hub_status = "installed"

    token_path = os.environ.get("HF_TOKEN_PATH")
    if token_path is None:
        hf_home = os.environ.get("HF_HOME")
        if hf_home is not None:
            token_path = str(Path(hf_home).expanduser() / "token")
        else:
            cache_home = os.environ.get("XDG_CACHE_HOME")
            base = Path(cache_home).expanduser() if cache_home else Path.home() / ".cache"
            token_path = str(base / "huggingface" / "token")
    credentials = bool(os.environ.get("HF_TOKEN")) or Path(token_path).expanduser().is_file()
    offline_environment = os.environ.get("HF_HUB_OFFLINE", "").casefold() in {
        "1",
        "true",
        "yes",
    }
    implicit_token_disabled = os.environ.get("HF_HUB_DISABLE_IMPLICIT_TOKEN", "").casefold() in {
        "1",
        "true",
        "yes",
    }
    return {
        "huggingface_hub": hub_status,
        "credentials": "configured" if credentials else "not configured",
        "offline": offline or offline_environment,
        "implicit_token_disabled": implicit_token_disabled,
        "gated_access": "not checked; diagnostic makes no network request",
    }
