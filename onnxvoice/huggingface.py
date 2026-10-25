from __future__ import annotations

import importlib
import os
import re
import socket
import tempfile
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any
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
    try:
        downloaded = hub.hf_hub_download(
            repo_id=source.repository,
            filename=source.path,
            revision=source.revision,
            local_dir=str(local_dir),
            local_files_only=offline,
            token=None,
            library_name="onnxvoice",
        )
    except Exception as exc:
        return _raise_huggingface_error(exc, source=source, offline=offline, local_dir=local_dir)

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
) -> Path:
    error_types = _hub_error_types()
    status = _http_status(exc)
    label = f"{source.repository}@{source.revision}:{source.path}"

    if _is_instance(exc, error_types["gated"]) or status == 403:
        raise AssetPermissionError(
            f"Hugging Face asset {label} requires access to a gated repository. "
            "Accept or request access on the repository page, authenticate with "
            "'hf auth login' or HF_TOKEN, verify with 'hf auth whoami', and retry."
        ) from exc
    if status == 401:
        raise AssetAuthenticationError(
            f"Authentication is required to download Hugging Face asset {label}. "
            "Authenticate with 'hf auth login' or configure HF_TOKEN, then retry."
        ) from exc
    if _is_instance(exc, error_types["offline"]) or (
        offline and _is_instance(exc, error_types["local_entry"])
    ):
        raise OfflineError(
            f"Hugging Face asset {label} is not available in the local cache"
        ) from exc
    if _is_instance(exc, error_types["missing"]) or status == 404:
        raise AssetNotFoundError(f"Hugging Face asset not found: {label}") from exc

    error_name = type(exc).__name__
    if error_name == "GatedRepoError":
        raise AssetPermissionError(
            f"Hugging Face asset {label} requires access to a gated repository. "
            "Accept or request access on the repository page, authenticate with "
            "'hf auth login' or HF_TOKEN, verify with 'hf auth whoami', and retry."
        ) from exc
    if error_name in {
        "RepositoryNotFoundError",
        "RevisionNotFoundError",
        "EntryNotFoundError",
        "RemoteEntryNotFoundError",
    }:
        raise AssetNotFoundError(f"Hugging Face asset not found: {label}") from exc
    if error_name == "LocalEntryNotFoundError":
        raise OfflineError(
            f"Hugging Face asset {label} is not available in the local cache"
        ) from exc

    if _is_missing_xet_transport_error(exc):
        limitation = (
            "Hugging Face Hub requires the optional hf_xet transport, but it is unavailable."
        )
        if offline or _hub_offline_enabled():
            raise OfflineError(
                f"Could not download Hugging Face asset {label} "
                f"(huggingface_download, {error_name}; fallback attempted=no, failed=no): "
                f"{limitation} Offline mode is local-cache-only; connect to the network or "
                "install a usable hf_xet transport, then retry."
            ) from exc
        if source.gated:
            raise AssetTransportError(
                f"Could not download Hugging Face asset {label} "
                f"(huggingface_download, {error_name}; fallback attempted=no, failed=no): "
                f"{limitation} The source is marked gated, so no unauthenticated HTTPS "
                "fallback was attempted. Authenticate and install a usable hf_xet transport, "
                "then retry.",
                repository=source.repository,
                revision=source.revision,
                path=source.path,
                stage="huggingface_download",
                native_error_type=error_name,
                fallback_attempted=False,
                fallback_failed=False,
            ) from exc
        try:
            return _download_xet_fallback(source, local_dir=local_dir)
        except Exception as fallback_exc:
            raise _xet_fallback_error(source, exc, fallback_exc) from fallback_exc

    if offline:
        raise OfflineError(
            f"Hugging Face asset {label} could not be resolved while offline"
        ) from exc

    raise AssetDownloadError(
        f"Could not download Hugging Face asset {label} ({error_name})"
    ) from exc


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


def _xet_fallback_error(
    source: HuggingFaceSource,
    hub_error: Exception,
    fallback_error: Exception,
) -> AssetTransportError:
    fallback_type = type(fallback_error).__name__
    hub_type = type(hub_error).__name__
    label = f"{source.repository}@{source.revision}:{source.path}"
    return AssetTransportError(
        f"Could not download Hugging Face asset {label} "
        f"(xet_fallback, {fallback_type}; fallback attempted=yes, failed=yes; "
        f"Hugging Face error={hub_type}): {_safe_fallback_summary(fallback_error)} "
        "Verify that the pinned public asset is reachable over HTTPS, then retry.",
        repository=source.repository,
        revision=source.revision,
        path=source.path,
        stage="xet_fallback",
        native_error_type=fallback_type,
        fallback_attempted=True,
        fallback_failed=True,
    )


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
    status = getattr(response, "status_code", None)
    return status if isinstance(status, int) else None


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
