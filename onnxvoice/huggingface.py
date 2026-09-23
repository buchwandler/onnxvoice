from __future__ import annotations

import importlib
import os
import re
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

from .errors import (
    AssetAuthenticationError,
    AssetDownloadError,
    AssetNotFoundError,
    AssetPermissionError,
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
    except ImportError as exc:
        raise OptionalDependencyError(
            "Remote Pocket downloads require Hugging Face support. Install 'onnxvoice[pocket]'."
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
        _raise_huggingface_error(exc, source=source, offline=offline)

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
) -> None:
    error_types = _hub_error_types()
    status = _http_status(exc)
    label = f"{source.repository}@{source.revision}:{source.path}"

    if _is_instance(exc, error_types["gated"]) or status == 403:
        raise AssetPermissionError(
            f"Pocket asset {label} requires access to a gated Hugging Face repository. "
            "Accept or request access on the repository page, authenticate with "
            "'hf auth login' or HF_TOKEN, verify with 'hf auth whoami', and retry."
        ) from exc
    if status == 401:
        raise AssetAuthenticationError(
            f"Authentication is required to download Pocket asset {label}. "
            "Authenticate with 'hf auth login' or configure HF_TOKEN, then retry."
        ) from exc
    if _is_instance(exc, error_types["offline"]) or (
        offline and _is_instance(exc, error_types["local_entry"])
    ):
        raise OfflineError(
            f"Pocket asset {label} is not available in the local Hugging Face cache"
        ) from exc
    if _is_instance(exc, error_types["missing"]) or status == 404:
        raise AssetNotFoundError(f"Hugging Face asset not found: {label}") from exc
    if offline:
        raise OfflineError(f"Pocket asset {label} could not be resolved while offline") from exc

    error_name = type(exc).__name__
    if error_name in {"GatedRepoError"}:
        raise AssetPermissionError(
            f"Pocket asset {label} requires access to a gated Hugging Face repository. "
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
            f"Pocket asset {label} is not available in the local Hugging Face cache"
        ) from exc

    raise AssetDownloadError(
        f"Could not download Pocket asset {label} from Hugging Face ({error_name})"
    ) from exc


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
    except ImportError:
        installed = False
    else:
        installed = True

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
        "huggingface_hub": "installed" if installed else "not installed",
        "credentials": "configured" if credentials else "not configured",
        "offline": offline or offline_environment,
        "implicit_token_disabled": implicit_token_disabled,
        "gated_access": "not checked; diagnostic makes no network request",
    }
