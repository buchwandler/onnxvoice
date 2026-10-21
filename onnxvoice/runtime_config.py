"""Persistent, user-level ONNXVoice runtime preferences.

This module intentionally has no dependency on ONNX Runtime so configuration can be
inspected, validated, and reset independently of the optional inference package.
"""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Sequence
from contextlib import suppress
from pathlib import Path

from platformdirs import user_config_path

from .errors import RuntimeContractError

_SCHEMA_VERSION = 1


def runtime_config_path() -> Path:
    """Return the path to the user-level runtime preference file."""
    return Path(user_config_path("onnxvoice")) / "runtime.json"


def _config_path(path: Path | str | None) -> Path:
    return Path(path) if path is not None else runtime_config_path()


def _validated_providers(value: object, *, source: str) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)) or not value:
        raise RuntimeContractError(
            f"Invalid ONNXVoice runtime configuration at {source}: providers must be a non-empty list"
        )
    providers: list[str] = []
    for provider in value:
        if not isinstance(provider, str) or not provider.strip():
            raise RuntimeContractError(
                f"Invalid ONNXVoice runtime configuration at {source}: "
                "providers must contain only non-empty strings"
            )
        providers.append(provider.strip())
    return tuple(providers)


def load_runtime_config(path: Path | str | None = None) -> tuple[str, ...] | None:
    """Load the saved ordered provider preference, or return ``None`` if absent.

    Malformed JSON, unsupported schema versions, and invalid provider lists raise a
    clear :class:`RuntimeContractError` so normal provider resolution cannot silently
    ignore a broken user preference.
    """
    config_path = _config_path(path)
    try:
        with config_path.open("r", encoding="utf-8") as stream:
            value = json.load(stream)
    except FileNotFoundError:
        return None
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RuntimeContractError(
            f"Could not read ONNXVoice runtime configuration at {config_path}: "
            f"{type(exc).__name__}: {exc}"
        ) from exc

    if not isinstance(value, dict):
        raise RuntimeContractError(
            f"Invalid ONNXVoice runtime configuration at {config_path}: expected a JSON object"
        )
    schema = value.get("schema")
    if type(schema) is not int or schema != _SCHEMA_VERSION:
        raise RuntimeContractError(
            f"Unsupported ONNXVoice runtime configuration schema at {config_path}: {schema!r}"
        )
    return _validated_providers(value.get("providers"), source=str(config_path))


def save_runtime_config(providers: Sequence[str], path: Path | str | None = None) -> Path:
    """Atomically save an ordered provider preference and return its path."""
    config_path = _config_path(path)
    provider_names = _validated_providers(providers, source=str(config_path))
    payload = {"schema": _SCHEMA_VERSION, "providers": list(provider_names)}
    payload = {"schema": _SCHEMA_VERSION, "providers": list(provider_names)}
    temporary_path: Path | None = None
    try:
        config_path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{config_path.name}.", suffix=".tmp", dir=config_path.parent
        )
        temporary_path = Path(temporary_name)
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, ensure_ascii=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, config_path)
        temporary_path = None
    except OSError as exc:
        raise RuntimeContractError(
            f"Could not write ONNXVoice runtime configuration at {config_path}: "
            f"{type(exc).__name__}: {exc}"
        ) from exc
    finally:
        if temporary_path is not None:
            with suppress(OSError):
                temporary_path.unlink(missing_ok=True)
    return config_path


def clear_runtime_config(path: Path | str | None = None) -> None:
    """Remove the saved preference; malformed file contents do not prevent reset."""
    config_path = _config_path(path)
    try:
        config_path.unlink(missing_ok=True)
    except OSError as exc:
        raise RuntimeContractError(
            f"Could not remove ONNXVoice runtime configuration at {config_path}: "
            f"{type(exc).__name__}: {exc}"
        ) from exc
