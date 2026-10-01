"""Build and verify pinned KittenTTS ONNX model catalogs."""

from __future__ import annotations

import hashlib
import json
import re
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

from ..catalog import _parse_kitten
from ..errors import CatalogError as OnnxCatalogError

DEFAULT_REVISION = "main"
USER_AGENT = "onnxvoice/0.2.0"
_SHA_RE = re.compile(r"^[0-9a-f]{40}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_SAFE_FILENAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
_SOURCE_NOTE = "Generated deterministically by OnnxVoice catalog kitten build."


class CatalogError(ValueError):
    """Raised when upstream metadata or a normalized catalog is invalid."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise CatalogError(message)


def huggingface_api_url(repository: str, revision: str) -> str:
    repo = "/".join(urllib.parse.quote(part, safe="") for part in repository.split("/"))
    return (
        f"https://huggingface.co/api/models/{repo}/revision/{urllib.parse.quote(revision, safe='')}"
    )


def huggingface_resolve_url(repository: str, revision: str, filename: str) -> str:
    repo = "/".join(urllib.parse.quote(part, safe="") for part in repository.split("/"))
    name = urllib.parse.quote(filename, safe="")
    return f"https://huggingface.co/{repo}/resolve/{revision}/{name}"


def _read_url(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            return response.read()
    except (OSError, urllib.error.URLError) as exc:
        raise CatalogError(f"Unable to fetch {url}") from exc


def _decode_json(payload: bytes, label: str) -> Any:
    try:
        return json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CatalogError(f"{label} is not valid UTF-8 JSON") from exc


def resolve_revision(repository: str, revision: str = DEFAULT_REVISION) -> str:
    """Resolve a branch or tag to its exact lowercase commit SHA."""
    if _SHA_RE.fullmatch(revision):
        return revision
    metadata = _decode_json(
        _read_url(huggingface_api_url(repository, revision)), "Repository metadata"
    )
    resolved = metadata.get("sha") if isinstance(metadata, Mapping) else None
    _require(
        isinstance(resolved, str) and _SHA_RE.fullmatch(resolved) is not None,
        "Unable to resolve an exact 40-character commit SHA",
    )
    return cast(str, resolved)


def _tree_url(repository: str, revision: str) -> str:
    repo = "/".join(urllib.parse.quote(part, safe="") for part in repository.split("/"))
    return f"https://huggingface.co/api/models/{repo}/tree/{revision}?recursive=true&expand=true"


def _repository_tree(repository: str, revision: str) -> dict[str, dict[str, Any]]:
    url = _tree_url(repository, revision)
    entries: dict[str, dict[str, Any]] = {}
    while url:
        request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                payload = response.read()
                next_link = response.headers.get("Link")
        except (OSError, urllib.error.URLError) as exc:
            raise CatalogError(f"Unable to fetch repository tree for {revision}") from exc
        page = _decode_json(payload, "Repository tree")
        _require(isinstance(page, list), "Repository tree must be a list")
        for item in page:
            if not isinstance(item, Mapping) or not isinstance(item.get("path"), str):
                continue
            path = item["path"]
            _require(path not in entries, f"Repository tree contains duplicate path {path!r}")
            entries[path] = dict(item)
        match = re.search(r'<([^>]+)>;\s*rel="next"', next_link or "")
        url = match.group(1) if match else ""
    return entries


def _positive_size(value: Any, label: str) -> int | None:
    if value is None:
        return None
    _require(
        isinstance(value, int) and not isinstance(value, bool) and value > 0,
        f"{label}: invalid size",
    )
    return value


def _artifact_integrity(
    repository: str,
    revision: str,
    filename: str,
    metadata: Mapping[str, Any],
) -> tuple[int, str]:
    size = _positive_size(metadata.get("size"), filename)
    lfs = metadata.get("lfs")
    sha256 = lfs.get("oid") if isinstance(lfs, Mapping) else None
    if sha256 is not None:
        _require(
            isinstance(sha256, str) and _SHA256_RE.fullmatch(sha256) is not None,
            f"{filename}: invalid LFS SHA-256",
        )
    if size is not None and isinstance(sha256, str):
        return size, sha256

    digest = hashlib.sha256()
    downloaded_size = 0
    url = huggingface_resolve_url(repository, revision, filename)
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            while chunk := response.read(1024 * 1024):
                digest.update(chunk)
                downloaded_size += len(chunk)
    except (OSError, urllib.error.URLError) as exc:
        raise CatalogError(f"Unable to fetch artifact integrity data for {filename}") from exc
    if size is not None:
        _require(
            downloaded_size == size,
            f"{filename}: downloaded size does not match repository metadata",
        )
    if isinstance(sha256, str) and _SHA256_RE.fullmatch(sha256):
        _require(
            digest.hexdigest() == sha256, f"{filename}: LFS SHA-256 does not match downloaded bytes"
        )
    return downloaded_size, digest.hexdigest()


def _config_text(config: Mapping[str, Any], field: str) -> str:
    value = config.get(field)
    _require(
        isinstance(value, str) and bool(value.strip()),
        f"config.json: {field} must be a non-empty string",
    )
    return cast(str, value)


def _config_aliases(config: Mapping[str, Any]) -> dict[str, str]:
    value = config.get("voice_aliases")
    if not isinstance(value, Mapping) or not value:
        raise CatalogError("config.json: voice_aliases must be a non-empty object")
    aliases: dict[str, str] = {}
    for display_name, internal_id in value.items():
        _require(
            isinstance(display_name, str)
            and _SAFE_FILENAME_RE.fullmatch(display_name) is not None
            and isinstance(internal_id, str)
            and _SAFE_FILENAME_RE.fullmatch(internal_id) is not None,
            "config.json: voice_aliases contains an unsafe name",
        )
        aliases[display_name] = internal_id
    return aliases


def _config_speed_priors(config: Mapping[str, Any], aliases: Mapping[str, str]) -> dict[str, float]:
    value = config.get("speed_priors")
    if not isinstance(value, Mapping):
        raise CatalogError("config.json: speed_priors must be an object")
    internal_ids = set(aliases.values())
    priors: dict[str, float] = {}
    for internal_id, speed in value.items():
        _require(
            internal_id in internal_ids,
            f"config.json: speed prior refers to unknown voice {internal_id!r}",
        )
        _require(
            isinstance(speed, (int, float)) and not isinstance(speed, bool),
            f"config.json: invalid speed prior for {internal_id!r}",
        )
        try:
            speed_value = float(speed)
        except OverflowError as exc:
            raise CatalogError(f"config.json: invalid speed prior for {internal_id!r}") from exc
        _require(
            speed_value > 0 and speed_value < float("inf"),
            f"config.json: invalid speed prior for {internal_id!r}",
        )
        priors[cast(str, internal_id)] = speed_value
    return priors


def build_catalog(seed_catalog: Any, *, revision: str = DEFAULT_REVISION) -> dict[str, Any]:
    """Refresh a normalized seed catalog from its configured Hugging Face repositories."""
    _require(isinstance(seed_catalog, Mapping), "Seed catalog must be an object")
    try:
        _parse_kitten(seed_catalog)
    except (TypeError, ValueError, OnnxCatalogError) as exc:
        raise CatalogError(str(exc)) from exc

    models: dict[str, Any] = {}
    for model_id in sorted(seed_catalog["models"]):
        seed_model = seed_catalog["models"][model_id]
        upstream_seed = seed_model["upstream"]
        repository = upstream_seed["repository"]
        actual_revision = resolve_revision(repository, revision)
        tree = _repository_tree(repository, actual_revision)
        config_path = "config.json"
        _require(config_path in tree, f"{repository}: missing upstream {config_path}")
        config_payload = _read_url(
            huggingface_resolve_url(repository, actual_revision, config_path)
        )
        config = _decode_json(config_payload, f"{repository}/{config_path}")
        _require(isinstance(config, Mapping), f"{repository}/{config_path} must contain an object")

        name = _config_text(config, "name")
        version = _config_text(config, "version")
        upstream_model = _config_text(config, "model")
        model_filename = _config_text(config, "model_file")
        voices_filename = _config_text(config, "voices")
        runtime_profile = _config_text(config, "type")
        _require(
            runtime_profile in {"ONNX1", "ONNX2"},
            f"{repository}: unsupported runtime type {runtime_profile!r}",
        )
        _require(
            _SAFE_FILENAME_RE.fullmatch(model_filename) is not None
            and model_filename.endswith(".onnx"),
            f"{repository}: invalid model_file in config.json",
        )
        _require(
            voices_filename == "voices.npz", f"{repository}: config.json voices must be voices.npz"
        )
        expected_model = seed_model["metadata"].get("upstream_model")
        if expected_model is not None:
            _require(
                upstream_model == expected_model,
                f"{repository}: config model identity {upstream_model!r} does not match seed {expected_model!r}",
            )

        voice_aliases = _config_aliases(config)
        speed_priors = _config_speed_priors(config, voice_aliases)
        artifacts: list[dict[str, Any]] = []
        for role, filename, format_name in (
            ("model", model_filename, "onnx"),
            ("voices", voices_filename, "npz"),
        ):
            _require(filename in tree, f"{repository}: missing upstream artifact {filename}")
            size, sha256 = _artifact_integrity(
                repository, actual_revision, filename, tree[filename]
            )
            artifacts.append(
                {
                    "role": role,
                    "format": format_name,
                    "filename": filename,
                    "url": huggingface_resolve_url(repository, actual_revision, filename),
                    "size": size,
                    "sha256": sha256,
                }
            )

        model = dict(seed_model)
        metadata = dict(seed_model["metadata"])
        metadata["upstream_model"] = upstream_model
        model.update(
            {
                "name": name,
                "version": version,
                "runtime": {"profile": runtime_profile},
                "upstream": {
                    **upstream_seed,
                    "revision": actual_revision,
                },
                "artifacts": artifacts,
                "voice_aliases": voice_aliases,
                "speed_priors": speed_priors,
                "metadata": metadata,
            }
        )
        models[model_id] = model

    catalog = {"schema": 1, "kind": "kitten-onnx-model-catalog", "models": models}
    verify_catalog(catalog)
    return catalog


def build_source(catalog: Mapping[str, Any]) -> dict[str, Any]:
    """Create deterministic companion provenance for a normalized catalog."""
    verify_catalog(catalog)
    repositories = {
        model["upstream"]["repository"]: {
            "repository": model["upstream"]["repository"],
            "revision": model["upstream"]["revision"],
            "license": model["upstream"]["license"],
        }
        for model in catalog["models"].values()
    }
    return {
        "schema": 1,
        "kind": "kitten-onnx-source-snapshot",
        "note": _SOURCE_NOTE,
        "repositories": [repositories[key] for key in sorted(repositories)],
    }


def verify_catalog(catalog: Any, source: Any | None = None) -> None:
    """Verify catalog integrity and optional companion source provenance."""
    _require(isinstance(catalog, Mapping), "Catalog must be an object")
    try:
        _parse_kitten(catalog)
    except (TypeError, ValueError, OnnxCatalogError) as exc:
        raise CatalogError(str(exc)) from exc
    if source is None:
        return

    _require(isinstance(source, Mapping), "Source metadata must be an object")
    _require(source.get("schema") == 1, "Source schema must be 1")
    _require(source.get("kind") == "kitten-onnx-source-snapshot", "Unexpected source metadata kind")
    if "note" in source:
        _require(isinstance(source["note"], str), "Source note must be text")
    if "snapshot_date" in source:
        _require(
            isinstance(source["snapshot_date"], str)
            and re.fullmatch(r"\d{4}-\d{2}-\d{2}", source["snapshot_date"]) is not None,
            "Source snapshot_date must use YYYY-MM-DD",
        )
    expected = build_source(catalog)
    actual_repositories = source.get("repositories")
    _require(isinstance(actual_repositories, list), "Source repositories must be a list")
    expected_by_repository = {item["repository"]: item for item in expected["repositories"]}
    actual_by_repository: dict[str, Any] = {}
    for item in actual_repositories:
        _require(isinstance(item, Mapping), "Source repository entry must be an object")
        repository = item.get("repository")
        _require(isinstance(repository, str), "Source repository entry must name a repository")
        _require(
            repository not in actual_by_repository, f"Duplicate source repository {repository!r}"
        )
        actual_by_repository[repository] = {
            "repository": repository,
            "revision": item.get("revision"),
            "license": item.get("license"),
        }
    _require(
        actual_by_repository == expected_by_repository,
        "Source repositories do not match catalog provenance",
    )


def load_catalog(path: Path, source: Path | None = None) -> dict[str, Any]:
    """Read and verify a catalog, optionally checking its companion source.json."""
    try:
        catalog = json.loads(path.read_text(encoding="utf-8"))
        source_data = json.loads(source.read_text(encoding="utf-8")) if source is not None else None
    except (OSError, json.JSONDecodeError) as exc:
        raise CatalogError(f"Unable to read Kitten catalog: {path}") from exc
    verify_catalog(catalog, source_data)
    return cast(dict[str, Any], catalog)
