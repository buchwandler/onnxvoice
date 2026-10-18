"""Build and verify pinned native KittenTTS 2 bundle catalogs."""

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

from ..catalog import _parse_kitten2
from ..errors import CatalogError
from ..types import validate_relative_path

DEFAULT_REPOSITORY = "holgern/kitten-tts-v2-onnx"
DEFAULT_REVISION = "main"
DEFAULT_BUNDLE_ID = "kitten-tts-v2"
DEFAULT_LICENSE = "stellon-labs-community-license"
USER_AGENT = "onnxvoice-kitten2-catalog"
_SHA_RE = re.compile(r"^[0-9a-f]{40}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_REPOSITORY_RE = re.compile(r"^[^/\\\s]+/[^/\\\s]+$")
_SAFE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise CatalogError(message)


def _repository(repository: str) -> str:
    _require(
        isinstance(repository, str) and _REPOSITORY_RE.fullmatch(repository) is not None,
        f"Invalid Hugging Face repository: {repository!r}",
    )
    return repository


def huggingface_api_url(repository: str, revision: str) -> str:
    repository = _repository(repository)
    repo = "/".join(urllib.parse.quote(part, safe="") for part in repository.split("/"))
    rev = urllib.parse.quote(revision, safe="")
    return f"https://huggingface.co/api/models/{repo}/revision/{rev}"


def huggingface_resolve_url(repository: str, revision: str, path: str) -> str:
    repository = _repository(repository)
    repo = "/".join(urllib.parse.quote(part, safe="") for part in repository.split("/"))
    rev = urllib.parse.quote(revision, safe="")
    asset = urllib.parse.quote(path, safe="/")
    return f"https://huggingface.co/{repo}/resolve/{rev}/{asset}?download=true"


def _read_url(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            return response.read()
    except (OSError, urllib.error.URLError) as exc:
        raise CatalogError(f"Unable to fetch {url}") from exc


def resolve_revision(repository: str, revision: str) -> str:
    """Resolve a branch or tag to an immutable lowercase Hugging Face commit SHA."""
    _repository(repository)
    _require(isinstance(revision, str) and bool(revision), "Source revision is required")
    if _SHA_RE.fullmatch(revision):
        return revision
    try:
        data = json.loads(_read_url(huggingface_api_url(repository, revision)).decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CatalogError("Hugging Face repository metadata is not valid JSON") from exc
    resolved = data.get("sha") if isinstance(data, Mapping) else None
    _require(
        isinstance(resolved, str) and _SHA_RE.fullmatch(resolved) is not None,
        "Unable to resolve an immutable 40-character source revision",
    )
    return cast(str, resolved)


def build_catalog(
    *,
    repository: str = DEFAULT_REPOSITORY,
    revision: str = DEFAULT_REVISION,
    bundle_id: str = DEFAULT_BUNDLE_ID,
    license: str = DEFAULT_LICENSE,
) -> dict[str, Any]:
    """Build a catalog entry from a bundle manifest at an immutable repository revision."""
    repository = _repository(repository)
    _require(_SAFE_ID_RE.fullmatch(bundle_id) is not None, f"Unsafe bundle ID: {bundle_id!r}")
    _require(isinstance(license, str) and bool(license), "Source license is required")
    resolved_revision = resolve_revision(repository, revision)
    bundle_bytes = _read_url(huggingface_resolve_url(repository, resolved_revision, "bundle.json"))
    try:
        manifest = json.loads(bundle_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CatalogError("Kitten2 bundle.json is not valid JSON") from exc
    _require(isinstance(manifest, Mapping), "Kitten2 bundle.json must be an object")
    layout = manifest.get("format")
    sample_rate = manifest.get("sample_rate")
    _require(layout == "kitten2-onnx-bundle-v1", "Unsupported Kitten2 bundle layout")
    _require(type(sample_rate) is int and sample_rate == 24000, "Kitten2 sample_rate must be 24000")
    files = manifest.get("files")
    _require(isinstance(files, Mapping) and bool(files), "Kitten2 bundle files must be a non-empty object")
    artifacts = [
        {
            "role": "bundle",
            "path": "bundle.json",
            "sha256": hashlib.sha256(bundle_bytes).hexdigest(),
        }
    ]
    for raw_path, sha256 in sorted(files.items()):
        _require(isinstance(raw_path, str), "Kitten2 bundle paths must be strings")
        try:
            path = validate_relative_path(raw_path, field_name="Kitten2 artifact path")
        except ValueError as exc:
            raise CatalogError(str(exc)) from exc
        _require(
            isinstance(sha256, str) and _SHA256_RE.fullmatch(sha256) is not None,
            f"Invalid SHA-256 for Kitten2 artifact {path!r}",
        )
        artifacts.append({"role": "asset", "path": path, "sha256": sha256})

    voices = manifest.get("voices")
    _require(isinstance(voices, Mapping) and bool(voices), "Kitten2 voices must be a non-empty object")
    default_voice = manifest.get("default_voice")
    _require(
        isinstance(default_voice, str) and default_voice in voices,
        "Kitten2 default_voice must name a declared voice",
    )
    record = {
        "id": bundle_id,
        "aliases": ["v2", "default"],
        "name": "KittenTTS 2 ONNX",
        "layout": layout,
        "sample_rate": sample_rate,
        "default_voice": default_voice,
        "voices": sorted(voices),
        "source": {
            "provider": "huggingface",
            "repository": repository,
            "revision": resolved_revision,
            "license": license,
        },
        "runtime": {"bundle_format": layout},
        "metadata": {"default_preset": manifest.get("default_preset", "stable")},
        "artifacts": artifacts,
    }
    catalog = {
        "schema": 1,
        "kind": "kitten2-onnx-bundle-catalog",
        "bundles": {bundle_id: record},
    }
    _parse_kitten2(catalog)
    return catalog


def build_source(catalog: Mapping[str, Any]) -> dict[str, Any]:
    """Derive a deterministic source snapshot from a validated pinned catalog."""
    _parse_kitten2(catalog)
    repositories = []
    for bundle_id, bundle in sorted(catalog["bundles"].items()):
        source = bundle["source"]
        repositories.append(
            {
                "id": bundle_id,
                "repository": source["repository"],
                "revision": source["revision"],
                "license": source["license"],
            }
        )
    return {
        "schema": 1,
        "kind": "kitten2-onnx-source-snapshot",
        "repositories": repositories,
    }


def verify_catalog(catalog: Mapping[str, Any], source: Mapping[str, Any] | None = None) -> None:
    """Validate a pinned catalog and, when supplied, its offline source snapshot."""
    _parse_kitten2(catalog)
    if source is not None and dict(source) != build_source(catalog):
        raise CatalogError("Source metadata does not match catalog provenance")


def load_catalog(
    catalog_path: str | Path,
    source_path: str | Path | None = None,
) -> dict[str, Any]:
    """Load and verify catalog files without network access."""
    try:
        catalog = json.loads(Path(catalog_path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CatalogError(f"Unable to load Kitten2 catalog: {catalog_path}") from exc
    source = None
    if source_path is not None:
        try:
            source = json.loads(Path(source_path).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise CatalogError(f"Unable to load Kitten2 source snapshot: {source_path}") from exc
    if not isinstance(catalog, Mapping):
        raise CatalogError("Kitten2 catalog must be an object")
    if source is not None and not isinstance(source, Mapping):
        raise CatalogError("Kitten2 source snapshot must be an object")
    verify_catalog(catalog, source)
    return dict(catalog)
