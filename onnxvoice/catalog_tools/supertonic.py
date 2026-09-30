"""Build and verify pinned Supertonic-3 ONNX bundle catalogs."""

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

from ..catalog import ST_ARTIFACT_ROLES, ST_MODEL_COMPONENTS, _parse_supertonic
from ..errors import CatalogError as OnnxCatalogError

DEFAULT_REPOSITORY = "supertone-oss-archive/supertonic-3"
DEFAULT_REVISION = "main"
CURRENT_SNAPSHOT = "aafc6e32416a594460b32413efc49d7fe4ce6d46"
USER_AGENT = "onnxvoice/0.1.0"
_SHA_RE = re.compile(r"^[0-9a-f]{40}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_SAFE_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
_MODEL_PATHS = {
    "duration_predictor": "onnx/duration_predictor.onnx",
    "text_encoder": "onnx/text_encoder.onnx",
    "vector_estimator": "onnx/vector_estimator.onnx",
    "vocoder": "onnx/vocoder.onnx",
}
_SUPPORT_PATHS = {
    "config": "onnx/tts.json",
    "unicode_indexer": "onnx/unicode_indexer.json",
}
_CURRENT_STYLES = tuple([f"F{i}" for i in range(1, 6)] + [f"M{i}" for i in range(1, 6)])
_LANGUAGES = (
    "en",
    "ko",
    "ja",
    "ar",
    "bg",
    "cs",
    "da",
    "de",
    "el",
    "es",
    "et",
    "fi",
    "fr",
    "hi",
    "hr",
    "hu",
    "id",
    "it",
    "lt",
    "lv",
    "nl",
    "pl",
    "pt",
    "ro",
    "ru",
    "sk",
    "sl",
    "sv",
    "tr",
    "uk",
    "vi",
    "na",
)


class CatalogError(ValueError):
    """Raised when upstream metadata or a normalized catalog is invalid."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise CatalogError(message)


def _safe_name(value: Any, label: str) -> str:
    _require(
        isinstance(value, str) and _SAFE_NAME_RE.fullmatch(value) is not None,
        f"{label}: unsafe name {value!r}",
    )
    return cast(str, value)


def huggingface_api_url(repository: str, revision: str) -> str:
    repo = "/".join(urllib.parse.quote(part, safe="") for part in repository.split("/"))
    return (
        f"https://huggingface.co/api/models/{repo}/revision/{urllib.parse.quote(revision, safe='')}"
    )


def huggingface_resolve_url(repository: str, revision: str, path: str) -> str:
    repo = "/".join(urllib.parse.quote(part, safe="") for part in repository.split("/"))
    asset = urllib.parse.quote(path, safe="/")
    return f"https://huggingface.co/{repo}/resolve/{revision}/{asset}?download=true"


def _read_url(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            return response.read()
    except (OSError, urllib.error.URLError) as exc:
        raise CatalogError(f"Unable to fetch {url}") from exc


def resolve_revision(repository: str, revision: str = DEFAULT_REVISION) -> str:
    """Resolve a branch or tag to its exact lowercase commit SHA."""
    if _SHA_RE.fullmatch(revision):
        return revision
    try:
        metadata = json.loads(_read_url(huggingface_api_url(repository, revision)))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CatalogError("Hugging Face repository metadata is not valid JSON") from exc
    resolved = metadata.get("sha") if isinstance(metadata, dict) else None
    _require(
        isinstance(resolved, str) and _SHA_RE.fullmatch(resolved) is not None,
        "Unable to resolve an exact 40-character commit SHA",
    )
    return cast(str, resolved)


def _tree_url(repository: str, revision: str) -> str:
    repo = "/".join(urllib.parse.quote(part, safe="") for part in repository.split("/"))
    rev = urllib.parse.quote(revision, safe="")
    return f"https://huggingface.co/api/models/{repo}/tree/{rev}?recursive=true&expand=true"


def _repository_tree(repository: str, revision: str) -> list[dict[str, Any]]:
    url = _tree_url(repository, revision)
    entries: list[dict[str, Any]] = []
    while url:
        request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                payload = response.read()
                next_link = response.headers.get("Link")
        except (OSError, urllib.error.URLError) as exc:
            raise CatalogError(f"Unable to fetch repository tree for {revision}") from exc
        try:
            page = json.loads(payload.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise CatalogError("Unable to parse repository tree") from exc
        _require(isinstance(page, list), "Repository tree must be a list")
        entries.extend(item for item in page if isinstance(item, dict))
        match = re.search(r'<([^>]+)>;\s*rel="next"', next_link or "")
        url = match.group(1) if match else ""
    return entries


def _file_integrity(
    repository: str,
    revision: str,
    path: str,
    metadata: Mapping[str, Any],
    *,
    payload: bytes | None = None,
) -> tuple[int, str]:
    size = metadata.get("size")
    if not isinstance(size, int) or isinstance(size, bool) or size <= 0:
        if payload is None:
            payload = _read_url(huggingface_resolve_url(repository, revision, path))
        size = len(payload)
    _require(
        isinstance(size, int) and not isinstance(size, bool) and size > 0, f"{path}: invalid size"
    )
    if payload is not None:
        _require(size == len(payload), f"{path}: remote size does not match tree metadata")

    lfs = metadata.get("lfs")
    lfs = metadata.get("lfs")
    sha256 = lfs.get("oid") if isinstance(lfs, Mapping) else None
    if not isinstance(sha256, str) or _SHA256_RE.fullmatch(sha256) is None:
        if payload is None:
            payload = _read_url(huggingface_resolve_url(repository, revision, path))
        sha256 = hashlib.sha256(payload).hexdigest()
    elif payload is not None:
        _require(
            hashlib.sha256(payload).hexdigest() == sha256,
            f"{path}: LFS SHA-256 does not match bytes",
        )
    _require(
        isinstance(sha256, str) and _SHA256_RE.fullmatch(sha256) is not None,
        f"{path}: invalid SHA-256",
    )
    return cast(int, size), cast(str, sha256)


def _positive_config_int(config: Mapping[str, Any], section: str, field: str) -> int:
    values = config.get(section)
    value = values.get(field) if isinstance(values, Mapping) else None
    _require(
        isinstance(value, int) and not isinstance(value, bool) and value > 0,
        f"onnx/tts.json: {section}.{field} must be a positive integer",
    )
    return cast(int, value)


def _runtime_from_tts(payload: bytes) -> tuple[int, dict[str, Any]]:
    try:
        config = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CatalogError("onnx/tts.json is not valid UTF-8 JSON") from exc
    _require(isinstance(config, Mapping), "onnx/tts.json must contain an object")
    sample_rate = _positive_config_int(config, "ae", "sample_rate")
    base_chunk_size = _positive_config_int(config, "ae", "base_chunk_size")
    chunk_compress_factor = _positive_config_int(config, "ttl", "chunk_compress_factor")
    latent_dim = _positive_config_int(config, "ttl", "latent_dim")
    return sample_rate, {
        "layout": "supertonic-3-v1",
        "sample_rate": sample_rate,
        "base_chunk_size": base_chunk_size,
        "chunk_compress_factor": chunk_compress_factor,
        "latent_dim": latent_dim,
        "chunk_size": base_chunk_size * chunk_compress_factor,
        "latent_width": latent_dim * chunk_compress_factor,
    }


def _build_artifact(
    *,
    role: str,
    component: str | None,
    path: str,
    repository: str,
    revision: str,
    metadata: Mapping[str, Any],
    payload: bytes | None = None,
) -> dict[str, Any]:
    filename = Path(path).name
    _safe_name(filename, f"{role} filename")
    size, sha256 = _file_integrity(repository, revision, path, metadata, payload=payload)
    return {
        "role": role,
        "component": component,
        "filename": filename,
        "path": path,
        "url": huggingface_resolve_url(repository, revision, path),
        "size": size,
        "sha256": sha256,
        "format": Path(path).suffix.lower().lstrip("."),
    }


def build_catalog(
    *,
    repository: str = DEFAULT_REPOSITORY,
    revision: str = DEFAULT_REVISION,
    resolved_revision: str | None = None,
) -> dict[str, Any]:
    """Build a deterministic catalog for one exact Supertonic-3 repository revision."""
    actual_revision = resolved_revision or resolve_revision(repository, revision)
    _require(
        _SHA_RE.fullmatch(actual_revision) is not None,
        "resolved revision must be a 40-character SHA",
    )
    tree = _repository_tree(repository, actual_revision)
    tree_by_path = {item["path"]: item for item in tree if isinstance(item.get("path"), str)}
    required_paths = _SUPPORT_PATHS | _MODEL_PATHS
    for path in required_paths.values():
        _require(path in tree_by_path, f"Missing upstream asset {path}")
    styles = sorted(
        path
        for path in tree_by_path
        if path.startswith("voice_styles/") and path.count("/") == 1 and path.endswith(".json")
    )
    _require(bool(styles), "No voice_styles/*.json assets found")
    style_components = [_safe_name(Path(path).stem, "voice style component") for path in styles]
    _require(
        len(style_components) == len(set(style_components)), "Duplicate voice style components"
    )
    if actual_revision == CURRENT_SNAPSHOT:
        _require(
            set(style_components) == set(_CURRENT_STYLES),
            "Current Supertonic-3 snapshot must contain F1-F5 and M1-M5 styles",
        )

    tts_path = _SUPPORT_PATHS["config"]
    tts_payload = _read_url(huggingface_resolve_url(repository, actual_revision, tts_path))
    sample_rate, runtime = _runtime_from_tts(tts_payload)
    artifacts: list[dict[str, Any]] = []
    for role, path in _SUPPORT_PATHS.items():
        payload = tts_payload if role == "config" else None
        artifacts.append(
            _build_artifact(
                role=role,
                component=None,
                path=path,
                repository=repository,
                revision=actual_revision,
                metadata=tree_by_path[path],
                payload=payload,
            )
        )
    for component, path in _MODEL_PATHS.items():
        artifacts.append(
            _build_artifact(
                role="model",
                component=component,
                path=path,
                repository=repository,
                revision=actual_revision,
                metadata=tree_by_path[path],
            )
        )
    for component, path in zip(style_components, styles, strict=True):
        artifacts.append(
            _build_artifact(
                role="voice_style",
                component=component,
                path=path,
                repository=repository,
                revision=actual_revision,
                metadata=tree_by_path[path],
            )
        )
    role_order = {"config": 0, "unicode_indexer": 1, "model": 2, "voice_style": 3}
    artifacts.sort(key=lambda item: (role_order[item["role"]], item["component"] or ""))
    bundle = {
        "aliases": [],
        "artifacts": artifacts,
        "sample_rate": sample_rate,
        "languages": list(_LANGUAGES),
        "default_language": "en",
        "voices": [
            {"name": component, "artifact_component": component} for component in style_components
        ],
        "default_voice": "F1" if "F1" in style_components else style_components[0],
        "runtime": runtime,
    }
    catalog = {
        "schema": 1,
        "kind": "supertonic-onnx-bundle-catalog",
        "source": {
            "provider": "huggingface",
            "repository": repository,
            "requested_revision": revision,
            "revision": actual_revision,
            "bundle_count": 1,
        },
        "bundles": {"supertonic-3": bundle},
    }
    verify_catalog(catalog)
    return catalog


def verify_catalog(catalog: Any, source: Any | None = None) -> None:
    """Verify normalized catalog integrity and optional source.json provenance."""
    _require(isinstance(catalog, Mapping), "Catalog must be an object")
    try:
        items = _parse_supertonic(dict(catalog))
    except (TypeError, ValueError, OnnxCatalogError) as exc:
        raise CatalogError(str(exc)) from exc
    catalog_source = catalog["source"]
    _require(
        isinstance(catalog_source, Mapping)
        and set(catalog_source)
        == {"provider", "repository", "requested_revision", "revision", "bundle_count"},
        "Catalog source has unexpected or missing fields",
    )
    _require(
        catalog_source.get("provider") == "huggingface",
        "Catalog source provider must be huggingface",
    )
    _require(
        len(items) == 1 and items[0].id == "supertonic-3",
        "Catalog must contain exactly supertonic-3",
    )
    if source is not None:
        expected_source = {"schema": 1, **dict(catalog_source)}
        _require(source == expected_source, "Source metadata does not match catalog provenance")

    bundle = catalog["bundles"]["supertonic-3"]
    item = items[0]
    artifacts = bundle["artifacts"]
    expected_fields = {
        "role",
        "component",
        "filename",
        "path",
        "url",
        "size",
        "sha256",
        "format",
    }
    for artifact in artifacts:
        _require(
            isinstance(artifact, Mapping)
            and frozenset(artifact)
            in {frozenset(expected_fields), frozenset(expected_fields | {"metadata"})},
            "Artifact has unexpected or missing fields",
        )
        _require(
            Path(artifact["path"]).name == artifact["filename"],
            "Artifact filename does not match path",
        )
    _require(
        bundle["languages"] == list(_LANGUAGES),
        "Catalog languages do not match compatibility contract",
    )
    if item.metadata["source_revision"] == CURRENT_SNAPSHOT:
        components = {
            artifact.component for artifact in item.artifacts if artifact.role == "voice_style"
        }
        _require(
            components == set(_CURRENT_STYLES), "Current snapshot is missing required voice styles"
        )
    runtime = bundle["runtime"]
    sample_rate = bundle["sample_rate"]
    _require(runtime.get("sample_rate") == sample_rate, "Runtime sample rate does not match config")
    base_chunk_size = runtime.get("base_chunk_size")
    chunk_compress_factor = runtime.get("chunk_compress_factor")
    latent_dim = runtime.get("latent_dim")
    for field, value in (
        ("base_chunk_size", base_chunk_size),
        ("chunk_compress_factor", chunk_compress_factor),
        ("latent_dim", latent_dim),
    ):
        _require(
            isinstance(value, int) and not isinstance(value, bool) and value > 0,
            f"Runtime {field} must be a positive integer",
        )
    _require(
        runtime.get("chunk_size") == base_chunk_size * chunk_compress_factor,
        "Runtime chunk size does not match config geometry",
    )
    _require(
        runtime.get("latent_width") == latent_dim * chunk_compress_factor,
        "Runtime latent width does not match config geometry",
    )
    _require(
        {"config", "unicode_indexer", "model", "voice_style"} == ST_ARTIFACT_ROLES,
        "Invalid role contract",
    )
    _require(set(_MODEL_PATHS) == ST_MODEL_COMPONENTS, "Invalid model component contract")


def load_catalog(path: Path, source: Path | None = None) -> dict[str, Any]:
    """Read and verify a catalog, optionally checking its companion source.json."""
    try:
        catalog = json.loads(path.read_text(encoding="utf-8"))
        source_data = json.loads(source.read_text(encoding="utf-8")) if source is not None else None
    except (OSError, json.JSONDecodeError) as exc:
        raise CatalogError(f"Unable to read Supertonic catalog: {path}") from exc
    verify_catalog(catalog, source_data)
    return cast(dict[str, Any], catalog)
