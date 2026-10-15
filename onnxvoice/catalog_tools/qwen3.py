"""Build and verify pinned Qwen3 WaveKat ONNX bundle catalogs."""

from __future__ import annotations

import fnmatch
import hashlib
import json
import re
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Mapping
from pathlib import Path, PurePosixPath
from typing import Any, cast

from ..catalog import _parse_qwen3
from ..errors import CatalogError as OnnxCatalogError

DEFAULT_SEED_PATH = Path(__file__).with_name("qwen3_seed.json")
DEFAULT_CATALOG_URL = (
    "https://raw.githubusercontent.com/buchwandler/qwen3-onnx-bundles/main/catalog/bundles.json"
)
USER_AGENT = "onnxvoice/0.3.0"
_SHA_RE = re.compile(r"^[0-9a-f]{40}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_SAFE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
_QUALITIES = ("fp32", "int4")
_GENERAL_MODELS = ("talker_prefill", "talker_decode", "code_predictor", "vocoder")
_BASE_MODELS = ("speaker_encoder", "tokenizer_encoder")
_EMBEDDINGS = (
    "text_embedding",
    "text_projection_fc1_weight",
    "text_projection_fc1_bias",
    "text_projection_fc2_weight",
    "text_projection_fc2_bias",
    "talker_codec_embedding",
    *(f"cp_codec_embedding_{index}" for index in range(15)),
)
_OPTIONAL_EMBEDDINGS = {"small_to_mtp_projection_bias", "small_to_mtp_projection_weight"}
_TOKENIZER_FILES = {
    "tokenizer.json": "tokenizer_json",
    "vocab.json": "vocab",
    "merges.txt": "merges",
    "tokenizer_config.json": "tokenizer_config",
    "special_tokens_map.json": "special_tokens",
    "added_tokens.json": "added_tokens",
}


class CatalogError(ValueError):
    """Raised when a Qwen3 seed, upstream tree, or normalized catalog is invalid."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise CatalogError(message)


def _safe_id(value: Any, label: str) -> str:
    _require(
        isinstance(value, str) and _SAFE_ID_RE.fullmatch(value) is not None, f"{label} is unsafe"
    )
    return cast(str, value)


def huggingface_api_url(repository: str, revision: str) -> str:
    repo = "/".join(urllib.parse.quote(part, safe="") for part in repository.split("/"))
    rev = urllib.parse.quote(revision, safe="")
    return f"https://huggingface.co/api/models/{repo}/revision/{rev}"


def huggingface_tree_url(repository: str, revision: str) -> str:
    repo = "/".join(urllib.parse.quote(part, safe="") for part in repository.split("/"))
    rev = urllib.parse.quote(revision, safe="")
    return f"https://huggingface.co/api/models/{repo}/tree/{rev}?recursive=true&expand=true"


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


def resolve_revision(repository: str, revision: str) -> str:
    """Resolve a branch or tag to an immutable lowercase 40-character commit."""
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


def _repository_tree(repository: str, revision: str) -> dict[str, dict[str, Any]]:
    url = huggingface_tree_url(repository, revision)
    entries: dict[str, dict[str, Any]] = {}
    while url:
        request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                payload = response.read()
                next_link = response.headers.get("Link", "")
        except (OSError, urllib.error.URLError) as exc:
            raise CatalogError(f"Unable to fetch repository tree for {revision}") from exc
        try:
            page = json.loads(payload.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise CatalogError("Hugging Face repository tree is not valid JSON") from exc
        _require(isinstance(page, list), "Hugging Face repository tree must be a list")
        for entry in page:
            if not isinstance(entry, Mapping) or not isinstance(entry.get("path"), str):
                continue
            path = entry["path"]
            _require(path not in entries, f"Repository tree contains duplicate path {path!r}")
            entries[path] = dict(entry)
        match = re.search(r'<([^>]+)>;\s*rel="next"', next_link)
        url = match.group(1) if match else ""
    return entries


def _stream_integrity(repository: str, revision: str, path: str) -> tuple[int, str]:
    request = urllib.request.Request(
        huggingface_resolve_url(repository, revision, path), headers={"User-Agent": USER_AGENT}
    )
    digest = hashlib.sha256()
    size = 0
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            while chunk := response.read(1024 * 1024):
                digest.update(chunk)
                size += len(chunk)
    except (OSError, urllib.error.URLError) as exc:
        raise CatalogError(f"Unable to stream integrity data for {path}") from exc
    _require(size > 0, f"{path}: empty artifacts are not supported")
    return size, digest.hexdigest()


def _artifact_integrity(
    repository: str, revision: str, path: str, metadata: Mapping[str, Any]
) -> tuple[int, str]:
    raw_size = metadata.get("size")
    lfs = metadata.get("lfs")
    raw_sha = lfs.get("oid") if isinstance(lfs, Mapping) else None
    size = raw_size if type(raw_size) is int and raw_size > 0 else None
    sha256 = raw_sha if isinstance(raw_sha, str) and _SHA256_RE.fullmatch(raw_sha) else None
    if size is not None and sha256 is not None:
        return size, sha256
    downloaded_size, downloaded_sha = _stream_integrity(repository, revision, path)
    _require(
        size is None or size == downloaded_size, f"{path}: size disagrees with upstream metadata"
    )
    _require(
        sha256 is None or sha256 == downloaded_sha, f"{path}: LFS SHA-256 disagrees with bytes"
    )
    return downloaded_size, downloaded_sha


def _seed_bundles(seed: Any) -> Mapping[str, Any]:
    _require(isinstance(seed, Mapping), "Qwen3 seed catalog must be an object")
    _require(
        type(seed.get("schema")) is int and seed.get("schema") == 1, "Qwen3 seed schema must be 1"
    )
    _require(seed.get("kind") == "qwen3-onnx-bundle-seed", "Unexpected Qwen3 seed kind")
    bundles = seed.get("bundles")
    expected_ids = {"base-0.6b", "voice-design-1.7b"}
    _require(
        isinstance(bundles, Mapping) and set(bundles) == expected_ids,
        "Qwen3 seed must contain exactly the supported Base and VoiceDesign bundles",
    )
    aliases_seen = set(expected_ids)
    for bundle_id, bundle in bundles.items():
        _require(isinstance(bundle, Mapping), f"{bundle_id}: seed bundle must be an object")
        model_type = bundle.get("model_type")
        expected_type = "base" if bundle_id == "base-0.6b" else "voice_design"
        _require(model_type == expected_type, f"{bundle_id}: unsupported model_type")
        _require(bundle.get("layout") == "qwen3-wavekat-v1", f"{bundle_id}: unsupported layout")
        _require(
            type(bundle.get("sample_rate")) is int and bundle["sample_rate"] == 24000,
            f"{bundle_id}: sample_rate must be 24000",
        )
        source = bundle.get("source")
        _require(isinstance(source, Mapping), f"{bundle_id}: source metadata is required")
        repository = source.get("repository")
        _require(
            isinstance(repository, str)
            and re.fullmatch(r"[^/\\\s]+/[^/\\\s]+", repository) is not None,
            f"{bundle_id}: invalid source repository",
        )
        _require(
            source.get("provider") == "huggingface", f"{bundle_id}: provider must be huggingface"
        )
        _require(
            isinstance(source.get("revision"), str) and bool(source["revision"]),
            f"{bundle_id}: source revision is required",
        )
        _require(
            isinstance(source.get("license"), str) and bool(source["license"]),
            f"{bundle_id}: source license is required",
        )
        aliases = bundle.get("aliases")
        _require(
            isinstance(aliases, list) and len(aliases) > 0,
            f"{bundle_id}: aliases must be a non-empty list",
        )
        for alias in aliases:
            alias = _safe_id(alias, f"{bundle_id} alias")
            _require(alias not in aliases_seen, f"Duplicate or ambiguous Qwen3 alias {alias!r}")
            aliases_seen.add(alias)
        profiles = bundle.get("profiles")
        _require(
            isinstance(profiles, Mapping) and set(profiles) == set(_QUALITIES),
            f"{bundle_id}: profiles must contain fp32 and int4",
        )
        for quality, profile in profiles.items():
            _require(
                isinstance(profile, Mapping), f"{bundle_id}/{quality}: profile must be an object"
            )
            includes = profile.get("include")
            _require(
                isinstance(includes, list) and len(includes) > 0,
                f"{bundle_id}/{quality}: include must be a non-empty list",
            )
            for pattern in includes:
                _require(
                    isinstance(pattern, str) and bool(pattern),
                    f"{bundle_id}/{quality}: invalid include",
                )
                _validate_seed_pattern(pattern, f"{bundle_id}/{quality}")
    return cast(Mapping[str, Any], bundles)


def _validate_seed_pattern(pattern: str, label: str) -> None:
    _require("\\" not in pattern and not pattern.startswith("/"), f"{label}: unsafe include path")
    parts = PurePosixPath(pattern).parts
    _require(all(part not in {"", ".", ".."} for part in parts), f"{label}: unsafe include path")


def _classify_path(path: str, model_type: str) -> tuple[str, str | None, str | None]:
    if path == "config.json":
        return "config", None, None
    if path.startswith("tokenizer/"):
        component = _TOKENIZER_FILES.get(PurePosixPath(path).name)
        _require(component is not None, f"Unsupported tokenizer asset {path!r}")
        return "tokenizer", component, None
    if path.startswith("embeddings/") and path.endswith(".npy"):
        component = PurePosixPath(path).stem
        _require(
            component in (set(_EMBEDDINGS) | _OPTIONAL_EMBEDDINGS),
            f"Unsupported Qwen3 embedding asset {path!r}",
        )
        return "embedding", component, None
    external = path.endswith(".onnx.data")
    graph = path.endswith(".onnx")
    _require(graph or external, f"Unsupported Qwen3 asset path {path!r}")
    model_path = path[:-5] if external else path
    parts = PurePosixPath(model_path).parts
    quality = parts[0] if len(parts) == 2 and parts[0] in _QUALITIES else None
    name = PurePosixPath(model_path).stem
    known = set(_GENERAL_MODELS) | (set(_BASE_MODELS) if model_type == "base" else set())
    _require(name in known, f"Unknown Qwen3 model component in {path!r}")
    if name in _GENERAL_MODELS:
        _require(quality is not None, f"Generation model {name} must be under fp32/ or int4/")
        expected = f"{quality}/{name}.onnx"
    else:
        _require(quality is None, f"Base encoder {name} must be shared across profiles")
        expected = f"{name}.onnx"
    _require(model_path == expected, f"Unexpected Qwen3 model path {path!r}")
    return ("model_data" if external else "model"), name, quality


def _build_artifact(
    role: str,
    component: str | None,
    quality: str | None,
    path: str,
    repository: str,
    revision: str,
    tree_entry: Mapping[str, Any],
) -> dict[str, Any]:
    try:
        from ..types import validate_relative_path

        validate_relative_path(path, field_name="Qwen3 artifact path")
    except ValueError as exc:
        raise CatalogError(str(exc)) from exc
    size, sha256 = _artifact_integrity(repository, revision, path, tree_entry)
    return {
        "role": role,
        "component": component,
        "quality": quality,
        "path": path,
        "url": huggingface_resolve_url(repository, revision, path),
        "size": size,
        "sha256": sha256,
    }


def build_catalog(seed_catalog: Any | None = None) -> dict[str, Any]:
    """Materialize the pinned bundle seed into explicit deterministic artifact records."""
    if seed_catalog is None:
        try:
            seed_catalog = json.loads(DEFAULT_SEED_PATH.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise CatalogError(f"Unable to load Qwen3 seed catalog: {DEFAULT_SEED_PATH}") from exc
    seeds = _seed_bundles(seed_catalog)
    bundles: dict[str, Any] = {}
    source_entries: list[dict[str, str]] = []
    for bundle_id in sorted(seeds):
        seed = seeds[bundle_id]
        source = seed["source"]
        repository = source["repository"]
        revision = resolve_revision(repository, source["revision"])
        tree = _repository_tree(repository, revision)
        raw_profiles = seed["profiles"]
        artifact_by_key: dict[tuple[str, str | None, str | None], dict[str, Any]] = {}
        profile_map: dict[str, dict[str, list[str]]] = {}
        integrity_cache: dict[str, tuple[int, str]] = {}
        for quality in _QUALITIES:
            includes = raw_profiles[quality]["include"]
            selected_paths: set[str] = set()
            for pattern in includes:
                matches = sorted(
                    path
                    for path, entry in tree.items()
                    if entry.get("type", "file") == "file" and fnmatch.fnmatchcase(path, pattern)
                )
                _require(
                    bool(matches),
                    f"{bundle_id}/{quality}: include pattern matched no files: {pattern}",
                )
                selected_paths.update(matches)
            profile_components: set[str] = set()
            for path in sorted(selected_paths):
                role, component, artifact_quality = _classify_path(path, seed["model_type"])
                if artifact_quality is not None:
                    _require(
                        artifact_quality == quality,
                        f"{bundle_id}/{quality}: include selected wrong quality asset {path!r}",
                    )
                if role == "model":
                    profile_components.add(cast(str, component))
                key = (role, component, artifact_quality)
                artifact = artifact_by_key.get(key)
                if artifact is not None:
                    _require(
                        artifact["path"] == path,
                        f"{bundle_id}: duplicate role/component/quality {key!r}",
                    )
                    continue
                if path in integrity_cache:
                    size, sha256 = integrity_cache[path]
                    entry = {"size": size, "lfs": {"oid": sha256}}
                else:
                    entry = tree[path]
                artifact = _build_artifact(
                    role, component, artifact_quality, path, repository, revision, entry
                )
                integrity_cache[path] = (artifact["size"], artifact["sha256"])
                artifact_by_key[key] = artifact
            expected_models = set(_GENERAL_MODELS)
            if seed["model_type"] == "base":
                expected_models.update(_BASE_MODELS)
            _require(
                profile_components == expected_models,
                f"{bundle_id}/{quality}: model components do not match required inventory",
            )
            profile_map[quality] = {"components": sorted(profile_components)}
        artifacts = sorted(
            artifact_by_key.values(),
            key=lambda artifact: (
                artifact["role"],
                artifact["component"] or "",
                artifact["quality"] or "",
                artifact["path"],
            ),
        )
        bundle = {
            "id": bundle_id,
            "aliases": list(seed["aliases"]),
            "layout": seed["layout"],
            "model_type": seed["model_type"],
            "sample_rate": seed["sample_rate"],
            "source": {
                "provider": "huggingface",
                "repository": repository,
                "revision": revision,
                "license": source["license"],
            },
            "profiles": profile_map,
            "artifacts": artifacts,
        }
        bundles[bundle_id] = bundle
        source_entries.append(
            {
                "id": bundle_id,
                "repository": repository,
                "revision": revision,
                "license": source["license"],
            }
        )
    catalog = {
        "schema": 1,
        "kind": "qwen3-onnx-bundle-catalog",
        "source": {"provider": "huggingface", "bundle_count": len(bundles)},
        "bundles": bundles,
    }
    verify_catalog(catalog, build_source(catalog))
    return catalog


def build_source(catalog: Mapping[str, Any]) -> dict[str, Any]:
    """Create deterministic provenance metadata for a normalized Qwen3 catalog."""
    try:
        _parse_qwen3(catalog)
    except (TypeError, ValueError, OnnxCatalogError) as exc:
        raise CatalogError(str(exc)) from exc
    bundles = catalog["bundles"]
    sources = [
        {
            "id": bundle_id,
            "repository": bundle["source"]["repository"],
            "revision": bundle["source"]["revision"],
            "license": bundle["source"]["license"],
        }
        for bundle_id, bundle in sorted(bundles.items())
    ]
    return {
        "schema": 1,
        "kind": "qwen3-onnx-source-snapshot",
        "sources": sources,
    }


def verify_catalog(catalog: Any, source: Any | None = None) -> None:
    """Verify a normalized catalog and optional companion provenance without network access."""
    _require(isinstance(catalog, Mapping), "Qwen3 catalog must be an object")
    try:
        _parse_qwen3(catalog)
    except (TypeError, ValueError, OnnxCatalogError) as exc:
        raise CatalogError(str(exc)) from exc
    if source is None:
        return
    _require(isinstance(source, Mapping), "Qwen3 source metadata must be an object")
    expected = build_source(catalog)
    _require(source == expected, "Source metadata does not match catalog provenance")


def load_catalog(path: Path, source: Path | None = None) -> dict[str, Any]:
    """Load and verify a catalog and optional source companion."""
    try:
        catalog = json.loads(path.read_text(encoding="utf-8"))
        source_data = json.loads(source.read_text(encoding="utf-8")) if source is not None else None
    except (OSError, json.JSONDecodeError) as exc:
        raise CatalogError(f"Unable to read Qwen3 catalog: {path}") from exc
    verify_catalog(catalog, source_data)
    return cast(dict[str, Any], catalog)
