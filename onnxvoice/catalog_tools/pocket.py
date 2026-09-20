"""Fetch, normalize, and validate the upstream Pocket ONNX bundle inventory."""

from __future__ import annotations

import hashlib
import json
import re
import urllib.error
import urllib.parse
import urllib.request
from copy import deepcopy
from pathlib import Path
from typing import Any, cast

DEFAULT_REPOSITORY = "KevinAHM/pocket-tts-onnx"
DEFAULT_REVISION = "main"
USER_AGENT = "onnxvoice/0.1.0"
_SHA_RE = re.compile(r"^[0-9a-f]{40}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_SAFE_NAME_RE = re.compile(r"^[^\W_][\w.-]*$", re.UNICODE)

REQUIRED_ONNX_ROLES = frozenset(
    {"flow_lm_main", "flow_lm_flow", "mimi_decoder", "mimi_encoder", "text_conditioner"}
)
STATIC_ROLES = frozenset({"bundle_metadata", "tokenizer", "bos_conditioning"})
VALID_ROLES = REQUIRED_ONNX_ROLES | STATIC_ROLES
_ARTIFACT_ORDER = (
    "bundle_metadata",
    "tokenizer",
    "bos_conditioning",
    "flow_lm_main",
    "flow_lm_flow",
    "mimi_decoder",
    "mimi_encoder",
    "text_conditioner",
)


class CatalogError(ValueError):
    """Raised when upstream or normalized catalog data is invalid."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise CatalogError(message)


def _safe_name(value: Any, label: str) -> None:
    if not isinstance(value, str) or not value:
        raise CatalogError(f"{label}: must be a non-empty string")
    _require("/" not in value and "\\" not in value, f"{label}: path separators are forbidden")
    _require(value not in {".", ".."}, f"{label}: dot segments are forbidden")
    _require(not Path(value).is_absolute(), f"{label}: absolute paths are forbidden")
    _require(_SAFE_NAME_RE.fullmatch(value) is not None, f"{label}: unsafe name")


def _safe_relative_path(value: Any, label: str) -> str:
    _require(isinstance(value, str) and bool(value), f"{label}: path is required")
    path = Path(value)
    _require(not path.is_absolute(), f"{label}: absolute paths are forbidden")
    _require(
        "\\" not in value and all(part not in {"", ".", ".."} for part in path.parts),
        f"{label}: unsafe path",
    )
    return value


def _canonical_json(data: Any) -> bytes:
    return (
        json.dumps(data, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("utf-8")


def huggingface_resolve_url(repository: str, revision: str, path: str) -> str:
    repo = "/".join(urllib.parse.quote(part, safe="") for part in repository.split("/"))
    rev = urllib.parse.quote(revision, safe="")
    asset = urllib.parse.quote(path, safe="/")
    return f"https://huggingface.co/{repo}/resolve/{rev}/{asset}?download=true"


def huggingface_api_url(repository: str, revision: str) -> str:
    repo = "/".join(urllib.parse.quote(part, safe="") for part in repository.split("/"))
    rev = urllib.parse.quote(revision, safe="")
    return f"https://huggingface.co/api/models/{repo}/revision/{rev}"


def _read_url(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            return response.read()
    except (OSError, urllib.error.URLError) as exc:
        raise CatalogError(f"Unable to fetch {url}") from exc


def resolve_revision(repository: str, revision: str = DEFAULT_REVISION) -> str:
    """Resolve a Hub branch or tag to an exact lowercase commit SHA."""
    payload = _read_url(huggingface_api_url(repository, revision))
    try:
        data = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CatalogError("Hugging Face repository metadata is not valid UTF-8 JSON") from exc
    resolved = data.get("sha") if isinstance(data, dict) else None
    if not isinstance(resolved, str) or _SHA_RE.fullmatch(resolved.lower()) is None:
        raise CatalogError("Unable to resolve an exact 40-character commit SHA")
    return resolved.lower()


def _fetch_bundle_json(
    repository: str, revision: str, bundle_path: str
) -> tuple[dict[str, Any], str]:
    """Fetch a single bundle.json and return data plus its SHA-256."""
    url = huggingface_resolve_url(repository, revision, bundle_path)
    payload = _read_url(url)
    try:
        data = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CatalogError(f"Bundle {bundle_path} is not valid UTF-8 JSON") from exc
    _require(isinstance(data, dict), f"Bundle {bundle_path} must be a JSON object")
    return data, hashlib.sha256(payload).hexdigest()


def _list_bundle_paths(repository: str, revision: str) -> list[str]:
    """List all bundle.json paths in the repository at the given revision."""
    payload = _read_url(f"{huggingface_api_url(repository, revision)}/tree/main")
    try:
        tree = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CatalogError("Unable to parse repository tree") from exc
    _require(isinstance(tree, list), "Repository tree must be a list")
    bundle_paths = [
        item["path"]
        for item in tree
        if isinstance(item, dict)
        and isinstance(item.get("path"), str)
        and item["path"].startswith("onnx/")
        and item["path"].endswith("/bundle.json")
    ]
    return sorted(bundle_paths)


def _extract_bundle_id(bundle_path: str) -> str:
    parts = bundle_path.split("/")
    _require(
        len(parts) == 3 and parts[0] == "onnx" and parts[2] == "bundle.json",
        f"Invalid bundle path: {bundle_path}",
    )
    return parts[1]


def _validate_artifact_role(role: str, bundle_id: str) -> None:
    _require(role in VALID_ROLES, f"{bundle_id}: unknown artifact role {role!r}")


def _infer_format(filename: str, value: Any) -> str:
    if isinstance(value, str) and value:
        return value
    suffix = Path(filename).suffix.lower().lstrip(".")
    return suffix or "binary"


def _build_artifact(
    *,
    role: str,
    filename: str,
    repository: str,
    revision: str,
    upstream_path: str,
    format: str | None = None,
    size: int | None = None,
    sha256: str | None = None,
    quality: str | None = None,
) -> dict[str, Any]:
    """Build a normalized artifact entry with an explicit public shape."""
    _safe_name(filename, f"artifact {role} filename")
    _safe_relative_path(upstream_path, f"artifact {role} path")
    if size is not None:
        _require(
            isinstance(size, int) and not isinstance(size, bool) and size > 0,
            f"artifact {role}: invalid size",
        )
    if sha256 is not None:
        _require(
            isinstance(sha256, str) and _SHA256_RE.fullmatch(sha256.lower()) is not None,
            f"artifact {role}: invalid sha256",
        )
        sha256 = sha256.lower()
    return {
        "role": role,
        "quality": quality,
        "format": _infer_format(filename, format),
        "filename": filename,
        "path": upstream_path,
        "url": huggingface_resolve_url(repository, revision, upstream_path),
        "size": size,
        "sha256": sha256,
    }


def _artifact_entries(raw: Any) -> list[tuple[str, Any]]:
    if isinstance(raw, dict):
        return list(raw.items())
    if isinstance(raw, list):
        result: list[tuple[str, Any]] = []
        for entry in raw:
            _require(isinstance(entry, dict), "Artifact list entries must be objects")
            result.append((str(entry.get("role") or ""), entry))
        return result
    raise CatalogError("artifacts must be a mapping or list")


def _parse_bundle_entry(
    bundle_id: str,
    bundle_data: dict[str, Any],
    repository: str,
    revision: str,
    bundle_path: str,
) -> dict[str, Any]:
    """Parse a single upstream bundle.json into a normalized catalog entry."""
    _require(isinstance(bundle_data, dict), f"{bundle_id}: bundle must be a JSON object")
    language = bundle_data.get("language")
    _require(isinstance(language, str) and bool(language), f"{bundle_id}: invalid language")
    sample_rate = bundle_data.get("sample_rate")
    _require(isinstance(sample_rate, int) and sample_rate > 0, f"{bundle_id}: invalid sample_rate")
    layers = bundle_data.get("layers")
    _require(isinstance(layers, int) and layers > 0, f"{bundle_id}: invalid layers")
    bundle_schema = bundle_data.get("schema_version")
    _require(
        isinstance(bundle_schema, int) and bundle_schema > 0, f"{bundle_id}: invalid schema_version"
    )

    aliases = bundle_data.get("aliases") or []
    _require(
        isinstance(aliases, list) and all(isinstance(alias, str) for alias in aliases),
        f"{bundle_id}: aliases must be strings",
    )
    profiles = bundle_data.get("profiles") or {}
    _require(isinstance(profiles, dict), f"{bundle_id}: profiles must be a dict")
    artifacts: list[dict[str, Any]] = []
    seen_role_quality: set[tuple[str, str | None]] = set()

    for role, artifact_info in _artifact_entries(bundle_data.get("artifacts") or {}):
        _validate_artifact_role(role, bundle_id)
        if isinstance(artifact_info, dict):
            filename = artifact_info.get("filename")
            raw_path = artifact_info.get("path")
            quality = artifact_info.get("quality")
            format = artifact_info.get("format")
            size = artifact_info.get("size")
            sha256 = artifact_info.get("sha256")
        elif isinstance(artifact_info, str):
            filename = artifact_info
            raw_path = None
            quality = None
            format = None
            size = None
            sha256 = None
        else:
            raise CatalogError(f"{bundle_id}: invalid artifact info for {role!r}")
        if not isinstance(filename, str) or not filename:
            if isinstance(raw_path, str) and raw_path:
                filename = Path(raw_path).name
            else:
                raise CatalogError(f"{bundle_id}: invalid filename for {role!r}")
        upstream_path = raw_path or f"onnx/{bundle_id}/{filename}"
        role_quality = (role, quality)
        _require(
            role_quality not in seen_role_quality,
            f"{bundle_id}: duplicate (role, quality) pair: {role_quality}",
        )
        seen_role_quality.add(role_quality)
        artifacts.append(
            _build_artifact(
                role=role,
                filename=filename,
                repository=repository,
                revision=revision,
                upstream_path=upstream_path,
                format=format,
                size=size,
                sha256=sha256,
                quality=quality,
            )
        )

    for role in STATIC_ROLES:
        _require(
            sum(artifact["role"] == role for artifact in artifacts) == 1,
            f"{bundle_id}: missing or duplicate static role {role!r}",
        )
    for role in REQUIRED_ONNX_ROLES:
        _require(
            any(artifact["role"] == role for artifact in artifacts),
            f"{bundle_id}: no quality variant for role {role!r}",
        )
    for profile_name, profile_roles in profiles.items():
        _require(
            isinstance(profile_roles, dict), f"{bundle_id}: profile {profile_name!r} must be a dict"
        )
        for role in REQUIRED_ONNX_ROLES:
            _require(
                role in profile_roles,
                f"{bundle_id}: profile {profile_name!r} omits required role {role!r}",
            )
            _require(
                (role, profile_roles[role]) in seen_role_quality,
                f"{bundle_id}: profile {profile_name!r} references missing ({role!r}, {profile_roles[role]!r})",
            )

    entry: dict[str, Any] = {
        "id": bundle_id,
        "aliases": aliases,
        "sample_rate": sample_rate,
        "language": language,
        "layers": layers,
        "bundle_schema": bundle_schema,
        "profiles": profiles,
        "artifacts": sorted(
            artifacts,
            key=lambda artifact: (
                _ARTIFACT_ORDER.index(artifact["role"]),
                artifact["quality"] or "",
            ),
        ),
    }
    for key in (
        "max_token_per_chunk",
        "model_recommended_frames_after_eos",
        "remove_semicolons",
        "pad_with_spaces_for_short_inputs",
    ):
        if key in bundle_data:
            entry[key] = bundle_data[key]
    return entry


def build_catalog(
    *,
    repository: str = DEFAULT_REPOSITORY,
    revision: str = DEFAULT_REVISION,
    resolved_revision: str | None = None,
) -> dict[str, Any]:
    """Build a deterministic Pocket catalog for one exact upstream revision."""
    actual_revision = (resolved_revision or resolve_revision(repository, revision)).lower()
    _require(
        _SHA_RE.fullmatch(actual_revision) is not None,
        "resolved_revision must be a 40-character SHA",
    )
    bundle_paths = _list_bundle_paths(repository, actual_revision)
    _require(len(bundle_paths) > 0, "No bundles found in repository")
    bundles: dict[str, dict[str, Any]] = {}
    for bundle_path in bundle_paths:
        bundle_id = _extract_bundle_id(bundle_path)
        _safe_name(bundle_id, "bundle id")
        bundle_data, _ = _fetch_bundle_json(repository, actual_revision, bundle_path)
        bundles[bundle_id] = _parse_bundle_entry(
            bundle_id=bundle_id,
            bundle_data=bundle_data,
            repository=repository,
            revision=actual_revision,
            bundle_path=bundle_path,
        )
    sorted_bundles = {bundle_id: bundles[bundle_id] for bundle_id in sorted(bundles)}
    return {
        "schema": 1,
        "kind": "pocket-onnx-bundle-catalog",
        "source": {
            "provider": "huggingface",
            "repository": repository,
            "requested_revision": revision,
            "revision": actual_revision,
            "bundle_root": "onnx",
            "bundle_count": len(sorted_bundles),
            "license": "cc-by-4.0",
            "snapshot_note": None,
        },
        "bundles": sorted_bundles,
    }


def _verify_artifact(
    artifact: dict[str, Any], bundle_id: str, repository: str, revision: str
) -> tuple[str, str | None]:
    required = {"role", "quality", "format", "filename", "path", "url", "size", "sha256"}
    _require(set(artifact) == required, f"{bundle_id}: artifact fields must be exactly {required}")
    role = artifact["role"]
    _require(role in VALID_ROLES, f"{bundle_id}: invalid role {role!r}")
    _safe_name(artifact["filename"], f"{bundle_id}/{role}: filename")
    path = _safe_relative_path(artifact["path"], f"{bundle_id}/{role}: path")
    _require(
        isinstance(artifact["format"], str) and bool(artifact["format"]),
        f"{bundle_id}/{role}: invalid format",
    )
    url = artifact["url"]
    _require(
        isinstance(url, str) and url == huggingface_resolve_url(repository, revision, path),
        f"{bundle_id}/{role}: URL does not match repository/revision/path",
    )
    size = artifact["size"]
    _require(
        size is None or (isinstance(size, int) and not isinstance(size, bool) and size > 0),
        f"{bundle_id}/{role}: invalid size",
    )
    sha256 = artifact["sha256"]
    _require(
        sha256 is None or (isinstance(sha256, str) and _SHA256_RE.fullmatch(sha256) is not None),
        f"{bundle_id}/{role}: invalid sha256",
    )
    return role, artifact["quality"]


def _legacy_catalog_view(catalog: dict[str, Any]) -> tuple[dict[str, Any], bool]:
    """Normalize the old private list catalog for temporary compatibility."""
    if catalog.get("kind") != "pocket-bundle-catalog" or not isinstance(
        catalog.get("bundles"), list
    ):
        return catalog, False
    legacy = deepcopy(catalog)
    source = dict(legacy.get("source") or {})
    revision = str(source.get("revision") or "0" * 40)
    source.setdefault("bundle_root", "onnx")
    source.setdefault("license", "cc-by-4.0")
    source.setdefault("snapshot_note", None)
    legacy["source"] = source
    normalized: dict[str, dict[str, Any]] = {}
    for raw_bundle in legacy["bundles"]:
        if not isinstance(raw_bundle, dict):
            continue
        bundle_id = raw_bundle.get("id")
        if not isinstance(bundle_id, str):
            continue
        bundle = dict(raw_bundle)
        artifacts: list[dict[str, Any]] = []
        for raw_artifact in bundle.get("artifacts") or []:
            if not isinstance(raw_artifact, dict):
                continue
            artifact = dict(raw_artifact)
            filename = artifact.get("filename") or "artifact.bin"
            path = artifact.get("path") or f"onnx/{bundle_id}/{filename}"
            artifact.setdefault("quality", None)
            artifact.setdefault("format", _infer_format(str(filename), None))
            artifact["path"] = path
            artifact["url"] = huggingface_resolve_url(str(source.get("repository")), revision, path)
            artifact.setdefault("size", None)
            artifact.setdefault("sha256", None)
            artifacts.append(artifact)
        bundle["artifacts"] = artifacts
        normalized[bundle_id] = bundle
    legacy["kind"] = "pocket-onnx-bundle-catalog"
    legacy["bundles"] = {key: normalized[key] for key in sorted(normalized)}
    return legacy, True


def verify_catalog(catalog: Any) -> None:
    """Verify the canonical Pocket catalog contract."""
    _require(isinstance(catalog, dict), "Catalog must be an object")
    catalog, _legacy = _legacy_catalog_view(catalog)
    _require(
        set(catalog) == {"schema", "kind", "source", "bundles"},
        "Catalog must have exactly schema, kind, source, bundles",
    )
    _require(catalog.get("schema") == 1, "Catalog schema must be 1")
    _require(
        catalog.get("kind") == "pocket-onnx-bundle-catalog",
        f"Unexpected catalog kind: {catalog.get('kind')!r}",
    )

    source = catalog.get("source")
    _require(isinstance(source, dict), "Catalog source must be an object")
    source = cast(dict[str, Any], source)
    required_source = {
        "provider",
        "repository",
        "requested_revision",
        "revision",
        "bundle_root",
        "bundle_count",
        "license",
        "snapshot_note",
    }
    _require(set(source) == required_source, f"Catalog source must have exactly {required_source}")
    _require(source.get("provider") == "huggingface", "Catalog source provider must be huggingface")
    repository = source.get("repository")
    _require(
        isinstance(repository, str) and re.fullmatch(r"[^/\\ ]+/[^/\\ ]+", repository) is not None,
        "Invalid source repository",
    )
    assert isinstance(repository, str)
    requested_revision = source.get("requested_revision")
    _require(
        isinstance(requested_revision, str) and bool(requested_revision),
        "Invalid requested revision",
    )
    revision = source.get("revision")
    _require(
        isinstance(revision, str) and _SHA_RE.fullmatch(revision) is not None,
        "Source revision must be a lowercase 40-character SHA",
    )
    assert isinstance(revision, str)
    _require(source.get("bundle_root") == "onnx", "Catalog bundle_root must be onnx")
    _require(
        isinstance(source.get("license"), str) and bool(source["license"]),
        "Catalog license is required",
    )
    _require(
        source.get("snapshot_note") is None or isinstance(source["snapshot_note"], str),
        "Invalid snapshot_note",
    )

    bundles = catalog.get("bundles")
    _require(isinstance(bundles, dict), "Catalog bundles must be a mapping")
    bundles = cast(dict[str, Any], bundles)
    bundle_count = source.get("bundle_count")
    _require(isinstance(bundle_count, int) and bundle_count > 0, "Invalid bundle count")
    _require(
        len(bundles) == bundle_count,
        f"Bundle count mismatch: expected {bundle_count}, got {len(bundles)}",
    )
    _require(list(bundles) == sorted(bundles), "Catalog bundles must be sorted by id")

    all_aliases: dict[str, str] = {}
    for map_id, raw_bundle in bundles.items():
        _require(isinstance(raw_bundle, dict), f"{map_id}: bundle must be an object")
        bundle = cast(dict[str, Any], raw_bundle)
        _require(
            bundle.get("id") == map_id,
            f"Bundle map key {map_id!r} does not match id {bundle.get('id')!r}",
        )
        _safe_name(map_id, f"bundle {map_id}")
        aliases = bundle.get("aliases", [])
        _require(
            isinstance(aliases, list) and all(isinstance(alias, str) for alias in aliases),
            f"{map_id}: aliases must be strings",
        )
        for alias in aliases:
            _require(
                alias not in all_aliases or all_aliases[alias] == map_id,
                f"Alias {alias!r} is ambiguous",
            )
            all_aliases[alias] = map_id
        sample_rate = bundle.get("sample_rate")
        _require(isinstance(sample_rate, int) and sample_rate > 0, f"{map_id}: invalid sample_rate")
        artifacts = bundle.get("artifacts")
        _require(isinstance(artifacts, list), f"{map_id}: artifacts must be a list")
        artifacts = cast(list[Any], artifacts)
        seen_static: set[str] = set()
        seen_role_quality: set[tuple[str, str | None]] = set()
        for raw_artifact in artifacts:
            _require(isinstance(raw_artifact, dict), f"{map_id}: artifact must be an object")
            role, quality = _verify_artifact(
                cast(dict[str, Any], raw_artifact), map_id, repository, revision
            )
            pair = (role, quality)
            _require(
                pair not in seen_role_quality, f"{map_id}: duplicate (role, quality) pair: {pair}"
            )
            seen_role_quality.add(pair)
            if role in STATIC_ROLES:
                _require(role not in seen_static, f"{map_id}: duplicate static role {role!r}")
                seen_static.add(role)
        for role in STATIC_ROLES:
            _require(role in seen_static, f"{map_id}: missing static role {role!r}")
        for role in REQUIRED_ONNX_ROLES:
            _require(
                any(pair[0] == role for pair in seen_role_quality),
                f"{map_id}: no quality variant for role {role!r}",
            )
        profiles = bundle.get("profiles", {})
        _require(isinstance(profiles, dict), f"{map_id}: profiles must be a dict")
        for profile_name, profile_roles in profiles.items():
            _require(
                isinstance(profile_roles, dict),
                f"{map_id}: profile {profile_name!r} must be a dict",
            )
            for role in profile_roles:
                _require(
                    role in REQUIRED_ONNX_ROLES,
                    f"{map_id}: profile {profile_name!r} references unknown role {role!r}",
                )
            for role in REQUIRED_ONNX_ROLES:
                _require(
                    role in profile_roles,
                    f"{map_id}: profile {profile_name!r} omits required role {role!r}",
                )
                _require(
                    (role, profile_roles[role]) in seen_role_quality,
                    f"{map_id}: profile {profile_name!r} references missing ({role!r}, {profile_roles[role]!r})",
                )


def load_catalog(path: Path) -> dict[str, Any]:
    """Load and verify a Pocket catalog from disk."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CatalogError(f"Unable to load catalog: {path}") from exc
    verify_catalog(data)
    return data
