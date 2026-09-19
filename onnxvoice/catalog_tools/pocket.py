"""Fetch, normalize, and validate the upstream Pocket ONNX bundle inventory."""

from __future__ import annotations

import hashlib
import json
import re
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, cast

DEFAULT_REPOSITORY = "KevinAHM/pocket-tts-onnx"
DEFAULT_REVISION = "main"
USER_AGENT = "onnxvoice/0.1.0"
_SHA_RE = re.compile(r"^[0-9a-f]{40}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_SAFE_NAME_RE = re.compile(r"^[^\W_][\w.-]*$", re.UNICODE)

# Required ONNX component roles that must have at least one quality variant
REQUIRED_ONNX_ROLES = frozenset(
    {
        "flow_lm_main",
        "flow_lm_flow",
        "mimi_decoder",
        "mimi_encoder",
        "text_conditioner",
    }
)

# Static roles that must have exactly one artifact
STATIC_ROLES = frozenset(
    {
        "bundle_metadata",
        "tokenizer",
        "bos_conditioning",
    }
)

# All valid artifact roles
VALID_ROLES = REQUIRED_ONNX_ROLES | STATIC_ROLES


class CatalogError(ValueError):
    """Raised when upstream or normalized catalog data is invalid."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise CatalogError(message)


def _safe_name(value: Any, label: str) -> None:
    if not isinstance(value, str) or not value:
        raise CatalogError(f"{label}: must be a non-empty string")
    _require(
        "/" not in value and "\\" not in value,
        f"{label}: path separators are forbidden",
    )
    _require(value not in {".", ".."}, f"{label}: dot segments are forbidden")
    _require(not Path(value).is_absolute(), f"{label}: absolute paths are forbidden")
    _require(_SAFE_NAME_RE.fullmatch(value) is not None, f"{label}: unsafe name")


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
    repository: str,
    revision: str,
    bundle_path: str,
) -> tuple[dict[str, Any], str]:
    """Fetch a single bundle.json and return data + SHA-256."""
    url = huggingface_resolve_url(repository, revision, bundle_path)
    payload = _read_url(url)
    try:
        data = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CatalogError(f"Bundle {bundle_path} is not valid UTF-8 JSON") from exc
    _require(isinstance(data, dict), f"Bundle {bundle_path} must be a JSON object")
    return data, hashlib.sha256(payload).hexdigest()


def _list_bundle_paths(
    repository: str,
    revision: str,
) -> list[str]:
    """List all bundle.json paths in the repository at the given revision."""
    # Use the Hugging Face API to list files
    api_url = f"{huggingface_api_url(repository, revision)}/tree/main"
    payload = _read_url(api_url)
    try:
        tree = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CatalogError("Unable to parse repository tree") from exc

    _require(isinstance(tree, list), "Repository tree must be a list")

    bundle_paths: list[str] = []
    for item in tree:
        if not isinstance(item, dict):
            continue
        path = item.get("path", "")
        if path.startswith("onnx/") and path.endswith("/bundle.json"):
            bundle_paths.append(path)

    return sorted(bundle_paths)


def _extract_bundle_id(bundle_path: str) -> str:
    """Extract bundle ID from path like 'onnx/english_2026-04/bundle.json'."""
    parts = bundle_path.split("/")
    _require(
        len(parts) == 3 and parts[0] == "onnx" and parts[2] == "bundle.json",
        f"Invalid bundle path: {bundle_path}",
    )
    return parts[1]


def _validate_artifact_role(role: str, bundle_id: str) -> None:
    """Validate that artifact role is known."""
    _require(role in VALID_ROLES, f"{bundle_id}: unknown artifact role {role!r}")


def _build_artifact(
    *,
    role: str,
    filename: str,
    repository: str,
    revision: str,
    upstream_path: str,
    size: int | None = None,
    sha256: str | None = None,
    quality: str | None = None,
) -> dict[str, Any]:
    """Build a normalized artifact entry."""
    _safe_name(filename, f"artifact {role} filename")

    url = huggingface_resolve_url(repository, revision, upstream_path)

    artifact: dict[str, Any] = {
        "role": role,
        "filename": filename,
        "url": url,
    }

    if size is not None:
        _require(
            isinstance(size, int) and not isinstance(size, bool) and size > 0,
            f"artifact {role}: invalid size",
        )
        artifact["size"] = size

    if sha256 is not None:
        _require(
            isinstance(sha256, str) and _SHA256_RE.fullmatch(sha256) is not None,
            f"artifact {role}: invalid sha256",
        )
        artifact["sha256"] = sha256

    if quality is not None:
        artifact["quality"] = quality

    return artifact


def _parse_bundle_entry(
    bundle_id: str,
    bundle_data: dict[str, Any],
    repository: str,
    revision: str,
    bundle_path: str,
) -> dict[str, Any]:
    """Parse a single bundle.json into a normalized catalog entry."""
    _require(isinstance(bundle_data, dict), f"{bundle_id}: bundle must be a JSON object")

    # Extract basic metadata
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
        isinstance(aliases, list) and all(isinstance(a, str) for a in aliases),
        f"{bundle_id}: aliases must be strings",
    )

    # Extract runtime metadata
    max_token_per_chunk = bundle_data.get("max_token_per_chunk")
    model_recommended_frames_after_eos = bundle_data.get("model_recommended_frames_after_eos")
    remove_semicolons = bundle_data.get("remove_semicolons")
    pad_with_spaces_for_short_inputs = bundle_data.get("pad_with_spaces_for_short_inputs")

    # Extract profiles
    profiles = bundle_data.get("profiles") or {}
    _require(isinstance(profiles, dict), f"{bundle_id}: profiles must be a dict")

    # Build artifacts list
    artifacts: list[dict[str, Any]] = []

    # Static artifacts from bundle.json
    static_artifacts = bundle_data.get("artifacts") or {}
    _require(isinstance(static_artifacts, dict), f"{bundle_id}: artifacts must be a dict")

    # Track which roles we've seen
    seen_roles: set[str] = set()
    seen_role_quality: set[tuple[str, str | None]] = set()

    for role, artifact_info in static_artifacts.items():
        _validate_artifact_role(role, bundle_id)
        _require(
            role not in seen_roles or role in REQUIRED_ONNX_ROLES,
            f"{bundle_id}: duplicate static role {role!r}",
        )
        seen_roles.add(role)

        if isinstance(artifact_info, dict):
            filename = artifact_info.get("filename")
            upstream_path = artifact_info.get("path") or f"onnx/{bundle_id}/{filename}"
            quality: str | None = artifact_info.get("quality")
            size = artifact_info.get("size")
            sha256 = artifact_info.get("sha256")
        elif isinstance(artifact_info, str):
            filename = artifact_info
            upstream_path = f"onnx/{bundle_id}/{filename}"
            quality = None
            size = None
            sha256 = None
        else:
            raise CatalogError(f"{bundle_id}: invalid artifact info for {role!r}")

        _require(
            isinstance(filename, str) and bool(filename),
            f"{bundle_id}: invalid filename for {role!r}",
        )
        assert isinstance(filename, str)
        # Check for duplicate (role, quality) pairs
        role_quality = (role, quality)
        _require(
            role_quality not in seen_role_quality,
            f"{bundle_id}: duplicate (role, quality) pair: {role_quality}",
        )
        seen_role_quality.add(role_quality)

        artifact = _build_artifact(
            role=role,
            filename=filename,
            repository=repository,
            revision=revision,
            upstream_path=upstream_path,
            size=size,
            sha256=sha256,
            quality=quality,
        )
        artifacts.append(artifact)

    # Check required static roles
    for role in STATIC_ROLES:
        _require(role in seen_roles, f"{bundle_id}: missing required static role {role!r}")

    # Check at least one quality for each required ONNX role
    for role in REQUIRED_ONNX_ROLES:
        qualities = {q for r, q in seen_role_quality if r == role}
        _require(len(qualities) > 0, f"{bundle_id}: no quality variant for required role {role!r}")

    # Validate profiles reference existing (role, quality) pairs
    for profile_name, profile_roles in profiles.items():
        _require(
            isinstance(profile_roles, dict), f"{bundle_id}: profile {profile_name!r} must be a dict"
        )
        for role, quality in profile_roles.items():
            _require(
                role in REQUIRED_ONNX_ROLES,
                f"{bundle_id}: profile {profile_name!r} references unknown role {role!r}",
            )
            _require(
                (role, quality) in seen_role_quality,
                f"{bundle_id}: profile {profile_name!r} references missing ({role!r}, {quality!r})",
            )

    # Build metadata
    metadata: dict[str, Any] = {
        "language": language,
        "layers": layers,
        "bundle_schema": bundle_schema,
        "profiles": profiles,
        "source_revision": revision,
        "source_repository": repository,
    }

    if max_token_per_chunk is not None:
        metadata["max_token_per_chunk"] = max_token_per_chunk
    if model_recommended_frames_after_eos is not None:
        metadata["model_recommended_frames_after_eos"] = model_recommended_frames_after_eos
    if remove_semicolons is not None:
        metadata["remove_semicolons"] = remove_semicolons
    if pad_with_spaces_for_short_inputs is not None:
        metadata["pad_with_spaces_for_short_inputs"] = pad_with_spaces_for_short_inputs

    return {
        "id": bundle_id,
        "aliases": aliases,
        "sample_rate": sample_rate,
        "artifacts": artifacts,
        **metadata,
    }


def build_catalog(
    *,
    repository: str = DEFAULT_REPOSITORY,
    revision: str = DEFAULT_REVISION,
    resolved_revision: str | None = None,
) -> dict[str, Any]:
    """Build Pocket catalog from upstream repository.

    Returns deterministic catalog for the same upstream revision.
    """
    actual_revision = resolved_revision or resolve_revision(repository, revision)

    # List all bundle paths
    bundle_paths = _list_bundle_paths(repository, actual_revision)
    _require(len(bundle_paths) > 0, "No bundles found in repository")

    # Parse each bundle
    bundles: list[dict[str, Any]] = []
    for bundle_path in bundle_paths:
        bundle_id = _extract_bundle_id(bundle_path)
        _safe_name(bundle_id, "bundle id")

        bundle_data, _ = _fetch_bundle_json(repository, actual_revision, bundle_path)

        entry = _parse_bundle_entry(
            bundle_id=bundle_id,
            bundle_data=bundle_data,
            repository=repository,
            revision=actual_revision,
            bundle_path=bundle_path,
        )
        bundles.append(entry)

    # Sort by bundle ID for deterministic output
    bundles.sort(key=lambda b: b["id"])

    return {
        "schema": 1,
        "kind": "pocket-bundle-catalog",
        "source": {
            "provider": "huggingface",
            "repository": repository,
            "requested_revision": revision,
            "revision": actual_revision,
            "bundle_count": len(bundles),
        },
        "bundles": bundles,
    }


def verify_catalog(catalog: dict[str, Any]) -> None:
    """Verify a Pocket catalog against the contract."""
    _require(isinstance(catalog, dict), "Catalog must be an object")

    # Check top-level fields
    required_fields = {"schema", "kind", "source", "bundles"}
    _require(set(catalog.keys()) == required_fields, f"Catalog must have exactly {required_fields}")

    _require(catalog.get("schema") == 1, "Catalog schema must be 1")
    _require(
        catalog.get("kind") == "pocket-bundle-catalog",
        f"Unexpected catalog kind: {catalog.get('kind')!r}",
    )

    # Verify source
    source = catalog.get("source")
    _require(isinstance(source, dict), "Catalog source must be an object")
    source = cast(dict[str, Any], source)

    required_source_fields = {
        "provider",
        "repository",
        "requested_revision",
        "revision",
        "bundle_count",
    }
    _require(
        set(source.keys()) == required_source_fields,
        f"Catalog source must have exactly {required_source_fields}",
    )

    _require(source.get("provider") == "huggingface", "Catalog source provider must be huggingface")

    repository = source.get("repository")
    _require(
        isinstance(repository, str) and re.fullmatch(r"[^/\\ ]+/[^/\\ ]+", repository) is not None,
        "Invalid source repository",
    )

    _require(
        isinstance(source.get("requested_revision"), str) and bool(source["requested_revision"]),
        "Invalid requested revision",
    )

    revision = source.get("revision")
    _require(
        isinstance(revision, str) and _SHA_RE.fullmatch(revision) is not None,
        "Source revision must be a lowercase 40-character SHA",
    )

    bundle_count = source.get("bundle_count")
    _require(isinstance(bundle_count, int) and bundle_count > 0, "Invalid bundle count")

    # Verify bundles
    bundles = catalog.get("bundles")
    _require(isinstance(bundles, list), "Catalog bundles must be a list")
    assert isinstance(bundles, list)
    _require(
        len(bundles) == bundle_count,
        f"Bundle count mismatch: expected {bundle_count}, got {len(bundles)}",
    )

    # Track aliases globally
    all_aliases: dict[str, str] = {}

    for bundle in bundles:
        _require(isinstance(bundle, dict), "Bundle must be an object")
        bundle = cast(dict[str, Any], bundle)

        bundle_id = bundle.get("id")
        _require(isinstance(bundle_id, str) and bool(bundle_id), "Bundle must have a non-empty id")
        assert isinstance(bundle_id, str)
        _safe_name(bundle_id, f"bundle {bundle_id}")
        aliases = bundle.get("aliases", [])
        _require(
            isinstance(aliases, list) and all(isinstance(a, str) for a in aliases),
            f"{bundle_id}: aliases must be strings",
        )
        assert isinstance(aliases, list)

        # Check alias uniqueness
        for alias in aliases:
            _require(
                alias not in all_aliases or all_aliases[alias] == bundle_id,
                f"Alias {alias!r} is ambiguous: {all_aliases.get(alias)} vs {bundle_id}",
            )
            all_aliases[alias] = bundle_id

        sample_rate = bundle.get("sample_rate")
        _require(
            isinstance(sample_rate, int) and sample_rate > 0, f"{bundle_id}: invalid sample_rate"
        )

        artifacts = bundle.get("artifacts")
        _require(isinstance(artifacts, list), f"{bundle_id}: artifacts must be a list")
        assert isinstance(artifacts, list)
        # Track roles in this bundle
        seen_static: set[str] = set()
        seen_role_quality: set[tuple[str, str | None]] = set()

        for artifact in artifacts:
            _require(isinstance(artifact, dict), f"{bundle_id}: artifact must be an object")
            artifact = cast(dict[str, Any], artifact)

            role = artifact.get("role")
            _require(role in VALID_ROLES, f"{bundle_id}: invalid role {role!r}")
            assert isinstance(role, str)
            filename = artifact.get("filename")
            _require(
                isinstance(filename, str) and bool(filename),
                f"{bundle_id}/{role}: invalid filename",
            )
            assert isinstance(filename, str)
            _safe_name(filename, f"{bundle_id}/{role}: filename")
            url = artifact.get("url")
            _require(isinstance(url, str) and bool(url), f"{bundle_id}/{role}: invalid url")
            assert isinstance(url, str)
            # Check URL matches repository/revision
            expected_prefix = f"https://huggingface.co/{repository}/resolve/{revision}/"
            _require(
                url.startswith(expected_prefix),
                f"{bundle_id}/{role}: URL does not match repository/revision",
            )

            # Track role/quality pairs
            quality: str | None = artifact.get("quality")
            role_quality = (role, quality)
            _require(
                role_quality not in seen_role_quality,
                f"{bundle_id}: duplicate (role, quality) pair: {role_quality}",
            )
            seen_role_quality.add(role_quality)

            if role in STATIC_ROLES:
                _require(role not in seen_static, f"{bundle_id}: duplicate static role {role!r}")
                seen_static.add(role)

        # Check required static roles
        for role in STATIC_ROLES:
            _require(role in seen_static, f"{bundle_id}: missing static role {role!r}")

        # Check at least one quality for each required ONNX role
        for role in REQUIRED_ONNX_ROLES:
            qualities = {q for r, q in seen_role_quality if r == role}
            _require(len(qualities) > 0, f"{bundle_id}: no quality variant for role {role!r}")

        # Verify profiles
        profiles = bundle.get("profiles", {})
        _require(isinstance(profiles, dict), f"{bundle_id}: profiles must be a dict")

        for profile_name, profile_roles in profiles.items():
            _require(
                isinstance(profile_roles, dict),
                f"{bundle_id}: profile {profile_name!r} must be a dict",
            )
            for role, quality in profile_roles.items():
                _require(
                    role in REQUIRED_ONNX_ROLES,
                    f"{bundle_id}: profile {profile_name!r} references unknown role {role!r}",
                )
                _require(
                    (role, quality) in seen_role_quality,
                    f"{bundle_id}: profile {profile_name!r} references missing ({role!r}, {quality!r})",
                )


def load_catalog(path: Path) -> dict[str, Any]:
    """Load and verify a Pocket catalog from disk."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CatalogError(f"Unable to load catalog: {path}") from exc
    verify_catalog(data)
    return data
