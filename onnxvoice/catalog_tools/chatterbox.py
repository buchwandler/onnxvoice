"""Build and verify deterministic, revision-pinned Chatterbox ONNX bundle catalogs."""

from __future__ import annotations

import hashlib
import json
import re
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Mapping
from pathlib import Path, PurePosixPath
from typing import Any, cast

from ..catalog import _parse_chatterbox
from ..errors import CatalogError as OnnxCatalogError

DEFAULT_SEED_PATH = Path(__file__).with_name("chatterbox_seed.json")
DEFAULT_CATALOG_URL = "https://raw.githubusercontent.com/buchwandler/chatterbox-onnx-bundles/main/catalog/bundles.json"
USER_AGENT = "onnxvoice-chatterbox-catalog"
_SHA_RE = re.compile(r"^[0-9a-f]{40}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_SAFE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
_COMPONENTS = ("speech_encoder", "embed_tokens", "language_model", "conditional_decoder")


class CatalogError(ValueError):
    """Raised when a Chatterbox seed, upstream tree, or generated catalog is invalid."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise CatalogError(message)


def _safe_path(value: Any, label: str) -> str:
    _require(isinstance(value, str) and bool(value), f"{label}: path is required")
    _require(not value.startswith("/") and "\\" not in value, f"{label}: unsafe path")
    parts = PurePosixPath(value).parts
    _require(all(part not in {"", ".", ".."} for part in parts), f"{label}: unsafe path")
    _require(not re.match(r"^[A-Za-z]:", value), f"{label}: absolute paths are forbidden")
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
        with urllib.request.urlopen(request, timeout=120) as response:
            return response.read()
    except (OSError, urllib.error.URLError) as exc:
        raise CatalogError(f"Unable to fetch {url}") from exc


def resolve_revision(repository: str, revision: str) -> str:
    """Resolve a branch or tag to an immutable lowercase 40-character commit."""
    if _SHA_RE.fullmatch(revision):
        return revision
    try:
        payload = json.loads(_read_url(huggingface_api_url(repository, revision)).decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CatalogError("Hugging Face revision response is not valid JSON") from exc
    resolved = payload.get("sha") if isinstance(payload, Mapping) else None
    _require(
        isinstance(resolved, str) and _SHA_RE.fullmatch(resolved) is not None,
        "Unable to resolve an exact 40-character commit SHA",
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
            path = _safe_path(entry["path"], "upstream artifact")
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
        raise CatalogError(f"Unable to read artifact integrity data for {path}") from exc
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
        sha256 is None or sha256 == downloaded_sha, f"{path}: SHA-256 disagrees with upstream bytes"
    )
    return downloaded_size, downloaded_sha


def _validate_seed(seed: Any) -> Mapping[str, Any]:
    _require(isinstance(seed, Mapping), "Chatterbox seed must be an object")
    _require(
        type(seed.get("schema")) is int and seed.get("schema") == 1,
        "Chatterbox seed schema must be 1",
    )
    _require(seed.get("kind") == "chatterbox-onnx-bundle-seed", "Unexpected Chatterbox seed kind")
    bundles = seed.get("bundles")
    _require(
        isinstance(bundles, Mapping) and bool(bundles), "Seed bundles must be a non-empty mapping"
    )
    ids = set(bundles)
    aliases: set[str] = set(ids)
    for bundle_id, bundle in bundles.items():
        _require(
            isinstance(bundle_id, str) and _SAFE_ID_RE.fullmatch(bundle_id) is not None,
            f"Unsafe Chatterbox bundle ID {bundle_id!r}",
        )
        _require(isinstance(bundle, Mapping), f"{bundle_id}: seed bundle must be an object")
        _require(
            bundle.get("id", bundle_id) == bundle_id, f"{bundle_id}: ID does not match map key"
        )
        raw_aliases = bundle.get("aliases", [])
        _require(isinstance(raw_aliases, list), f"{bundle_id}: aliases must be a list")
        for alias in raw_aliases:
            _require(
                isinstance(alias, str) and _SAFE_ID_RE.fullmatch(alias) is not None,
                f"{bundle_id}: unsafe alias {alias!r}",
            )
            _require(alias not in aliases, f"Duplicate Chatterbox alias {alias!r}")
            aliases.add(alias)
        source = bundle.get("source")
        _require(isinstance(source, Mapping), f"{bundle_id}: source must be an object")
        repository = source.get("repository")
        _require(
            isinstance(repository, str) and re.fullmatch(r"[^/\\\s]+/[^/\\\s]+", repository),
            f"{bundle_id}: invalid Hugging Face repository",
        )
        _require(
            source.get("provider") == "huggingface", f"{bundle_id}: provider must be huggingface"
        )
        _require(
            isinstance(source.get("requested_revision"), str) and source["requested_revision"],
            f"{bundle_id}: requested_revision is required",
        )
        _require(
            isinstance(bundle.get("language"), str) and bundle["language"],
            f"{bundle_id}: language is required",
        )
        _require(
            type(bundle.get("sample_rate")) is int and bundle["sample_rate"] > 0,
            f"{bundle_id}: sample_rate must be positive",
        )
        frontend = bundle.get("frontend")
        _require(isinstance(frontend, Mapping), f"{bundle_id}: frontend is required")
        _safe_path(frontend.get("tokenizer_path"), f"{bundle_id} tokenizer")
        _require(
            isinstance(frontend.get("tokenizer_contract"), str) and frontend["tokenizer_contract"],
            f"{bundle_id}: tokenizer_contract is required",
        )
        runtime = bundle.get("runtime")
        _require(isinstance(runtime, Mapping), f"{bundle_id}: runtime metadata is required")
        profiles = bundle.get("profiles")
        _require(
            isinstance(profiles, Mapping) and bool(profiles), f"{bundle_id}: profiles are required"
        )
        _require(
            bundle.get("default_profile") in profiles,
            f"{bundle_id}: default_profile must name a profile",
        )
        for profile_name, profile in profiles.items():
            _require(
                isinstance(profile_name, str) and _SAFE_ID_RE.fullmatch(profile_name) is not None,
                f"{bundle_id}: unsafe profile {profile_name!r}",
            )
            _require(
                isinstance(profile, Mapping) and set(profile) == set(_COMPONENTS),
                f"{bundle_id}/{profile_name}: must map exactly the four graph components",
            )
            for component, declaration in profile.items():
                _require(
                    isinstance(declaration, Mapping),
                    f"{bundle_id}/{profile_name}/{component}: graph declaration must be an object",
                )
                _safe_path(declaration.get("path"), f"{bundle_id}/{profile_name}/{component}")
                _require(
                    isinstance(declaration.get("quality"), str) and declaration["quality"],
                    f"{bundle_id}/{profile_name}/{component}: quality is required",
                )
        static = bundle.get("static_artifacts", {})
        _require(isinstance(static, Mapping), f"{bundle_id}: static_artifacts must be an object")
        for role, path in static.items():
            _require(
                isinstance(role, str) and role not in {"graph", "external_data"},
                f"{bundle_id}: invalid static artifact role {role!r}",
            )
            _safe_path(path, f"{bundle_id}/{role}")
    return cast(Mapping[str, Any], bundles)


def _resolve_artifact(
    repository: str, revision: str, path: str, tree: Mapping[str, Mapping[str, Any]]
) -> dict[str, Any]:
    _require(path in tree, f"Requested upstream artifact does not exist: {path}")
    metadata = tree[path]
    size, sha256 = _artifact_integrity(repository, revision, path, metadata)
    return {
        "path": path,
        "url": huggingface_resolve_url(repository, revision, path),
        "size": size,
        "sha256": sha256,
    }


def build_catalog(seed: Any | None = None) -> dict[str, Any]:
    """Resolve seed revisions and materialize a normalized deterministic catalog."""
    if seed is None:
        try:
            seed = json.loads(DEFAULT_SEED_PATH.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise CatalogError(f"Unable to load Chatterbox seed: {DEFAULT_SEED_PATH}") from exc
    seeds = _validate_seed(seed)
    bundles: dict[str, Any] = {}
    for bundle_id in sorted(seeds):
        spec = seeds[bundle_id]
        source_spec = spec["source"]
        repository = source_spec["repository"]
        requested = source_spec["requested_revision"]
        revision = resolve_revision(repository, requested)
        tree = _repository_tree(repository, revision)
        artifacts: dict[tuple[str, str | None, str | None], dict[str, Any]] = {}

        def add(
            role: str,
            component: str | None,
            quality: str | None,
            path: str,
            *,
            artifact_map: dict[tuple[str, str | None, str | None], dict[str, Any]] = artifacts,
            bundle_name: str = bundle_id,
            source_repository: str = repository,
            source_revision: str = revision,
            repository_tree: Mapping[str, Mapping[str, Any]] = tree,
        ) -> None:
            key = (role, component, quality)
            existing = artifact_map.get(key)
            if existing is not None:
                _require(
                    existing["source_path"] == path,
                    f"{bundle_name}: duplicate artifact role/component/profile {key!r}",
                )
                return
            record = _resolve_artifact(source_repository, source_revision, path, repository_tree)
            artifact_map[key] = {
                "role": role,
                "component": component,
                "quality": quality,
                "filename": PurePosixPath(path).name,
                "source_path": path,
                "url": record["url"],
                "size": record["size"],
                "sha256": record["sha256"],
            }

        add("tokenizer", None, None, spec["frontend"]["tokenizer_path"])
        for role, path in sorted(spec.get("static_artifacts", {}).items()):
            add(role, None, None, path)
        profile_maps: dict[str, dict[str, str]] = {}
        for profile_name, profile in sorted(spec["profiles"].items()):
            profile_maps[profile_name] = {}
            for component in _COMPONENTS:
                declaration = profile[component]
                path, quality = declaration["path"], declaration["quality"]
                add("graph", component, quality, path)
                companions = [
                    candidate for candidate in (path + "_data", path + ".data") if candidate in tree
                ]
                _require(
                    len(companions) <= 1,
                    f"{bundle_id}/{profile_name}/{component}: ambiguous external-data companions",
                )
                if companions:
                    add("external_data", component, quality, companions[0])
                profile_maps[profile_name][component] = quality
        ordered_artifacts = sorted(
            artifacts.values(),
            key=lambda row: (
                row["role"],
                row["component"] or "",
                row["quality"] or "",
                row["source_path"],
            ),
        )
        bundle = {
            "id": bundle_id,
            "system": "chatterbox",
            "aliases": sorted(spec.get("aliases", [])),
            "language": spec["language"],
            "sample_rate": spec["sample_rate"],
            "default_profile": spec["default_profile"],
            "source": {
                "provider": "huggingface",
                "repository": repository,
                "requested_revision": requested,
                "revision": revision,
            },
            "frontend": dict(spec["frontend"]),
            "runtime": dict(spec["runtime"]),
            "profiles": profile_maps,
            "artifacts": ordered_artifacts,
        }
        bundles[bundle_id] = bundle
    catalog = {"schema": 1, "kind": "chatterbox-onnx-bundle-catalog", "bundles": bundles}
    verify_catalog(catalog)
    return catalog


def build_source(catalog: Mapping[str, Any]) -> dict[str, Any]:
    """Build deterministic source provenance and counts for a normalized catalog."""
    try:
        _parse_chatterbox(catalog)
    except (TypeError, ValueError, OnnxCatalogError) as exc:
        raise CatalogError(str(exc)) from exc
    bundles = catalog["bundles"]
    sources = []
    artifact_count = 0
    for bundle_id, bundle in sorted(bundles.items()):
        artifact_count += len(bundle["artifacts"])
        sources.append({"id": bundle_id, **dict(bundle["source"])})
    return {
        "schema": 1,
        "kind": "chatterbox-onnx-source-snapshot",
        "bundle_count": len(bundles),
        "artifact_count": artifact_count,
        "sources": sources,
    }


def verify_catalog(catalog: Any, source: Any | None = None) -> None:
    """Verify normalized catalog structure and optional source summary offline."""
    try:
        _parse_chatterbox(catalog)
    except (TypeError, ValueError, OnnxCatalogError) as exc:
        raise CatalogError(str(exc)) from exc
    if source is not None:
        _require(isinstance(source, Mapping), "Chatterbox source metadata must be an object")
        expected = build_source(cast(Mapping[str, Any], catalog))
        _require(source == expected, "Source metadata does not match catalog revisions/counts")


def load_catalog(path: Path, source_path: Path | None = None) -> dict[str, Any]:
    """Load and verify a generated catalog and optional provenance file."""
    try:
        catalog = json.loads(path.read_text(encoding="utf-8"))
        source = json.loads(source_path.read_text(encoding="utf-8")) if source_path else None
    except (OSError, json.JSONDecodeError) as exc:
        raise CatalogError(f"Unable to read Chatterbox catalog: {path}") from exc
    verify_catalog(catalog, source)
    return cast(dict[str, Any], catalog)


def canonical_json(data: Any) -> bytes:
    """Serialize JSON in deterministic compact key order for reproducible generation."""
    return (
        json.dumps(data, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode()
