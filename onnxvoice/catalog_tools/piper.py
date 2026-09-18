"""Fetch, normalize, and validate the upstream Piper voice inventory."""

from __future__ import annotations

import hashlib
import json
import re
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, cast

DEFAULT_REPOSITORY = "rhasspy/piper-voices"
DEFAULT_REVISION = "main"
USER_AGENT = "onnxvoice/0.1.0"
_SHA_RE = re.compile(r"^[0-9a-f]{40}$")
_DIGEST_RE = re.compile(r"^[0-9a-f]{32}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_SAFE_NAME_RE = re.compile(r"^[^\W_][\w.-]*$", re.UNICODE)


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


def _fetch_upstream_catalog_metadata(
    *, repository: str = DEFAULT_REPOSITORY, revision: str = DEFAULT_REVISION
) -> tuple[dict[str, Any], str, str, str]:
    resolved_revision = resolve_revision(repository, revision)
    url = huggingface_resolve_url(repository, resolved_revision, "voices.json")
    payload = _read_url(url)
    try:
        data = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CatalogError("Upstream voices.json is not valid UTF-8 JSON") from exc
    _require(isinstance(data, dict) and bool(data), "Upstream voices.json is empty")
    return data, resolved_revision, url, hashlib.sha256(payload).hexdigest()


def fetch_upstream_catalog(
    *, repository: str = DEFAULT_REPOSITORY, revision: str = DEFAULT_REVISION
) -> tuple[dict[str, Any], str, str]:
    """Return upstream voices.json, resolved revision, and source URL."""
    data, resolved_revision, url, _ = _fetch_upstream_catalog_metadata(
        repository=repository, revision=revision
    )
    return data, resolved_revision, url


def _pick_artifact_paths(files: dict[str, Any], voice_id: str) -> dict[str, str]:
    roles: dict[str, list[str]] = {"model_card": [], "model": [], "config": []}
    for path in files:
        if path.endswith("/MODEL_CARD") or path == "MODEL_CARD":
            roles["model_card"].append(path)
        elif path.endswith(".onnx.json"):
            roles["config"].append(path)
        elif path.endswith(".onnx"):
            roles["model"].append(path)
    for role, matches in roles.items():
        _require(
            len(matches) == 1,
            f"{voice_id}: expected exactly one {role} artifact, found {len(matches)}",
        )
    return {role: paths[0] for role, paths in roles.items()}


def _artifact(
    *, role: str, path: str, metadata: dict[str, Any], repository: str, revision: str
) -> dict[str, Any]:
    size = metadata.get("size_bytes")
    md5 = metadata.get("md5_digest")
    if not isinstance(size, int) or isinstance(size, bool) or size <= 0:
        raise CatalogError(f"{path}: invalid size_bytes")
    if not isinstance(md5, str) or _DIGEST_RE.fullmatch(md5.lower()) is None:
        raise CatalogError(f"{path}: invalid md5_digest")
    return {
        "role": role,
        "path": path,
        "filename": Path(path).name,
        "url": huggingface_resolve_url(repository, revision, path),
        "size": size,
        "md5": md5.lower(),
    }


def build_catalog(
    upstream: dict[str, Any],
    *,
    repository: str = DEFAULT_REPOSITORY,
    requested_revision: str = DEFAULT_REVISION,
    resolved_revision: str | None = None,
    catalog_url: str | None = None,
    catalog_sha256: str | None = None,
) -> dict[str, Any]:
    """Normalize all entries from upstream voices.json into the catalog contract."""
    _require(isinstance(upstream, dict) and bool(upstream), "Upstream catalog is empty")
    revision = (resolved_revision or requested_revision).lower()
    _require(isinstance(revision, str), "Catalog revision must be a string")
    voices: dict[str, Any] = {}
    for voice_id in sorted(upstream):
        _safe_name(voice_id, "voice id")
        item = upstream[voice_id]
        _require(isinstance(item, dict), f"{voice_id}: entry must be an object")
        _require(item.get("key") == voice_id, f"{voice_id}: key does not match map key")
        files = item.get("files")
        _require(isinstance(files, dict), f"{voice_id}: files must be an object")
        paths = _pick_artifact_paths(files, voice_id)
        language = item.get("language")
        _require(isinstance(language, dict), f"{voice_id}: language must be an object")
        name = item.get("name")
        quality = item.get("quality")
        num_speakers = item.get("num_speakers")
        _require(isinstance(name, str) and bool(name), f"{voice_id}: invalid name")
        _require(isinstance(quality, str) and bool(quality), f"{voice_id}: invalid quality")
        _require(
            isinstance(num_speakers, int)
            and not isinstance(num_speakers, bool)
            and num_speakers >= 1,
            f"{voice_id}: invalid num_speakers",
        )
        speaker_id_map = item.get("speaker_id_map") or {}
        aliases = item.get("aliases") or []
        _require(
            isinstance(speaker_id_map, dict),
            f"{voice_id}: speaker_id_map must be an object",
        )
        _require(
            isinstance(aliases, list) and all(isinstance(alias, str) for alias in aliases),
            f"{voice_id}: aliases must be strings",
        )
        voices[voice_id] = {
            "id": voice_id,
            "name": name,
            "language": language,
            "quality": quality,
            "num_speakers": num_speakers,
            "speaker_id_map": speaker_id_map,
            "aliases": aliases,
            "artifacts": {
                role: _artifact(
                    role=role,
                    path=paths[role],
                    metadata=files[paths[role]],
                    repository=repository,
                    revision=revision,
                )
                for role in ("model_card", "model", "config")
            },
        }
    return {
        "schema": 1,
        "kind": "piper-voice-catalog",
        "source": {
            "provider": "huggingface",
            "repository": repository,
            "requested_revision": requested_revision,
            "revision": revision,
            "catalog_url": catalog_url
            or huggingface_resolve_url(repository, revision, "voices.json"),
            "catalog_sha256": catalog_sha256
            or hashlib.sha256(_canonical_json(upstream)).hexdigest(),
            "upstream_catalog_path": "voices.json",
            "voice_count": len(voices),
        },
        "voices": voices,
    }


def fetch_and_build_catalog(
    *, repository: str = DEFAULT_REPOSITORY, revision: str = DEFAULT_REVISION
) -> dict[str, Any]:
    upstream, resolved_revision, source_url, catalog_sha256 = _fetch_upstream_catalog_metadata(
        repository=repository, revision=revision
    )
    return build_catalog(
        upstream,
        repository=repository,
        requested_revision=revision,
        resolved_revision=resolved_revision,
        catalog_url=source_url,
        catalog_sha256=catalog_sha256,
    )


def load_catalog(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CatalogError(f"Unable to load catalog: {path}") from exc
    verify_catalog(data)
    return data


def _verify_language(voice_id: str, language: Any) -> None:
    _require(isinstance(language, dict), f"{voice_id}: language must be an object")
    _require(
        isinstance(language.get("code"), str) and bool(language["code"]),
        f"{voice_id}: invalid language code",
    )
    _require(
        isinstance(language.get("family"), str) and bool(language["family"]),
        f"{voice_id}: invalid language family",
    )
    for key, value in language.items():
        _require(isinstance(key, str), f"{voice_id}: language keys must be strings")
        _require(
            isinstance(value, str) and bool(value),
            f"{voice_id}: invalid language field {key}",
        )


def _verify_artifact(
    source: dict[str, Any], voice_id: str, role: str, artifact: Any, seen_urls: set[str]
) -> None:
    _require(isinstance(artifact, dict), f"{voice_id}/{role}: artifact must be an object")
    artifact = cast(dict[str, Any], artifact)
    _require(
        set(artifact) == {"role", "path", "filename", "url", "size", "md5"},
        f"{voice_id}/{role}: invalid artifact fields",
    )
    _require(artifact.get("role") == role, f"{voice_id}/{role}: wrong role")
    path = cast(str, artifact.get("path"))
    filename = cast(str, artifact.get("filename"))
    _require(isinstance(path, str) and bool(path), f"{voice_id}/{role}: invalid path")
    _require(
        "\\" not in path
        and not path.startswith("/")
        and all(part not in {"", ".", ".."} for part in path.split("/")),
        f"{voice_id}/{role}: unsafe path",
    )
    _safe_name(filename, f"{voice_id}/{role}: filename")
    _require(filename == Path(path).name, f"{voice_id}/{role}: filename does not match path")
    if role == "model_card":
        _require(filename == "MODEL_CARD", f"{voice_id}/{role}: filename must be MODEL_CARD")
    elif role == "model":
        _require(
            filename.endswith(".onnx") and not filename.endswith(".onnx.json"),
            f"{voice_id}/{role}: invalid model filename",
        )
    else:
        _require(
            filename.endswith(".onnx.json"),
            f"{voice_id}/{role}: invalid config filename",
        )
    size = artifact.get("size")
    if not isinstance(size, int) or isinstance(size, bool) or size <= 0:
        raise CatalogError(f"{voice_id}/{role}: invalid size")
    md5 = artifact.get("md5")
    if not isinstance(md5, str) or _DIGEST_RE.fullmatch(md5) is None:
        raise CatalogError(f"{voice_id}/{role}: invalid md5")
    url = cast(str, artifact.get("url"))
    expected_url = huggingface_resolve_url(source["repository"], source["revision"], path)
    _require(
        url == expected_url,
        f"{voice_id}/{role}: URL does not match pinned source path",
    )
    _require(url not in seen_urls, f"Duplicate artifact URL: {url}")
    seen_urls.add(url)


def verify_catalog(catalog: dict[str, Any]) -> None:
    _require(isinstance(catalog, dict), "Catalog must be an object")
    _require(
        set(catalog) == {"schema", "kind", "source", "voices"},
        "Catalog has invalid fields",
    )
    _require(catalog.get("schema") == 1, "Catalog schema must be 1")
    _require(catalog.get("kind") == "piper-voice-catalog", "Unexpected catalog kind")
    source = catalog.get("source")
    _require(isinstance(source, dict), "Catalog source must be an object")
    source = cast(dict[str, Any], source)
    _require(
        set(source)
        == {
            "provider",
            "repository",
            "requested_revision",
            "revision",
            "catalog_url",
            "catalog_sha256",
            "upstream_catalog_path",
            "voice_count",
        },
        "Catalog source has invalid fields",
    )
    _require(
        source.get("provider") == "huggingface",
        "Catalog source provider must be huggingface",
    )
    repository = cast(str, source.get("repository"))
    _require(
        isinstance(repository, str) and re.fullmatch(r"[^/\\ ]+/[^/\\ ]+", repository) is not None,
        "Invalid source repository",
    )
    _require(
        isinstance(source.get("requested_revision"), str) and bool(source["requested_revision"]),
        "Invalid requested revision",
    )
    _require(
        isinstance(source.get("revision"), str)
        and _SHA_RE.fullmatch(source["revision"]) is not None,
        "Source revision must be a lowercase 40-character SHA",
    )
    _require(
        source.get("upstream_catalog_path") == "voices.json",
        "Invalid upstream catalog path",
    )
    _require(
        isinstance(source.get("catalog_sha256"), str)
        and _SHA256_RE.fullmatch(source["catalog_sha256"]) is not None,
        "Invalid catalog SHA-256",
    )
    _require(
        isinstance(source.get("voice_count"), int) and source["voice_count"] > 0,
        "Invalid voice count",
    )
    catalog_url = source.get("catalog_url")
    _require(
        catalog_url == huggingface_resolve_url(repository, source["revision"], "voices.json"),
        "Catalog URL is not pinned to the source revision",
    )
    voices = catalog.get("voices")
    _require(isinstance(voices, dict) and bool(voices), "Catalog has no voices")
    voices = cast(dict[str, Any], voices)
    _require(
        source["voice_count"] == len(voices),
        "Catalog source voice count does not match voices",
    )
    seen_urls: set[str] = set()
    canonical_ids = set(voices)
    aliases: dict[str, str] = {}
    for voice_id, voice in voices.items():
        _safe_name(voice_id, "voice id")
        _require(isinstance(voice, dict), f"{voice_id}: voice must be an object")
        voice = cast(dict[str, Any], voice)
        _require(
            set(voice)
            == {
                "id",
                "name",
                "language",
                "quality",
                "num_speakers",
                "speaker_id_map",
                "aliases",
                "artifacts",
            },
            f"{voice_id}: invalid voice fields",
        )
        _require(voice.get("id") == voice_id, f"{voice_id}: mismatched id")
        _require(
            isinstance(voice.get("name"), str) and bool(voice["name"]),
            f"{voice_id}: invalid name",
        )
        _verify_language(voice_id, voice.get("language"))
        _require(
            isinstance(voice.get("quality"), str) and bool(voice["quality"]),
            f"{voice_id}: invalid quality",
        )
        num_speakers = voice.get("num_speakers")
        _require(
            isinstance(num_speakers, int)
            and not isinstance(num_speakers, bool)
            and num_speakers >= 1,
            f"{voice_id}: invalid num_speakers",
        )
        num_speakers = cast(int, num_speakers)
        speaker_map = voice.get("speaker_id_map")
        _require(
            isinstance(speaker_map, dict),
            f"{voice_id}: speaker_id_map must be an object",
        )
        speaker_map = cast(dict[str, Any], speaker_map)
        speaker_values: set[int] = set()
        for speaker_name, speaker_id in speaker_map.items():
            _require(
                isinstance(speaker_name, str),
                f"{voice_id}: speaker map keys must be strings",
            )
            _require(
                isinstance(speaker_id, int)
                and not isinstance(speaker_id, bool)
                and 0 <= speaker_id < num_speakers,
                f"{voice_id}: invalid speaker id",
            )
            _require(
                speaker_id not in speaker_values,
                f"{voice_id}: duplicate numeric speaker id",
            )
            speaker_values.add(speaker_id)
        voice_aliases = voice.get("aliases")
        _require(
            isinstance(voice_aliases, list)
            and all(isinstance(alias, str) for alias in voice_aliases),
            f"{voice_id}: aliases must be strings",
        )
        voice_aliases = cast(list[Any], voice_aliases)
        _require(
            len(voice_aliases) == len(set(voice_aliases)),
            f"{voice_id}: duplicate aliases",
        )
        for alias in voice_aliases:
            _safe_name(alias, f"{voice_id}: alias")
            if alias in canonical_ids:
                _require(
                    alias == voice_id,
                    f"{voice_id}: alias collides with canonical id {alias}",
                )
            previous = aliases.setdefault(alias, voice_id)
            _require(previous == voice_id, f"Alias is ambiguous: {alias}")
        artifacts = voice.get("artifacts")
        _require(
            isinstance(artifacts, dict) and set(artifacts) == {"model_card", "model", "config"},
            f"{voice_id}: voice must expose exactly model_card/model/config",
        )
        artifacts = cast(dict[str, Any], artifacts)
        for role in ("model_card", "model", "config"):
            _verify_artifact(source, voice_id, role, artifacts[role], seen_urls)


def get_voice(catalog: dict[str, Any], voice_id_or_alias: str) -> dict[str, Any]:
    voices = catalog["voices"]
    if voice_id_or_alias in voices:
        return voices[voice_id_or_alias]
    matches = [voice for voice in voices.values() if voice_id_or_alias in voice.get("aliases", [])]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise CatalogError(f"Alias is ambiguous: {voice_id_or_alias}")
    raise CatalogError(f"Unknown Piper voice: {voice_id_or_alias}")


def list_voices(
    catalog: dict[str, Any], *, language: str | None = None, quality: str | None = None
) -> list[dict[str, Any]]:
    values = list(catalog["voices"].values())
    if language:
        wanted = language.casefold()
        values = [
            voice
            for voice in values
            if str(voice.get("language", {}).get("code", "")).casefold() == wanted
            or str(voice.get("language", {}).get("family", "")).casefold() == wanted
        ]
    if quality:
        wanted_quality = quality.casefold()
        values = [
            voice for voice in values if voice.get("quality", "").casefold() == wanted_quality
        ]
    return sorted(values, key=lambda voice: voice["id"])
