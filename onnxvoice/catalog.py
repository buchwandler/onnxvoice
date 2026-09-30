from __future__ import annotations

import json
import os
import re
import time
import urllib.parse
import urllib.request
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from platformdirs import user_cache_path

from .errors import AssetNotFoundError, CatalogError, OfflineError
from .inventory import VALID_GENDERS, language_base, normalize_language_tag
from .store import FileLock, ProgressCallback
from .types import (
    Artifact,
    AssetProgress,
    CatalogItem,
    validate_relative_path,
    validate_safe_component,
)

POCKET_ARTIFACT_ROLES = frozenset(
    {
        "bundle_metadata",
        "tokenizer",
        "bos_conditioning",
        "flow_lm_main",
        "flow_lm_flow",
        "mimi_decoder",
        "mimi_encoder",
        "text_conditioner",
    }
)
_POCKET_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_POCKET_HF_REPOSITORY_RE = re.compile(r"^[^/\\\s]+/[^/\\\s]+$")
_POCKET_HF_REVISION_RE = re.compile(r"^[0-9a-f]{40}$")
_ST_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_ST_SHA_RE = re.compile(r"^[0-9a-f]{40}$")
_ST_REPOSITORY_RE = re.compile(r"^[^/\\\s]+/[^/\\\s]+$")
_ST_SAFE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
ST_MODEL_COMPONENTS = frozenset(
    {"duration_predictor", "text_encoder", "vector_estimator", "vocoder"}
)
ST_ARTIFACT_ROLES = frozenset({"config", "unicode_indexer", "model", "voice_style"})

POCKET_QUALIFIED_ROLES = frozenset(
    {
        "flow_lm_main",
        "flow_lm_flow",
        "mimi_decoder",
        "mimi_encoder",
        "text_conditioner",
    }
)

DEFAULT_SOURCES = {
    "piper": "https://raw.githubusercontent.com/buchwandler/piper-onnx-voices/main/catalog/voices.json",
    "kokoro": "https://raw.githubusercontent.com/buchwandler/kokoro-onnx-models/main/catalog/models.json",
    "pocket": "https://raw.githubusercontent.com/buchwandler/pocket-onnx-bundles/main/catalog/bundles.json",
    "supertonic": "https://raw.githubusercontent.com/buchwandler/supertonic-onnx-bundles/main/catalog/bundles.json",
}


@dataclass(slots=True)
class CatalogClient:
    """Read and normalize configured system catalogs.


    A catalog load can read the network when cached data is stale or missing. In
    offline mode, only cached catalog data is available.
    """

    cache_dir: Path | None = None
    sources: dict[str, str] | None = None
    ttl_seconds: int = 24 * 60 * 60
    offline: bool = False

    def __post_init__(self) -> None:
        cache_root = Path(self.cache_dir or user_cache_path("onnxvoice")) / "catalogs"
        self.cache_dir = cache_root
        cache_root.mkdir(parents=True, exist_ok=True)
        merged = dict(DEFAULT_SOURCES)
        if self.sources:
            merged.update(self.sources)
        for system in tuple(merged):
            env_name = f"ONNXVOICE_{system.upper()}_CATALOG"
            if os.environ.get(env_name):
                merged[system] = os.environ[env_name]
        self.sources = merged

    def systems(self) -> tuple[str, ...]:
        """Return configured systems with catalog sources."""
        return tuple(sorted(self.sources or {}))

    def _cache_path(self, system: str) -> Path:
        assert self.cache_dir is not None
        return self.cache_dir / f"{system}.json"

    def _catalog_lock_path(self, system: str) -> Path:
        assert self.cache_dir is not None
        return self.cache_dir.parent / "locks" / "catalog" / f"{system}.lock"

    def _read_source(self, source: str) -> bytes:
        if source.startswith(("http://", "https://")):
            if self.offline:
                raise OfflineError(f"Catalog source requires network access: {source}")
            request = urllib.request.Request(source, headers={"User-Agent": "onnxvoice/0"})
            with urllib.request.urlopen(request, timeout=30) as response:
                return response.read()
        return Path(source).expanduser().read_bytes()

    def load_raw(
        self,
        system: str,
        *,
        refresh: bool = False,
        progress: ProgressCallback | None = None,
    ) -> dict[str, Any]:
        """Load raw catalog data from cache or its configured source.


        Network access occurs when the cache cannot satisfy the request and offline
        mode is disabled. ``refresh=True`` bypasses the normal cache freshness check.
        """
        system = system.lower()
        if not self.sources or system not in self.sources:
            raise CatalogError(f"No catalog source configured for system {system!r}")
        self._emit(progress, AssetProgress("catalog_started", ref=system))
        cache_path = self._cache_path(system)
        fresh = (
            cache_path.exists() and (time.time() - cache_path.stat().st_mtime) < self.ttl_seconds
        )
        if cache_path.exists() and (fresh or self.offline) and not refresh:
            data = json.loads(cache_path.read_text(encoding="utf-8"))
            self._emit(progress, AssetProgress("catalog_cached", ref=system))
            return data
        payload = self._read_source(self.sources[system])
        try:
            data = json.loads(payload)
        except json.JSONDecodeError as exc:
            raise CatalogError(f"Invalid JSON in {system} catalog") from exc
        with FileLock(self._catalog_lock_path(system)):
            tmp = cache_path.with_suffix(".json.tmp")
            tmp.write_bytes(payload)
            os.replace(tmp, cache_path)
        self._emit(progress, AssetProgress("catalog_completed", ref=system))
        return data

    def list(
        self,
        system: str,
        *,
        refresh: bool = False,
        progress: ProgressCallback | None = None,
    ) -> list[CatalogItem]:
        """Return parsed catalog items for a system."""
        system = system.lower()
        raw = self.load_raw(system, refresh=refresh, progress=progress)
        if system == "piper":
            return _parse_piper(raw)
        if system == "kokoro":
            return _parse_kokoro(raw)
        if system == "pocket":
            return _parse_pocket(raw)
        if system == "supertonic":
            return _parse_supertonic(raw)
        raise CatalogError(f"No parser for system {system!r}")

    def resolve(
        self,
        ref: str,
        *,
        refresh: bool = False,
        quality: str | None = None,
        distribution: str | None = None,
        progress: ProgressCallback | None = None,
    ) -> CatalogItem:
        """Resolve a canonical reference or alias to a selected catalog item.


        This may load the system catalog and can perform network I/O. Quality and
        distribution select system-specific artifact variants.
        """
        system, item_id = parse_ref(ref)
        if system == "kokoro":
            raw = self.load_raw(system, refresh=refresh, progress=progress)
            models = _kokoro_models(raw)
            entry = models.get(item_id)
            if entry is None:
                # Check aliases
                for mid, mentry in models.items():
                    aliases = mentry.get("aliases") or ()
                    if item_id in aliases:
                        entry = mentry
                        item_id = mid
                        break
            if entry is None:
                raise AssetNotFoundError(f"Unknown catalog item: {ref}")
            item = _parse_kokoro_entry(item_id, entry, distribution=distribution)
        else:
            items = self.list(system, refresh=refresh, progress=progress)
            item = None
            for candidate in items:
                if candidate.id == item_id or item_id in candidate.aliases:
                    item = candidate
                    break
            if item is None:
                raise AssetNotFoundError(f"Unknown catalog item: {ref}")
        assert item is not None
        if item.system == "supertonic" and quality is not None:
            raise AssetNotFoundError("Supertonic has no quality profiles")
        if item.system == "pocket":
            return _select_pocket_profile(item, quality)
        selected_quality = quality
        model_artifacts = [a for a in item.artifacts if a.role == "model"]
        if selected_quality is None and len(model_artifacts) > 1:
            selected_quality = (
                "fp32"
                if any(a.quality == "fp32" for a in model_artifacts)
                else model_artifacts[0].quality
            )
        if selected_quality is None:
            return item
        artifacts = tuple(
            artifact
            for artifact in item.artifacts
            if artifact.role != "model" or artifact.quality in {None, selected_quality}
        )
        if not any(a.role == "model" for a in artifacts):
            raise AssetNotFoundError(
                f"{ref} has no model artifact with quality={selected_quality!r}"
            )
        return CatalogItem(
            system=item.system,
            id=item.id,
            kind=item.kind,
            artifacts=artifacts,
            aliases=item.aliases,
            sample_rate=item.sample_rate,
            voices=item.voices,
            default_voice=item.default_voice,
            metadata={**item.metadata, "selected_quality": selected_quality},
        )

    @staticmethod
    def _emit(progress: ProgressCallback | None, event: AssetProgress) -> None:
        if progress is not None:
            progress(event)


def parse_ref(ref: str) -> tuple[str, str]:
    if ":" not in ref:
        raise ValueError(
            "Expected reference in the form 'system:id', e.g. piper:en_US-lessac-medium"
        )
    system, item_id = ref.split(":", 1)
    if not system or not item_id:
        raise ValueError(f"Invalid reference: {ref!r}")
    return system.lower(), item_id


def _parse_piper(data: dict[str, Any]) -> list[CatalogItem]:
    voices = data.get("voices")
    if not isinstance(voices, dict):
        raise CatalogError("Piper catalog is missing the 'voices' mapping")
    source = data.get("source") or {}
    if not isinstance(source, dict):
        raise CatalogError("Piper catalog source metadata must be an object")
    result: list[CatalogItem] = []
    for voice_id, entry in voices.items():
        artifacts: list[Artifact] = []
        for key, raw in (entry.get("artifacts") or {}).items():
            artifacts.append(
                Artifact(
                    role=str(raw.get("role") or key),
                    filename=str(raw.get("filename") or Path(str(raw.get("path", key))).name),
                    url=raw.get("url"),
                    size=raw.get("size"),
                    md5=raw.get("md5"),
                    quality=raw.get("quality"),
                    component=raw.get("component"),
                    format=raw.get("format"),
                    metadata={"path": raw.get("path"), **(raw.get("metadata") or {})},
                )
            )
        language = entry.get("language") or {}
        result.append(
            CatalogItem(
                system="piper",
                id=str(entry.get("id") or voice_id),
                kind="voice",
                artifacts=tuple(artifacts),
                aliases=tuple(entry.get("aliases") or ()),
                metadata={
                    "language": language,
                    "quality": entry.get("quality"),
                    "name": entry.get("name"),
                    "gender": entry.get("gender"),
                    "num_speakers": entry.get("num_speakers"),
                    "speaker_id_map": entry.get("speaker_id_map") or {},
                    "source_revision": source.get("revision"),
                    "source_repository": source.get("repository"),
                    "requested_revision": source.get("requested_revision"),
                },
            )
        )
    return result


def _normalize_kokoro_runtime(
    entry: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Copy and normalize Kokoro runtime and ONNX contract metadata."""
    raw_runtime = entry.get("runtime") or {}
    if not isinstance(raw_runtime, Mapping):
        raise CatalogError("Kokoro runtime metadata must be an object")
    runtime = dict(raw_runtime)
    raw_contract = entry.get("onnx_contract") or {}
    if not isinstance(raw_contract, Mapping):
        raise CatalogError("Kokoro ONNX contract metadata must be an object")
    onnx_contract = dict(raw_contract)
    raw_timing = onnx_contract.get("timing") or {}
    if not isinstance(raw_timing, Mapping):
        raise CatalogError("Kokoro ONNX timing metadata must be an object")
    runtime_output = runtime.get("timings_output")
    contract_output = raw_timing.get("output")
    if (
        isinstance(runtime_output, str)
        and runtime_output
        and isinstance(contract_output, str)
        and contract_output
        and runtime_output != contract_output
    ):
        raise CatalogError(
            "Conflicting Kokoro timing declarations: "
            f"runtime.timings_output={runtime_output!r} conflicts with "
            f"onnx_contract.timing.output={contract_output!r}"
        )
    if isinstance(contract_output, str) and contract_output:
        runtime["timings_output"] = contract_output
    return runtime, onnx_contract


def _parse_kokoro_entry(
    model_id: str,
    entry: Mapping[str, Any],
    *,
    distribution: str | None = None,
) -> CatalogItem:
    """Parse a single Kokoro catalog model entry.

    When *distribution* is ``None`` the first runtime-ready distribution
    is chosen (the catalog default).  When a distribution id is supplied it
    must match one of the model's own distributions — no other model's
    distributions are inspected.
    """
    raw_distributions = entry.get("distributions", ())
    distributions = [value for value in raw_distributions if value.get("runtime_ready", True)]
    if not distributions:
        raise CatalogError(f"Kokoro model {model_id!r} has no runtime-ready distributions")
    choices = tuple(str(value.get("id")) for value in distributions if value.get("id"))
    if distribution is None:
        selected = distributions[0]
    else:
        selected = next(
            (value for value in distributions if str(value.get("id")) == distribution),
            None,
        )
        if selected is None:
            raise CatalogError(
                f"Unknown Kokoro distribution {distribution!r} for {model_id!r}; "
                f"valid choices: {', '.join(choices)}"
            )
    artifacts: list[Artifact] = []
    for raw in selected.get("artifacts", ()):
        artifacts.append(
            Artifact(
                role=str(raw.get("role") or raw.get("id") or "artifact"),
                filename=str(raw.get("local_name") or raw.get("id")),
                url=raw.get("url"),
                size=raw.get("size"),
                sha256=raw.get("sha256"),
                quality=raw.get("quality"),
                component=raw.get("component"),
                format=raw.get("format"),
                metadata={
                    "id": raw.get("id"),
                    "handling": raw.get("handling"),
                    **(raw.get("metadata") or {}),
                },
            )
        )
    runtime, onnx_contract = _normalize_kokoro_runtime(entry)
    metadata = {
        "model_version": entry.get("model_version"),
        "frontend": entry.get("frontend"),
        "language_codes": entry.get("language_codes") or [],
        "runtime": runtime,
        "onnx_contract": onnx_contract,
        "distribution_id": selected.get("id"),
        "distribution_choices": choices,
        "release_tag": selected.get("release_tag"),
    }
    return CatalogItem(
        system="kokoro",
        id=str(model_id),
        kind="model",
        artifacts=tuple(artifacts),
        sample_rate=entry.get("sample_rate"),
        voices=tuple(runtime.get("voices") or ()),
        default_voice=runtime.get("default_voice"),
        metadata=metadata,
    )


def _parse_kokoro(data: dict[str, Any]) -> list[CatalogItem]:
    """Parse the full Kokoro catalog — each model uses its own default distribution."""
    models = data.get("models")
    if not isinstance(models, dict):
        raise CatalogError("Kokoro catalog is missing the 'models' mapping")
    result: list[CatalogItem] = []
    for model_id, entry in models.items():
        distributions = entry.get("distributions", ())
        if not any(distribution.get("runtime_ready", True) for distribution in distributions):
            continue
        result.append(_parse_kokoro_entry(model_id, entry))
    return result


def _kokoro_models(data: dict[str, Any]) -> dict[str, Mapping[str, Any]]:
    """Return the raw models mapping from Kokoro catalog data."""
    models = data.get("models")
    if not isinstance(models, dict):
        raise CatalogError("Kokoro catalog is missing the 'models' mapping")
    return models


def _validate_canonical_pocket_integrity(
    raw: Mapping[str, Any],
    bundle_id: str,
    role: str,
    *,
    canonical: bool,
) -> None:
    if not canonical:
        return
    size = raw.get("size")
    if not isinstance(size, int) or isinstance(size, bool) or size <= 0:
        raise CatalogError(f"{bundle_id}/{role}: canonical artifact size must be positive")
    sha256 = raw.get("sha256")
    if not isinstance(sha256, str) or _POCKET_SHA256_RE.fullmatch(sha256) is None:
        raise CatalogError(f"{bundle_id}/{role}: canonical artifact sha256 must be lowercase hex")


def _pocket_voice_source_url(repository: str, revision: str, path: str) -> str:
    repo = "/".join(urllib.parse.quote(part, safe="") for part in repository.split("/"))
    encoded_revision = urllib.parse.quote(revision, safe="")
    encoded_path = urllib.parse.quote(path, safe="/")
    return f"https://huggingface.co/{repo}/resolve/{encoded_revision}/{encoded_path}?download=true"


def _parse_pocket_voice_states(
    entry: Mapping[str, Any],
    bundle_id: str,
) -> tuple[tuple[str, ...], list[dict[str, Any]]]:
    raw_states = entry.get("voice_states", [])
    if raw_states in (None, []):
        return (), []
    if not isinstance(raw_states, Sequence) or isinstance(raw_states, (str, bytes)):
        raise CatalogError(f"{bundle_id}: voice_states must be a sequence")
    names: list[str] = []
    records: list[dict[str, Any]] = []
    for index, raw in enumerate(raw_states):
        if not isinstance(raw, Mapping):
            raise CatalogError(f"{bundle_id}: voice state {index} must be an object")
        name = raw.get("name")
        if not isinstance(name, str):
            raise CatalogError(f"{bundle_id}: voice state {index} has no name")
        _require_pocket_safe_id(name, f"{bundle_id} voice state name")
        if name in names:
            raise CatalogError(f"{bundle_id}: duplicate voice state {name!r}")
        if raw.get("compatible_bundle") != bundle_id:
            raise CatalogError(f"{bundle_id}/{name}: incompatible bundle")
        source = raw.get("source")
        if not isinstance(source, Mapping):
            raise CatalogError(f"{bundle_id}/{name}: source must be an object")
        for field in ("provider", "repository", "revision", "path"):
            if not isinstance(source.get(field), str) or not source[field]:
                raise CatalogError(f"{bundle_id}/{name}: source.{field} is required")
        if source["provider"] != "huggingface":
            raise CatalogError(f"{bundle_id}/{name}: source provider must be huggingface")
        source_repository = source["repository"]
        if _POCKET_HF_REPOSITORY_RE.fullmatch(source_repository) is None:
            raise CatalogError(f"{bundle_id}/{name}: invalid source repository")
        source_revision = source["revision"]
        if _POCKET_HF_REVISION_RE.fullmatch(source_revision) is None:
            raise CatalogError(f"{bundle_id}/{name}: source revision must be a 40-character SHA")
        source_path = source["path"]
        try:
            validate_relative_path(source_path, field_name="Pocket voice-state source path")
        except ValueError as exc:
            raise CatalogError(f"{bundle_id}/{name}: {exc}") from exc
        access = raw.get("access")
        if not isinstance(access, Mapping):
            raise CatalogError(f"{bundle_id}/{name}: access must be an object")
        if not isinstance(access.get("gated"), bool) or not isinstance(
            access.get("distributable"), bool
        ):
            raise CatalogError(f"{bundle_id}/{name}: access flags must be boolean")
        if not isinstance(access.get("license"), str) or not access["license"]:
            raise CatalogError(f"{bundle_id}/{name}: access.license is required")
        if not isinstance(raw.get("format"), str) or not raw["format"]:
            raise CatalogError(f"{bundle_id}/{name}: format is required")
        size = raw.get("size")
        if not isinstance(size, int) or isinstance(size, bool) or size <= 0:
            raise CatalogError(f"{bundle_id}/{name}: size must be positive")
        sha256 = raw.get("sha256")
        if not isinstance(sha256, str) or _POCKET_SHA256_RE.fullmatch(sha256) is None:
            raise CatalogError(f"{bundle_id}/{name}: sha256 must be lowercase hex")
        url = raw.get("url")
        resolver = raw.get("resolver")
        if url is not None and (not isinstance(url, str) or not url):
            raise CatalogError(f"{bundle_id}/{name}: url must be a non-empty string")
        if url is not None and url != _pocket_voice_source_url(
            source_repository, source_revision, source_path
        ):
            raise CatalogError(f"{bundle_id}/{name}: url is not pinned to its source")
        if resolver is not None and (not isinstance(resolver, str) or not resolver):
            raise CatalogError(f"{bundle_id}/{name}: resolver must be a non-empty string")
        if access["distributable"] and not isinstance(url, str):
            raise CatalogError(f"{bundle_id}/{name}: distributable state requires a url")
        if (
            not access["distributable"]
            and not isinstance(url, str)
            and not isinstance(resolver, str)
        ):
            raise CatalogError(f"{bundle_id}/{name}: state requires a resolver or url")
        names.append(name)
        records.append(dict(raw))
    return tuple(names), records


def _normalize_pocket_voice_tag(value: Any, field: str, bundle_id: str) -> str:
    if value is None or value == "":
        return ""
    if not isinstance(value, str):
        raise CatalogError(f"{bundle_id}: voice detail {field} must be a string")
    normalized = normalize_language_tag(value)
    if not normalized or any(
        not part or any(char.isspace() for char in part) for part in normalized.split("-")
    ):
        raise CatalogError(f"{bundle_id}: invalid voice detail {field}")
    if not normalized.split("-", 1)[0].isalpha():
        raise CatalogError(f"{bundle_id}: invalid voice detail {field}")
    return normalized


def _parse_pocket_voice_details(
    raw: Any, bundle_id: str, declared_names: set[str]
) -> list[dict[str, str]]:
    if raw is None:
        return []
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        raise CatalogError(f"{bundle_id}: voice_details must be a sequence")
    result: list[dict[str, str]] = []
    seen: set[str] = set()
    allowed = {"id", "language", "locale", "language_label", "gender"}
    for index, value in enumerate(raw):
        if not isinstance(value, Mapping):
            raise CatalogError(f"{bundle_id}: voice detail {index} must be an object")
        if set(value) - allowed:
            raise CatalogError(f"{bundle_id}: voice detail {index} has unexpected fields")
        voice_id = value.get("id")
        if not isinstance(voice_id, str):
            raise CatalogError(f"{bundle_id}: voice detail {index} has no id")
        _require_pocket_safe_id(voice_id, f"{bundle_id} voice detail id")
        if voice_id not in declared_names:
            raise CatalogError(f"{bundle_id}: voice detail references unknown voice {voice_id!r}")
        if voice_id in seen:
            raise CatalogError(f"{bundle_id}: duplicate voice detail {voice_id!r}")
        seen.add(voice_id)

        language_tag = _normalize_pocket_voice_tag(value.get("language"), "language", bundle_id)
        locale = _normalize_pocket_voice_tag(value.get("locale"), "locale", bundle_id)
        if language_tag and locale and language_base(language_tag) != language_base(locale):
            raise CatalogError(f"{bundle_id}/{voice_id}: language and locale do not agree")
        locale = locale or language_tag
        language = language_base(locale) if locale else ""

        label = value.get("language_label")
        label = "" if not isinstance(label, str) else label.strip()
        if not label or (len(label) == 2 and label.isalpha() and label.isupper()):
            label = locale or language or "unknown"
        gender = value.get("gender", "unknown")
        if not isinstance(gender, str) or gender not in VALID_GENDERS:
            raise CatalogError(f"{bundle_id}/{voice_id}: invalid gender")
        result.append(
            {
                "id": voice_id,
                "language": language,
                "locale": locale,
                "language_label": label,
                "gender": gender,
            }
        )
    return result


def _require_supertonic_safe_id(value: Any, label: str) -> str:
    if not isinstance(value, str) or _ST_SAFE_ID_RE.fullmatch(value) is None:
        raise CatalogError(f"{label} is unsafe: {value!r}")
    return value


def _supertonic_artifact_url(repository: str, revision: str, path: str) -> str:
    repo = "/".join(urllib.parse.quote(part, safe="") for part in repository.split("/"))
    encoded_path = urllib.parse.quote(path, safe="/")
    return f"https://huggingface.co/{repo}/resolve/{revision}/{encoded_path}?download=true"


def _parse_supertonic(data: dict[str, Any]) -> list[CatalogItem]:
    if not isinstance(data, Mapping):
        raise CatalogError("Supertonic catalog must be an object")
    if set(data) != {"schema", "kind", "source", "bundles"}:
        raise CatalogError("Supertonic catalog must contain exactly schema, kind, source, bundles")
    if type(data.get("schema")) is not int or data.get("schema") != 1:
        raise CatalogError("Supertonic catalog schema must be 1")
    if data.get("kind") != "supertonic-onnx-bundle-catalog":
        raise CatalogError(f"Unexpected Supertonic catalog kind: {data.get('kind')!r}")

    source = data.get("source")
    if not isinstance(source, Mapping):
        raise CatalogError("Supertonic catalog source metadata must be an object")
    if source.get("provider") != "huggingface":
        raise CatalogError("Supertonic source provider must be huggingface")
    repository = source.get("repository")
    if not isinstance(repository, str) or _ST_REPOSITORY_RE.fullmatch(repository) is None:
        raise CatalogError("Supertonic source repository must have owner/name form")
    revision = source.get("revision")
    if not isinstance(revision, str) or _ST_SHA_RE.fullmatch(revision) is None:
        raise CatalogError("Supertonic source revision must be a lowercase 40-character SHA")
    requested_revision = source.get("requested_revision")
    if not isinstance(requested_revision, str) or not requested_revision.strip():
        raise CatalogError("Supertonic source requested_revision is required")

    bundles = data.get("bundles")
    if not isinstance(bundles, Mapping):
        raise CatalogError("Supertonic catalog bundles must be a mapping")
    bundle_ids = set(bundles)
    if any(not isinstance(bundle_id, str) for bundle_id in bundle_ids):
        raise CatalogError("Supertonic bundle IDs must be strings")
    bundle_count = source.get("bundle_count")
    if bundle_count is not None and (
        isinstance(bundle_count, bool)
        or not isinstance(bundle_count, int)
        or bundle_count != len(bundles)
    ):
        raise CatalogError("Supertonic source bundle_count does not match bundles")

    result: list[CatalogItem] = []
    all_aliases: set[str] = set()
    for bundle_id, raw_entry in bundles.items():
        _require_supertonic_safe_id(bundle_id, "Supertonic bundle ID")
        if not isinstance(raw_entry, Mapping):
            raise CatalogError(f"{bundle_id}: bundle entry must be an object")
        entry = raw_entry
        if "id" in entry and entry["id"] != bundle_id:
            raise CatalogError(f"Supertonic bundle map key {bundle_id!r} does not match id")

        aliases = entry.get("aliases", [])
        if not isinstance(aliases, Sequence) or isinstance(aliases, (str, bytes)):
            raise CatalogError(f"{bundle_id}: aliases must be a sequence")
        parsed_aliases: list[str] = []
        for alias in aliases:
            _require_supertonic_safe_id(alias, f"{bundle_id} alias")
            if alias == bundle_id or alias in bundle_ids or alias in all_aliases:
                raise CatalogError(f"{bundle_id}: duplicate or conflicting alias {alias!r}")
            all_aliases.add(alias)
            parsed_aliases.append(alias)

        raw_artifacts = entry.get("artifacts")
        if not isinstance(raw_artifacts, Sequence) or isinstance(raw_artifacts, (str, bytes)):
            raise CatalogError(f"{bundle_id}: artifacts must be a sequence")
        artifacts: list[Artifact] = []
        seen_pairs: set[tuple[str, str | None]] = set()
        seen_roles: set[str] = set()
        style_components: set[str] = set()
        for raw in raw_artifacts:
            if not isinstance(raw, Mapping):
                raise CatalogError(f"{bundle_id}: artifact must be an object")
            role = raw.get("role")
            if not isinstance(role, str) or role not in ST_ARTIFACT_ROLES:
                raise CatalogError(f"{bundle_id}: unknown Supertonic artifact role {role!r}")
            component = raw.get("component")
            if role in {"model", "voice_style"}:
                component = _require_supertonic_safe_id(component, f"{bundle_id}/{role} component")
            elif component is not None:
                raise CatalogError(f"{bundle_id}/{role}: component must be omitted")
            if role == "model" and component not in ST_MODEL_COMPONENTS:
                raise CatalogError(f"{bundle_id}: unknown model component {component!r}")
            pair = (role, component)
            if pair in seen_pairs:
                raise CatalogError(f"{bundle_id}: duplicate (role, component) pair {pair!r}")
            seen_pairs.add(pair)
            if role in {"config", "unicode_indexer"}:
                if role in seen_roles:
                    raise CatalogError(f"{bundle_id}: duplicate {role!r} artifact")
                seen_roles.add(role)
            if role == "voice_style":
                assert component is not None
                if component in style_components:
                    raise CatalogError(
                        f"{bundle_id}: duplicate voice style component {component!r}"
                    )
                style_components.add(component)

            filename = raw.get("filename")
            if not isinstance(filename, str):
                raise CatalogError(f"{bundle_id}/{role}: invalid filename")
            try:
                validate_safe_component(filename, field_name="Supertonic artifact filename")
            except ValueError as exc:
                raise CatalogError(f"{bundle_id}/{role}: invalid filename") from exc
            path = raw.get("path")
            if not isinstance(path, str):
                raise CatalogError(f"{bundle_id}/{role}: unsafe artifact path")
            try:
                validate_relative_path(path, field_name="Supertonic artifact path")
            except ValueError as exc:
                raise CatalogError(f"{bundle_id}/{role}: unsafe artifact path") from exc
            url = raw.get("url")
            expected_url = _supertonic_artifact_url(repository, revision, path)
            if not isinstance(url, str) or url != expected_url:
                raise CatalogError(f"{bundle_id}/{role}: artifact URL is not pinned to source")
            size = raw.get("size")
            if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
                raise CatalogError(f"{bundle_id}/{role}: artifact size must be a positive integer")
            sha256 = raw.get("sha256")
            if not isinstance(sha256, str) or _ST_SHA256_RE.fullmatch(sha256) is None:
                raise CatalogError(f"{bundle_id}/{role}: artifact sha256 must be lowercase hex")
            format_name = raw.get("format")
            if not isinstance(format_name, str) or not format_name:
                raise CatalogError(f"{bundle_id}/{role}: artifact format is required")
            raw_metadata = raw.get("metadata", {})
            if not isinstance(raw_metadata, Mapping):
                raise CatalogError(f"{bundle_id}/{role}: artifact metadata must be an object")
            artifacts.append(
                Artifact(
                    role=role,
                    component=component,
                    filename=filename,
                    url=url,
                    size=size,
                    sha256=sha256,
                    format=format_name,
                    metadata={
                        **raw_metadata,
                        "path": path,
                        "source": {
                            "provider": "huggingface",
                            "repository": repository,
                            "revision": revision,
                            "path": path,
                            "gated": False,
                        },
                    },
                )
            )

        for role in ("config", "unicode_indexer"):
            if role not in seen_roles:
                raise CatalogError(f"{bundle_id}: missing required {role!r} artifact")
        for component_name in ST_MODEL_COMPONENTS:
            if ("model", component_name) not in seen_pairs:
                raise CatalogError(f"{bundle_id}: missing model component {component_name!r}")

        languages = entry.get("languages")
        if (
            not isinstance(languages, Sequence)
            or isinstance(languages, (str, bytes))
            or not languages
        ):
            raise CatalogError(f"{bundle_id}: languages must be a non-empty sequence")
        if any(not isinstance(language, str) or not language for language in languages):
            raise CatalogError(f"{bundle_id}: languages must contain non-empty strings")
        if len(set(languages)) != len(languages):
            raise CatalogError(f"{bundle_id}: languages must be unique")
        default_language = entry.get("default_language")
        if default_language not in languages:
            raise CatalogError(f"{bundle_id}: default_language must be included in languages")
        sample_rate = entry.get("sample_rate")
        if isinstance(sample_rate, bool) or not isinstance(sample_rate, int) or sample_rate <= 0:
            raise CatalogError(f"{bundle_id}: sample_rate must be a positive integer")
        runtime = entry.get("runtime")
        if not isinstance(runtime, Mapping):
            raise CatalogError(f"{bundle_id}: runtime metadata must be an object")
        if runtime.get("layout") != "supertonic-3-v1":
            raise CatalogError(f"{bundle_id}: runtime layout must be 'supertonic-3-v1'")

        voices = entry.get("voices")
        if not isinstance(voices, Sequence) or isinstance(voices, (str, bytes)) or not voices:
            raise CatalogError(f"{bundle_id}: voices must be a non-empty sequence")
        voice_names: list[str] = []
        for voice in voices:
            if not isinstance(voice, Mapping):
                raise CatalogError(f"{bundle_id}: voice entry must be an object")
            name = _require_supertonic_safe_id(voice.get("name"), f"{bundle_id} voice name")
            if name in voice_names:
                raise CatalogError(f"{bundle_id}: duplicate voice {name!r}")
            voice_names.append(name)
            component = _require_supertonic_safe_id(
                voice.get("artifact_component"), f"{bundle_id}/{name} voice style component"
            )
            if ("voice_style", component) not in seen_pairs:
                raise CatalogError(
                    f"{bundle_id}/{name}: voice references missing style artifact {component!r}"
                )
        default_voice = entry.get("default_voice")
        if default_voice not in voice_names:
            raise CatalogError(f"{bundle_id}: default_voice must name a declared voice")
        metadata = entry.get("metadata", {})
        if not isinstance(metadata, Mapping):
            raise CatalogError(f"{bundle_id}: metadata must be an object")
        result.append(
            CatalogItem(
                system="supertonic",
                id=bundle_id,
                kind="bundle",
                artifacts=tuple(artifacts),
                aliases=tuple(parsed_aliases),
                sample_rate=sample_rate,
                voices=tuple(voice_names),
                default_voice=default_voice,
                metadata={
                    **metadata,
                    "language_codes": tuple(languages),
                    "runtime": dict(runtime),
                    "voices": list(voices),
                    "source_revision": revision,
                    "source_repository": repository,
                    "requested_revision": requested_revision,
                },
            )
        )
    return result


def _parse_pocket(data: dict[str, Any]) -> list[CatalogItem]:

    if "schema" in data and data["schema"] != 1:
        raise CatalogError("Pocket catalog schema must be 1")
    if "kind" in data and data["kind"] != "pocket-onnx-bundle-catalog":
        raise CatalogError(f"Unexpected Pocket catalog kind: {data['kind']!r}")
    bundles = data.get("bundles")
    if not isinstance(bundles, (dict, list)):
        raise CatalogError("Pocket catalog is missing the 'bundles' list or mapping")
    if data.get("kind") == "pocket-onnx-bundle-catalog" and not isinstance(bundles, dict):
        raise CatalogError("Canonical Pocket catalog bundles must be a mapping")
    canonical = data.get("kind") == "pocket-onnx-bundle-catalog"
    source = data.get("source") or {}
    if not isinstance(source, Mapping):
        raise CatalogError("Pocket catalog source metadata must be an object")
    result: list[CatalogItem] = []
    entries = bundles.items() if isinstance(bundles, dict) else ((None, entry) for entry in bundles)
    for map_id, raw_entry in entries:
        if not isinstance(raw_entry, Mapping):
            raise CatalogError("Pocket bundle entry must be an object")
        entry = raw_entry
        bundle_id = entry.get("id") if map_id is None else map_id
        if not isinstance(bundle_id, str) or not bundle_id:
            raise CatalogError("Pocket bundle entry is missing 'id'")
        _require_pocket_safe_id(bundle_id, "Pocket bundle id")
        if map_id is not None and entry.get("id") != map_id:
            raise CatalogError(
                f"Pocket bundle map key {map_id!r} does not match entry id {entry.get('id')!r}"
            )
        aliases = entry.get("aliases") or []
        if not isinstance(aliases, Sequence) or isinstance(aliases, (str, bytes)):
            raise CatalogError(f"{bundle_id}: aliases must be a sequence")
        for alias in aliases:
            if not isinstance(alias, str):
                raise CatalogError(f"{bundle_id}: aliases must be strings")
            _require_pocket_safe_id(alias, f"{bundle_id} alias")
        raw_artifacts = entry.get("artifacts", ())
        if isinstance(raw_artifacts, Mapping):
            raw_artifacts = raw_artifacts.values()
        if not isinstance(raw_artifacts, Sequence) or isinstance(raw_artifacts, (str, bytes)):
            raise CatalogError(f"{bundle_id}: artifacts must be a sequence")
        artifacts: list[Artifact] = []
        for raw in raw_artifacts:
            if not isinstance(raw, Mapping):
                raise CatalogError(f"{bundle_id}: artifact must be an object")
            role = raw.get("role")
            if role not in POCKET_ARTIFACT_ROLES:
                raise CatalogError(f"Unknown Pocket artifact role: {role!r}")
            _validate_canonical_pocket_integrity(raw, bundle_id, str(role), canonical=canonical)
            raw_path = raw.get("path")
            filename = raw.get("filename") or (Path(str(raw_path)).name if raw_path else None)
            if not isinstance(filename, str) or not filename:
                raise CatalogError(f"{bundle_id}/{role}: artifact filename is required")
            _require_pocket_safe_id(filename, f"{bundle_id}/{role} filename")
            metadata = dict(raw.get("metadata") or {})
            if raw_path is not None:
                metadata.setdefault("path", raw_path)
            if source.get("provider") == "huggingface":
                if not isinstance(source.get("repository"), str) or not source["repository"]:
                    raise CatalogError(f"{bundle_id}/{role}: Hugging Face repository is required")
                if not isinstance(source.get("revision"), str) or not source["revision"]:
                    raise CatalogError(f"{bundle_id}/{role}: Hugging Face revision is required")
                if not isinstance(raw_path, str) or not raw_path:
                    raise CatalogError(f"{bundle_id}/{role}: Hugging Face path is required")
                try:
                    validate_relative_path(raw_path, field_name="Hugging Face repository path")
                except ValueError as exc:
                    raise CatalogError(str(exc)) from exc
                metadata["source"] = {
                    "provider": "huggingface",
                    "repository": source["repository"],
                    "revision": source["revision"],
                    "path": raw_path,
                    "gated": False,
                }
            artifacts.append(
                Artifact(
                    role=str(role),
                    filename=filename,
                    url=raw.get("url"),
                    size=raw.get("size"),
                    sha256=raw.get("sha256"),
                    quality=raw.get("quality"),
                    component=raw.get("component"),
                    format=raw.get("format"),
                    metadata=metadata,
                )
            )
        voice_names, voice_states = _parse_pocket_voice_states(entry, bundle_id)

        raw_predefined_names = entry.get("predefined_voice_names") or []
        if not isinstance(raw_predefined_names, Sequence) or isinstance(
            raw_predefined_names, (str, bytes)
        ):
            raise CatalogError(f"{bundle_id}: predefined_voice_names must be a sequence")
        predefined_voice_names: list[str] = []
        for name in raw_predefined_names:
            if not isinstance(name, str):
                raise CatalogError(f"{bundle_id}: predefined voice names must be strings")
            _require_pocket_safe_id(name, f"{bundle_id} predefined voice name")
            if name in predefined_voice_names:
                raise CatalogError(f"{bundle_id}: duplicate predefined voice {name!r}")
            predefined_voice_names.append(name)
        declared_voice_names = set(voice_names) | set(predefined_voice_names)
        voice_details = _parse_pocket_voice_details(
            entry.get("voice_details"), bundle_id, declared_voice_names
        )
        metadata = {
            **(entry.get("metadata") or {}),
            "language": entry.get("language"),
            "layers": entry.get("layers"),
            "bundle_schema": entry.get("bundle_schema"),
            "profiles": entry.get("profiles") or {},
            "source_revision": source.get("revision"),
            "source_repository": source.get("repository"),
            "requested_revision": source.get("requested_revision"),
            "max_token_per_chunk": entry.get("max_token_per_chunk"),
            "model_recommended_frames_after_eos": entry.get("model_recommended_frames_after_eos"),
            "remove_semicolons": entry.get("remove_semicolons"),
            "pad_with_spaces_for_short_inputs": entry.get("pad_with_spaces_for_short_inputs"),
            "voice_states": voice_states,
            "predefined_voice_names": predefined_voice_names,
            "canonical_catalog": canonical,
        }
        if "voice_details" in entry:
            metadata["voice_details"] = voice_details
        result.append(
            CatalogItem(
                system="pocket",
                id=bundle_id,
                kind="bundle",
                artifacts=tuple(artifacts),
                aliases=tuple(aliases),
                voices=voice_names,
                sample_rate=entry.get("sample_rate"),
                metadata=metadata,
            )
        )
    return result


def _require_pocket_safe_id(value: str, label: str) -> None:
    if (
        value in {".", ".."}
        or "/" in value
        or "\\" in value
        or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", value)
    ):
        raise CatalogError(f"{label} is unsafe: {value!r}")


def _select_pocket_profile(item: CatalogItem, quality: str | None) -> CatalogItem:
    """Select exactly one deterministic artifact for each Pocket profile role."""
    profiles = item.metadata.get("profiles") or {}
    if not isinstance(profiles, Mapping):
        raise CatalogError(f"Pocket bundle {item.id!r} has invalid profiles metadata")
    selected_quality = quality or "int8"
    profile = profiles.get(selected_quality)
    if not isinstance(profile, Mapping):
        available = ", ".join(sorted(str(name) for name in profiles)) or "none"
        raise CatalogError(
            f"Unknown Pocket profile {selected_quality!r} for {item.id!r}; available profiles: {available}"
        )
    missing_roles = [role for role in POCKET_QUALIFIED_ROLES if role not in profile]
    if missing_roles:
        raise CatalogError(
            f"Profile {selected_quality!r} does not specify required roles: {', '.join(sorted(missing_roles))}"
        )
    static: dict[str, list[Artifact]] = {
        role: [] for role in ("bundle_metadata", "tokenizer", "bos_conditioning")
    }
    for artifact in item.artifacts:
        if artifact.role in static:
            static[artifact.role].append(artifact)
    for role, matches in static.items():
        if len(matches) != 1:
            raise CatalogError(
                f"Pocket bundle {item.id!r} must contain exactly one static artifact for {role!r}"
            )
    selected: list[Artifact] = [
        static[role][0] for role in ("bundle_metadata", "tokenizer", "bos_conditioning")
    ]
    for role in (
        "flow_lm_main",
        "flow_lm_flow",
        "mimi_decoder",
        "mimi_encoder",
        "text_conditioner",
    ):
        target_quality = profile[role]
        matches = [
            artifact
            for artifact in item.artifacts
            if artifact.role == role and artifact.quality == target_quality
        ]
        if len(matches) != 1:
            raise CatalogError(
                f"Profile {selected_quality!r} requires exactly one {role!r} artifact with "
                f"quality={target_quality!r}; found {len(matches)}"
            )
        selected.append(matches[0])
    return CatalogItem(
        system=item.system,
        id=item.id,
        kind=item.kind,
        artifacts=tuple(selected),
        aliases=item.aliases,
        sample_rate=item.sample_rate,
        voices=item.voices,
        default_voice=item.default_voice,
        metadata={**item.metadata, "selected_quality": selected_quality},
    )


def _item_supports_quality(item: CatalogItem, quality: str) -> bool:
    if item.system == "pocket":
        profiles = item.metadata.get("profiles") or {}
        return isinstance(profiles, Mapping) and quality in profiles
    return bool(
        item.metadata.get("quality") == quality
        or any(a.role == "model" and a.quality == quality for a in item.artifacts)
    )


def filter_items(
    items: Iterable[CatalogItem],
    *,
    language: str | None = None,
    quality: str | None = None,
) -> list[CatalogItem]:
    from .inventory import matches_language

    result = list(items)
    if language:
        result = [item for item in result if matches_language(item.metadata, language)]
    if quality:
        result = [item for item in result if _item_supports_quality(item, quality)]
    return result
