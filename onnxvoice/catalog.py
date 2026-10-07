from __future__ import annotations

import json
import math
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
    "kitten": "https://raw.githubusercontent.com/buchwandler/kitten-onnx-bundles/main/catalog/models.json",
    "inflect": "https://raw.githubusercontent.com/buchwandler/inflect-onnx-bundles/main/catalog/models.json",
}
_KOKORO_CLONING_MODEL_COMPONENTS = frozenset(
    {
        "reference_wavlm",
        "reference_encoders",
        "reference_mapper",
        "prosody",
        "curves",
        "decoder",
    }
)


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

    @staticmethod
    def _cache_generation(path: Path) -> tuple[int, int, int, int, int] | None:
        """Return a stat fingerprint that changes when atomic cache publication replaces a file."""
        try:
            stat = path.stat()
        except FileNotFoundError:
            return None
        return (
            stat.st_dev,
            stat.st_ino,
            stat.st_size,
            stat.st_mtime_ns,
            stat.st_ctime_ns,
        )

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
        mode is disabled. ``refresh=True`` bypasses normal cache freshness, but callers
        waiting behind an overlapping refresh reuse a cache generation published after
        they began waiting. Calls that start after publication perform their own refresh.
        """
        system = system.lower()
        if not self.sources or system not in self.sources:
            raise CatalogError(f"No catalog source configured for system {system!r}")
        self._emit(progress, AssetProgress("catalog_started", ref=system))
        cache_path = self._cache_path(system)

        def cached(
            generation: tuple[int, int, int, int, int] | None,
            *,
            accept_refresh: bool = False,
        ) -> dict[str, Any] | None:
            fresh = (
                generation is not None
                and time.time_ns() - generation[3] < self.ttl_seconds * 1_000_000_000
            )
            if generation is not None and (
                accept_refresh or ((fresh or self.offline) and not refresh)
            ):
                data = json.loads(cache_path.read_text(encoding="utf-8"))
                self._emit(progress, AssetProgress("catalog_cached", ref=system))
                return data
            return None

        # Snapshot before waiting. For refreshes, this distinguishes our own forced
        # fetch from a newer cache another overlapping caller has already published.
        cache_generation = self._cache_generation(cache_path)
        data = cached(cache_generation)
        if data is not None:
            return data

        with FileLock(self._catalog_lock_path(system)):
            current_generation = self._cache_generation(cache_path)
            if refresh and not self.offline and current_generation != cache_generation:
                data = cached(current_generation, accept_refresh=True)
                if data is not None:
                    return data
            data = cached(current_generation)
            if data is not None:
                return data
            payload = self._read_source(self.sources[system])
            try:
                data = json.loads(payload)
            except json.JSONDecodeError as exc:
                raise CatalogError(f"Invalid JSON in {system} catalog") from exc
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
        if system == "kitten":
            return _parse_kitten(raw)
        if system == "inflect":
            return _parse_inflect(raw)
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
        model_artifacts = [
            artifact
            for artifact in item.artifacts
            if artifact.role == "model"
            and not (item.system == "kokoro" and artifact.component == _KOKORO_INNO_GRAPH_COMPONENT)
        ]
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
            if (
                artifact.role != "model"
                or (item.system == "kokoro" and artifact.component == _KOKORO_INNO_GRAPH_COMPONENT)
                or artifact.quality in {None, selected_quality}
            )
        )
        if not any(
            artifact.role == "model"
            and not (item.system == "kokoro" and artifact.component == _KOKORO_INNO_GRAPH_COMPONENT)
            for artifact in artifacts
        ):
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


_KOKORO_INNO_ID = "inno-v0.2"
_KOKORO_INNO_GRAPH_COMPONENT = "inno_voicepack"
_KOKORO_INNO_METADATA_COMPONENT = "inno_tuner"
_KOKORO_SPLIT_MODEL_COMPONENTS = frozenset({"prosody", "curves", "decoder"})
_KOKORO_INNO_LAYOUTS = frozenset({"single", "single-onnx-v1", "split", "multi", "split-onnx-v1"})


def _normalize_kokoro_voice_enrollers(
    runtime: Mapping[str, Any], artifacts: Sequence[Artifact]
) -> tuple[Mapping[str, Any], ...] | None:
    """Keep only capabilities whose assets are included in the selected distribution."""
    capabilities = runtime.get("voice_enrollers")
    if capabilities is None:
        return None
    if not isinstance(capabilities, (list, tuple)):
        raise CatalogError("Kokoro runtime voice_enrollers must be a list")
    available: list[Mapping[str, Any]] = []
    seen_ids: set[str] = set()
    for capability in capabilities:
        if not isinstance(capability, Mapping):
            raise CatalogError("Kokoro voice enroller declarations must be objects")
        enroller_id = capability.get("id")
        if not isinstance(enroller_id, str) or not enroller_id:
            raise CatalogError("Kokoro voice enroller declaration is missing its id")
        if enroller_id in seen_ids:
            raise CatalogError(f"Kokoro voice enroller id {enroller_id!r} is duplicated")
        seen_ids.add(enroller_id)
        if not isinstance(capability.get("kind"), str) or not capability["kind"]:
            raise CatalogError(f"Kokoro voice enroller {enroller_id!r} is missing its kind")
        if enroller_id != _KOKORO_INNO_ID:
            available.append(capability)
            continue

        expected = {
            "kind": "kokoro-voicepack-tuner",
            "input": "reference-audio",
            "model_component": _KOKORO_INNO_GRAPH_COMPONENT,
            "metadata_component": _KOKORO_INNO_METADATA_COMPONENT,
        }
        for key, value in expected.items():
            if capability.get(key) != value:
                raise CatalogError(f"Kokoro Inno capability has invalid {key!r}")
        if capability.get("transcript_required") is not False:
            raise CatalogError("Kokoro Inno capability must declare transcript_required=false")
        for key, expected_seconds in (
            ("min_seconds", 3.0),
            ("recommended_seconds", 5.0),
            ("max_seconds", 30.0),
        ):
            actual = capability.get(key)
            if isinstance(actual, bool) or not isinstance(actual, (int, float)):
                raise CatalogError(f"Kokoro Inno capability has invalid {key!r}")
            try:
                actual_value = float(actual)
            except (OverflowError, ValueError):
                raise CatalogError(f"Kokoro Inno capability has invalid {key!r}") from None
            if not math.isfinite(actual_value) or actual_value != expected_seconds:
                raise CatalogError(f"Kokoro Inno capability has invalid {key!r}")
        output = capability.get("output")
        shape = output.get("shape") if isinstance(output, Mapping) else None
        if (
            not isinstance(output, Mapping)
            or output.get("format") != "kokoro-voicepack-v1"
            or not isinstance(shape, (list, tuple))
            or tuple(shape) != (510, 1, 256)
            or output.get("dtype") != "float32"
        ):
            raise CatalogError("Kokoro Inno capability has an invalid output contract")

        graph_artifacts = [
            artifact
            for artifact in artifacts
            if artifact.role == "model" and artifact.component == _KOKORO_INNO_GRAPH_COMPONENT
        ]
        metadata_artifacts = [
            artifact
            for artifact in artifacts
            if artifact.role == "metadata" and artifact.component == _KOKORO_INNO_METADATA_COMPONENT
        ]
        if not graph_artifacts and not metadata_artifacts:
            continue
        if len(graph_artifacts) != 1 or len(metadata_artifacts) != 1:
            raise CatalogError(
                "Kokoro Inno distribution requires one model:inno_voicepack and "
                "one metadata:inno_tuner artifact"
            )

        layout = runtime.get("layout", "single")
        if not isinstance(layout, str) or layout not in _KOKORO_INNO_LAYOUTS:
            raise CatalogError(f"Kokoro Inno capability does not support layout {layout!r}")
        base_models = [
            artifact
            for artifact in artifacts
            if artifact.role == "model" and artifact.component != _KOKORO_INNO_GRAPH_COMPONENT
        ]
        if not base_models:
            raise CatalogError("Kokoro Inno distribution is missing its base model artifacts")
        for quality in {artifact.quality for artifact in base_models}:
            selected_models = [
                artifact for artifact in base_models if artifact.quality in {None, quality}
            ]
            components = {artifact.component for artifact in selected_models}
            if layout in {"single", "single-onnx-v1"}:
                if None not in components:
                    raise CatalogError(
                        f"Kokoro Inno quality {quality!r} is missing its single base model"
                    )
            elif not components >= _KOKORO_SPLIT_MODEL_COMPONENTS:
                missing = _KOKORO_SPLIT_MODEL_COMPONENTS - components
                raise CatalogError(
                    f"Kokoro Inno quality {quality!r} is missing base model components: "
                    + ", ".join(sorted(missing))
                )

        graph, metadata = graph_artifacts[0], metadata_artifacts[0]
        if graph.quality is not None or metadata.quality is not None:
            raise CatalogError("Kokoro Inno artifacts must be shared across base model qualities")
        if graph.format not in {None, "onnx"}:
            raise CatalogError("Kokoro Inno model:inno_voicepack must use ONNX format")
        if metadata.format not in {None, "npz", "numpy-npz", "json"}:
            raise CatalogError("Kokoro Inno metadata:inno_tuner must use NPZ or JSON format")
        available.append(capability)
    return tuple(available)


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
    available_enrollers = _normalize_kokoro_voice_enrollers(runtime, artifacts)
    if available_enrollers is not None:
        runtime["voice_enrollers"] = list(available_enrollers)
    if runtime.get("layout") == "cloning-onnx-v1":
        model_artifacts = [artifact for artifact in artifacts if artifact.role == "model"]
        model_components = {artifact.component for artifact in model_artifacts}
        missing = _KOKORO_CLONING_MODEL_COMPONENTS - model_components
        if missing:
            raise CatalogError(
                "Kokoro cloning distribution is missing model components: "
                + ", ".join(sorted(missing))
            )
        for quality in {artifact.quality for artifact in model_artifacts}:
            quality_components = {
                artifact.component
                for artifact in model_artifacts
                if artifact.quality in {None, quality}
            }
            missing_for_quality = _KOKORO_CLONING_MODEL_COMPONENTS - quality_components
            if missing_for_quality:
                raise CatalogError(
                    f"Kokoro cloning quality {quality!r} is missing model components: "
                    + ", ".join(sorted(missing_for_quality))
                )
        if not any(
            artifact.role == "metadata" and artifact.component == "source_params"
            for artifact in artifacts
        ):
            raise CatalogError("Kokoro cloning distribution is missing metadata source_params")
        if not any(artifact.role in {"config", "bundle", "manifest"} for artifact in artifacts):
            raise CatalogError("Kokoro cloning distribution is missing its config/bundle")
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
        if raw.get("format") != "safetensors":
            raise CatalogError(f"{bundle_id}/{name}: format must be safetensors")
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


_KITTEN_SAFE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
_KITTEN_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_KITTEN_HF_REPOSITORY_RE = re.compile(r"^[^/\\\\\s]+/[^/\\\\\s]+$")
_KITTEN_HF_REVISION_RE = re.compile(r"^[0-9a-f]{40}$")
_KITTEN_MODEL_FIELDS = frozenset(
    {
        "id",
        "aliases",
        "name",
        "version",
        "language",
        "sample_rate",
        "quality",
        "runtime",
        "upstream",
        "artifacts",
        "voice_aliases",
        "speed_priors",
        "metadata",
    }
)


def _require_kitten_safe_id(value: Any, label: str) -> str:
    if not isinstance(value, str) or _KITTEN_SAFE_ID_RE.fullmatch(value) is None:
        raise CatalogError(f"{label} is unsafe: {value!r}")
    return value


def _require_kitten_text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise CatalogError(f"{label} must be a non-empty string")
    return value


def _require_kitten_fields(
    value: Mapping[str, Any], expected: set[str] | frozenset[str], label: str
) -> None:
    if set(value) != set(expected):
        missing = sorted(set(expected) - set(value))
        unexpected = sorted(set(value) - set(expected))
        details = []
        if missing:
            details.append(f"missing: {', '.join(missing)}")
        if unexpected:
            details.append(f"unexpected: {', '.join(unexpected)}")
        raise CatalogError(f"{label} fields mismatch ({'; '.join(details)})")


def _kitten_artifact_url(repository: str, revision: str, filename: str) -> str:
    repo = "/".join(urllib.parse.quote(part, safe="") for part in repository.split("/"))
    name = urllib.parse.quote(filename, safe="")
    return f"https://huggingface.co/{repo}/resolve/{revision}/{name}"


def _parse_kitten(data: Mapping[str, Any]) -> list[CatalogItem]:
    if not isinstance(data, Mapping):
        raise CatalogError("Kitten catalog must be an object")
    if set(data) != {"schema", "kind", "models"}:
        raise CatalogError("Kitten catalog must contain exactly schema, kind, and models")
    if type(data.get("schema")) is not int or data["schema"] != 1:
        raise CatalogError("Kitten catalog schema must be 1")
    if data.get("kind") != "kitten-onnx-model-catalog":
        raise CatalogError(f"Unexpected Kitten catalog kind: {data.get('kind')!r}")

    models = data.get("models")
    if not isinstance(models, Mapping) or not models:
        raise CatalogError("Kitten catalog models must be a non-empty mapping")
    model_ids = set(models)
    for model_id in model_ids:
        _require_kitten_safe_id(model_id, "Kitten model ID")

    result: list[CatalogItem] = []
    all_aliases: set[str] = set()
    for model_id, raw_model in models.items():
        if not isinstance(raw_model, Mapping):
            raise CatalogError(f"{model_id}: model entry must be an object")
        _require_kitten_fields(raw_model, _KITTEN_MODEL_FIELDS, f"{model_id} model")
        if raw_model.get("id") != model_id:
            raise CatalogError(f"Kitten model map key {model_id!r} does not match id")

        raw_aliases = raw_model["aliases"]
        if not isinstance(raw_aliases, Sequence) or isinstance(raw_aliases, (str, bytes)):
            raise CatalogError(f"{model_id}: aliases must be a sequence")
        aliases: list[str] = []
        for raw_alias in raw_aliases:
            alias = _require_kitten_safe_id(raw_alias, f"{model_id} alias")
            if alias == model_id or alias in model_ids or alias in all_aliases:
                raise CatalogError(f"{model_id}: duplicate or conflicting alias {alias!r}")
            all_aliases.add(alias)
            aliases.append(alias)

        name = _require_kitten_text(raw_model["name"], f"{model_id} name")
        version = _require_kitten_text(raw_model["version"], f"{model_id} version")
        language = _require_kitten_text(raw_model["language"], f"{model_id} language")
        sample_rate = raw_model["sample_rate"]
        if isinstance(sample_rate, bool) or not isinstance(sample_rate, int) or sample_rate <= 0:
            raise CatalogError(f"{model_id}: sample_rate must be a positive integer")

        quality = raw_model["quality"]
        if quality is not None and (
            not isinstance(quality, str) or quality not in {"int8", "fp32"}
        ):
            raise CatalogError(f"{model_id}: quality must be 'int8', 'fp32', or null")

        runtime = raw_model["runtime"]
        if not isinstance(runtime, Mapping):
            raise CatalogError(f"{model_id}: runtime must be an object")
        _require_kitten_fields(runtime, {"profile"}, f"{model_id} runtime")
        if not isinstance(runtime["profile"], str) or runtime["profile"] not in {"ONNX1", "ONNX2"}:
            raise CatalogError(f"{model_id}: runtime profile must be ONNX1 or ONNX2")

        upstream = raw_model["upstream"]
        if not isinstance(upstream, Mapping):
            raise CatalogError(f"{model_id}: upstream must be an object")
        _require_kitten_fields(
            upstream, {"provider", "repository", "revision", "license"}, f"{model_id} upstream"
        )
        if upstream["provider"] != "huggingface":
            raise CatalogError(f"{model_id}: upstream provider must be huggingface")
        repository = upstream["repository"]
        if (
            not isinstance(repository, str)
            or _KITTEN_HF_REPOSITORY_RE.fullmatch(repository) is None
        ):
            raise CatalogError(f"{model_id}: upstream repository must have owner/name form")
        revision = upstream["revision"]
        if not isinstance(revision, str) or _KITTEN_HF_REVISION_RE.fullmatch(revision) is None:
            raise CatalogError(
                f"{model_id}: upstream revision must be a lowercase 40-character SHA"
            )
        license_name = _require_kitten_text(upstream["license"], f"{model_id} upstream license")

        raw_artifacts = raw_model["artifacts"]
        if not isinstance(raw_artifacts, Sequence) or isinstance(raw_artifacts, (str, bytes)):
            raise CatalogError(f"{model_id}: artifacts must be a sequence")
        artifacts: list[Artifact] = []
        seen_roles: set[str] = set()
        for raw_artifact in raw_artifacts:
            if not isinstance(raw_artifact, Mapping):
                raise CatalogError(f"{model_id}: artifact must be an object")
            _require_kitten_fields(
                raw_artifact,
                {"role", "format", "filename", "url", "size", "sha256"},
                f"{model_id} artifact",
            )
            role = raw_artifact["role"]
            expected_formats = {"model": "onnx", "voices": "npz"}
            if not isinstance(role, str) or role not in expected_formats:
                raise CatalogError(f"{model_id}: unknown Kitten artifact role {role!r}")
            if role in seen_roles:
                raise CatalogError(f"{model_id}: duplicate {role!r} artifact")
            seen_roles.add(role)
            format_name = raw_artifact["format"]
            if format_name != expected_formats[role]:
                raise CatalogError(f"{model_id}/{role}: format must be {expected_formats[role]!r}")
            filename = _require_kitten_safe_id(
                raw_artifact["filename"], f"{model_id}/{role} filename"
            )
            url = raw_artifact["url"]
            if not isinstance(url, str) or url != _kitten_artifact_url(
                repository, revision, filename
            ):
                raise CatalogError(
                    f"{model_id}/{role}: URL is not pinned to upstream repository/revision"
                )
            size = raw_artifact["size"]
            if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
                raise CatalogError(f"{model_id}/{role}: size must be a positive integer")
            sha256 = raw_artifact["sha256"]
            if not isinstance(sha256, str) or _KITTEN_SHA256_RE.fullmatch(sha256) is None:
                raise CatalogError(f"{model_id}/{role}: sha256 must be lowercase 64-character hex")
            artifacts.append(
                Artifact(
                    role=role,
                    filename=filename,
                    url=url,
                    size=size,
                    sha256=sha256,
                    quality=quality if role == "model" else None,
                    format=format_name,
                    metadata={
                        "source": {
                            "provider": "huggingface",
                            "repository": repository,
                            "revision": revision,
                            "path": filename,
                            "filename": filename,
                        }
                    },
                )
            )
        if seen_roles != {"model", "voices"}:
            raise CatalogError(f"{model_id}: artifacts must contain exactly model and voices roles")

        raw_voice_aliases = raw_model["voice_aliases"]
        if not isinstance(raw_voice_aliases, Mapping) or not raw_voice_aliases:
            raise CatalogError(f"{model_id}: voice_aliases must be a non-empty mapping")
        voice_aliases: dict[str, str] = {}
        for public_name, internal_id in raw_voice_aliases.items():
            public_name = _require_kitten_safe_id(public_name, f"{model_id} public voice alias")
            internal_id = _require_kitten_safe_id(
                internal_id, f"{model_id}/{public_name} internal voice ID"
            )
            voice_aliases[public_name] = internal_id

        raw_speed_priors = raw_model["speed_priors"]
        if not isinstance(raw_speed_priors, Mapping):
            raise CatalogError(f"{model_id}: speed_priors must be an object")
        speed_priors: dict[str, float] = {}
        internal_ids = set(voice_aliases.values())
        for internal_id, speed in raw_speed_priors.items():
            if internal_id not in internal_ids:
                raise CatalogError(
                    f"{model_id}: speed prior references unknown voice {internal_id!r}"
                )
            if isinstance(speed, bool) or not isinstance(speed, (int, float)):
                raise CatalogError(
                    f"{model_id}/{internal_id}: speed prior must be a positive finite number"
                )
            try:
                speed_value = float(speed)
            except OverflowError as exc:
                raise CatalogError(
                    f"{model_id}/{internal_id}: speed prior must be a positive finite number"
                ) from exc
            if not math.isfinite(speed_value) or speed_value <= 0:
                raise CatalogError(
                    f"{model_id}/{internal_id}: speed prior must be a positive finite number"
                )
            speed_priors[internal_id] = speed_value

        raw_metadata = raw_model["metadata"]
        if not isinstance(raw_metadata, Mapping):
            raise CatalogError(f"{model_id}: metadata must be an object")
        metadata = {
            **raw_metadata,
            "name": name,
            "version": version,
            "language": language,
            "quality": quality,
            "runtime": dict(runtime),
            "upstream": dict(upstream),
            "source_repository": repository,
            "source_revision": revision,
            "license": license_name,
            "voice_aliases": voice_aliases,
            "speed_priors": speed_priors,
        }
        result.append(
            CatalogItem(
                system="kitten",
                id=model_id,
                kind="model",
                artifacts=tuple(artifacts),
                aliases=tuple(aliases),
                sample_rate=sample_rate,
                voices=tuple(voice_aliases),
                default_voice=None,
                metadata=metadata,
            )
        )
    return result


_INFLECT_MODEL_FIELDS = frozenset(
    {
        "id",
        "aliases",
        "name",
        "version",
        "language",
        "sample_rate",
        "voice_mode",
        "default_voice",
        "voices",
        "controls",
        "runtime",
        "upstream",
        "artifacts",
        "metadata",
    }
)
_INFLECT_SAFE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
_INFLECT_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_INFLECT_HF_REPOSITORY_RE = re.compile(r"^[^/\\\\\s]+/[^/\\\\\s]+$")
_INFLECT_HF_REVISION_RE = re.compile(r"^[0-9a-f]{40}$")
_INFLECT_LANGUAGE_RE = re.compile(r"^[A-Za-z]{2,8}(?:-[A-Za-z0-9]{1,8})*$")
_INFLECT_ARTIFACT_FILENAMES = {"duration": "duration.onnx", "decode": "decode.onnx"}


def _require_inflect_fields(
    value: Mapping[str, Any], expected: set[str] | frozenset[str], label: str
) -> None:
    if set(value) == set(expected):
        return
    missing = sorted(set(expected) - set(value))
    unexpected = sorted(set(value) - set(expected))
    details = []
    if missing:
        details.append(f"missing: {', '.join(missing)}")
    if unexpected:
        details.append(f"unexpected: {', '.join(unexpected)}")
    raise CatalogError(f"{label} fields mismatch ({'; '.join(details)})")


def _require_inflect_safe_id(value: Any, label: str) -> str:
    if not isinstance(value, str) or _INFLECT_SAFE_ID_RE.fullmatch(value) is None:
        raise CatalogError(f"{label} is unsafe: {value!r}")
    return value


def _require_inflect_text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise CatalogError(f"{label} must be a non-empty string")
    return value


def _require_inflect_language(value: Any, label: str) -> str:
    if not isinstance(value, str) or _INFLECT_LANGUAGE_RE.fullmatch(value) is None:
        raise CatalogError(f"{label} must be a valid language tag")
    return normalize_language_tag(value)


def _inflect_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CatalogError(f"{label} must be a finite real number")
    try:
        number = float(value)
    except (OverflowError, ValueError):
        raise CatalogError(f"{label} must be a finite real number") from None
    if not math.isfinite(number):
        raise CatalogError(f"{label} must be a finite real number")
    return number


def _parse_inflect(data: Mapping[str, Any]) -> list[CatalogItem]:
    """Parse and validate the pinned schema-1 Inflect v2 model catalog."""
    if not isinstance(data, Mapping):
        raise CatalogError("Inflect catalog must be an object")
    if set(data) != {"schema", "kind", "models"}:
        raise CatalogError("Inflect catalog must contain exactly schema, kind, and models")
    if type(data.get("schema")) is not int or data["schema"] != 1:
        raise CatalogError("Inflect catalog schema must be 1")
    if data.get("kind") != "inflect-onnx-model-catalog":
        raise CatalogError(f"Unexpected Inflect catalog kind: {data.get('kind')!r}")

    models = data.get("models")
    if not isinstance(models, Mapping) or not models:
        raise CatalogError("Inflect catalog models must be a non-empty mapping")
    model_ids: dict[str, str] = {}
    for model_id in models:
        safe_id = _require_inflect_safe_id(model_id, "Inflect model ID")
        folded_id = safe_id.casefold()
        if folded_id in model_ids:
            raise CatalogError(f"Inflect model IDs collide after normalization: {safe_id!r}")
        model_ids[folded_id] = safe_id

    result: list[CatalogItem] = []
    normalized_names = dict(model_ids)
    for model_id, raw_model in models.items():
        if not isinstance(raw_model, Mapping):
            raise CatalogError(f"{model_id}: model entry must be an object")
        _require_inflect_fields(raw_model, _INFLECT_MODEL_FIELDS, f"{model_id} model")
        if raw_model.get("id") != model_id:
            raise CatalogError(f"Inflect model map key {model_id!r} does not match id")

        raw_aliases = raw_model["aliases"]
        if not isinstance(raw_aliases, Sequence) or isinstance(raw_aliases, (str, bytes)):
            raise CatalogError(f"{model_id}: aliases must be a sequence")
        aliases: list[str] = []
        for raw_alias in raw_aliases:
            alias = _require_inflect_safe_id(raw_alias, f"{model_id} alias")
            folded_alias = alias.casefold()
            if folded_alias in normalized_names:
                raise CatalogError(f"{model_id}: duplicate or conflicting alias {alias!r}")
            normalized_names[folded_alias] = alias
            aliases.append(alias)

        name = _require_inflect_text(raw_model["name"], f"{model_id} name")
        version = _require_inflect_text(raw_model["version"], f"{model_id} version")
        language = _require_inflect_language(raw_model["language"], f"{model_id} language")
        sample_rate = raw_model["sample_rate"]
        if type(sample_rate) is not int or sample_rate != 24000:
            raise CatalogError(f"{model_id}: sample_rate must be 24000 for Inflect v2")

        if raw_model["voice_mode"] != "fixed":
            raise CatalogError(f"{model_id}: voice_mode must be 'fixed'")
        default_voice = _require_inflect_safe_id(raw_model["default_voice"], f"{model_id} default_voice")
        if default_voice != "default":
            raise CatalogError(f"{model_id}: fixed voice default_voice must be 'default'")
        raw_voices = raw_model["voices"]
        if not isinstance(raw_voices, Mapping) or set(raw_voices) != {default_voice}:
            raise CatalogError(f"{model_id}: fixed voice catalog must contain exactly 'default'")
        raw_voice = raw_voices[default_voice]
        if not isinstance(raw_voice, Mapping):
            raise CatalogError(f"{model_id}: fixed voice entry must be an object")
        _require_inflect_fields(
            raw_voice, {"id", "name", "language", "gender", "synthetic"}, f"{model_id} voice"
        )
        voice_id = _require_inflect_safe_id(raw_voice["id"], f"{model_id} voice id")
        if voice_id != default_voice:
            raise CatalogError(f"{model_id}: fixed voice id must match default_voice")
        voice_name = _require_inflect_text(raw_voice["name"], f"{model_id} voice name")
        voice_language = _require_inflect_language(
            raw_voice["language"], f"{model_id} voice language"
        )
        if voice_language != language:
            raise CatalogError(f"{model_id}: fixed voice language must match model language")
        gender = raw_voice["gender"]
        if not isinstance(gender, str) or gender not in VALID_GENDERS:
            raise CatalogError(f"{model_id}: voice gender is not a supported descriptive value")
        synthetic = raw_voice["synthetic"]
        if not isinstance(synthetic, bool):
            raise CatalogError(f"{model_id}: voice synthetic flag must be boolean")
        voice_details = {
            default_voice: {
                "id": voice_id,
                "name": voice_name,
                "language": language_base(voice_language),
                "locale": voice_language,
                "gender": gender,
                "synthetic": synthetic,
            }
        }

        controls = raw_model["controls"]
        if not isinstance(controls, Mapping):
            raise CatalogError(f"{model_id}: controls must be an object")
        _require_inflect_fields(controls, {"speed", "variation", "seed_default"}, f"{model_id} controls")
        normalized_controls: dict[str, Any] = {}
        for control_name, lower_limit, upper_limit in (
            ("speed", 0.5, 2.0),
            ("variation", 0.0, 1.0),
        ):
            raw_control = controls[control_name]
            if not isinstance(raw_control, Mapping):
                raise CatalogError(f"{model_id}: {control_name} control must be an object")
            _require_inflect_fields(
                raw_control,
                {"default", "minimum", "maximum"},
                f"{model_id} {control_name} control",
            )
            control_values = {
                field: _inflect_number(raw_control[field], f"{model_id} {control_name} control {field}")
                for field in ("default", "minimum", "maximum")
            }
            minimum = control_values["minimum"]
            maximum = control_values["maximum"]
            if (
                minimum > control_values["default"]
                or control_values["default"] > maximum
                or minimum < lower_limit
                or maximum > upper_limit
            ):
                raise CatalogError(f"{model_id}: {control_name} control has incoherent bounds")
            normalized_controls[control_name] = control_values
        seed_default = controls["seed_default"]
        if isinstance(seed_default, bool) or not isinstance(seed_default, int):
            raise CatalogError(f"{model_id}: seed_default must be an integer, not bool")
        normalized_controls["seed_default"] = seed_default

        runtime = raw_model["runtime"]
        if not isinstance(runtime, Mapping):
            raise CatalogError(f"{model_id}: runtime must be an object")
        _require_inflect_fields(
            runtime, {"profile", "precision", "onnx_opset"}, f"{model_id} runtime"
        )
        if runtime["profile"] != "inflect-v2-split-v1":
            raise CatalogError(f"{model_id}: unsupported runtime profile {runtime['profile']!r}")
        if runtime["precision"] != "fp32":
            raise CatalogError(f"{model_id}: Inflect v2 runtime precision must be 'fp32'")
        if type(runtime["onnx_opset"]) is not int or runtime["onnx_opset"] <= 0:
            raise CatalogError(f"{model_id}: runtime onnx_opset must be a positive integer")
        normalized_runtime = {**runtime, "layout": "split"}

        upstream = raw_model["upstream"]
        if not isinstance(upstream, Mapping):
            raise CatalogError(f"{model_id}: upstream must be an object")
        _require_inflect_fields(
            upstream,
            {"provider", "repository", "revision", "license", "source_repository", "source_revision"},
            f"{model_id} upstream",
        )
        if upstream["provider"] != "huggingface":
            raise CatalogError(f"{model_id}: upstream provider must be huggingface")
        repository = upstream["repository"]
        if not isinstance(repository, str) or _INFLECT_HF_REPOSITORY_RE.fullmatch(repository) is None:
            raise CatalogError(f"{model_id}: upstream repository must have owner/name form")
        revision = upstream["revision"]
        if not isinstance(revision, str) or _INFLECT_HF_REVISION_RE.fullmatch(revision) is None:
            raise CatalogError(f"{model_id}: upstream revision must be a lowercase 40-character SHA")
        source_repository = upstream["source_repository"]
        if (
            not isinstance(source_repository, str)
            or _INFLECT_HF_REPOSITORY_RE.fullmatch(source_repository) is None
        ):
            raise CatalogError(f"{model_id}: source_repository must have owner/name form")
        source_revision = upstream["source_revision"]
        if (
            not isinstance(source_revision, str)
            or _INFLECT_HF_REVISION_RE.fullmatch(source_revision) is None
        ):
            raise CatalogError(
                f"{model_id}: source_revision must be a lowercase 40-character SHA"
            )
        license_name = _require_inflect_text(upstream["license"], f"{model_id} upstream license")

        raw_artifacts = raw_model["artifacts"]
        if not isinstance(raw_artifacts, Sequence) or isinstance(raw_artifacts, (str, bytes)):
            raise CatalogError(f"{model_id}: artifacts must be a sequence")
        if len(raw_artifacts) != 2:
            raise CatalogError(f"{model_id}: artifacts must contain exactly duration and decode")
        artifacts: list[Artifact] = []
        seen_roles: set[str] = set()
        for raw_artifact in raw_artifacts:
            if not isinstance(raw_artifact, Mapping):
                raise CatalogError(f"{model_id}: artifact must be an object")
            _require_inflect_fields(
                raw_artifact,
                {"role", "format", "filename", "url", "size", "sha256"},
                f"{model_id} artifact",
            )
            role = raw_artifact["role"]
            if not isinstance(role, str) or role not in _INFLECT_ARTIFACT_FILENAMES:
                raise CatalogError(f"{model_id}: unknown Inflect artifact role {role!r}")
            if role in seen_roles:
                raise CatalogError(f"{model_id}: duplicate {role!r} artifact")
            seen_roles.add(role)
            if raw_artifact["format"] != "onnx":
                raise CatalogError(f"{model_id}/{role}: format must be 'onnx'")
            filename = _require_inflect_safe_id(
                raw_artifact["filename"], f"{model_id}/{role} filename"
            )
            if filename != _INFLECT_ARTIFACT_FILENAMES[role]:
                raise CatalogError(f"{model_id}/{role}: filename must be {role}.onnx")
            expected_url = (
                f"https://huggingface.co/{repository}/resolve/{revision}/onnx/{filename}"
            )
            url = raw_artifact["url"]
            if not isinstance(url, str) or url != expected_url:
                raise CatalogError(
                    f"{model_id}/{role}: URL is not pinned to upstream repository/revision"
                )
            size = raw_artifact["size"]
            if type(size) is not int or size <= 0:
                raise CatalogError(f"{model_id}/{role}: size must be a positive integer")
            sha256 = raw_artifact["sha256"]
            if not isinstance(sha256, str) or _INFLECT_SHA256_RE.fullmatch(sha256) is None:
                raise CatalogError(
                    f"{model_id}/{role}: sha256 must be lowercase 64-character hex"
                )
            source_path = f"onnx/{filename}"
            artifacts.append(
                Artifact(
                    role=role,
                    filename=filename,
                    url=url,
                    size=size,
                    sha256=sha256,
                    format="onnx",
                    metadata={
                        "path": source_path,
                        "source": {
                            "provider": "huggingface",
                            "repository": repository,
                            "revision": revision,
                            "path": source_path,
                        },
                    },
                )
            )
        if seen_roles != set(_INFLECT_ARTIFACT_FILENAMES):
            raise CatalogError(f"{model_id}: artifacts must contain exactly duration and decode")

        raw_metadata = raw_model["metadata"]
        if not isinstance(raw_metadata, Mapping):
            raise CatalogError(f"{model_id}: metadata must be an object")
        metadata = {
            **raw_metadata,
            "name": name,
            "version": version,
            "language": language,
            "language_codes": (language,),
            "voice_mode": "fixed",
            "controls": normalized_controls,
            "runtime": normalized_runtime,
            "upstream": dict(upstream),
            "source_repository": source_repository,
            "source_revision": source_revision,
            "license": license_name,
            "voice_details": voice_details,
        }
        result.append(
            CatalogItem(
                system="inflect",
                id=model_id,
                kind="model",
                artifacts=tuple(artifacts),
                aliases=tuple(aliases),
                sample_rate=sample_rate,
                voices=(default_voice,),
                default_voice=default_voice,
                metadata=metadata,
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
