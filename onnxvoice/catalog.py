from __future__ import annotations

import json
import os
import time
import urllib.request
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from platformdirs import user_cache_path

from .errors import AssetNotFoundError, CatalogError, OfflineError
from .store import FileLock, ProgressCallback
from .types import Artifact, AssetProgress, CatalogItem

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
}


@dataclass(slots=True)
class CatalogClient:
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
        system = system.lower()
        raw = self.load_raw(system, refresh=refresh, progress=progress)
        if system == "piper":
            return _parse_piper(raw)
        if system == "kokoro":
            return _parse_kokoro(raw)
        if system == "pocket":
            return _parse_pocket(raw)
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
                    "num_speakers": entry.get("num_speakers"),
                    "speaker_id_map": entry.get("speaker_id_map") or {},
                    "source_revision": source.get("revision"),
                    "source_repository": source.get("repository"),
                    "requested_revision": source.get("requested_revision"),
                },
            )
        )
    return result


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
    runtime = entry.get("runtime") or {}
    metadata = {
        "model_version": entry.get("model_version"),
        "frontend": entry.get("frontend"),
        "language_codes": entry.get("language_codes") or [],
        "runtime": runtime,
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
        result.append(_parse_kokoro_entry(model_id, entry))
    return result


def _kokoro_models(data: dict[str, Any]) -> dict[str, Mapping[str, Any]]:
    """Return the raw models mapping from Kokoro catalog data."""
    models = data.get("models")
    if not isinstance(models, dict):
        raise CatalogError("Kokoro catalog is missing the 'models' mapping")
    return models

def _parse_pocket(data: dict[str, Any]) -> list[CatalogItem]:
    bundles = data.get("bundles")
    if not isinstance(bundles, list):
        raise CatalogError("Pocket catalog is missing the 'bundles' list")
    source = data.get("source") or {}
    if not isinstance(source, dict):
        raise CatalogError("Pocket catalog source metadata must be an object")
    result: list[CatalogItem] = []
    for entry in bundles:
        if not isinstance(entry, dict):
            continue
        bundle_id = entry.get("id")
        if not bundle_id:
            raise CatalogError("Pocket bundle entry is missing 'id'")
        artifacts: list[Artifact] = []
        for raw in entry.get("artifacts", ()):
            if not isinstance(raw, dict):
                continue
            role = str(raw.get("role") or "")
            if role not in POCKET_ARTIFACT_ROLES:
                raise CatalogError(f"Unknown Pocket artifact role: {role!r}")
            artifacts.append(
                Artifact(
                    role=role,
                    filename=str(raw.get("filename") or Path(str(raw.get("url", ""))).name),
                    url=raw.get("url"),
                    size=raw.get("size"),
                    sha256=raw.get("sha256"),
                    quality=raw.get("quality"),
                    component=raw.get("component"),
                    format=raw.get("format"),
                    metadata=raw.get("metadata") or {},
                )
            )
        metadata = {
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
            **(entry.get("metadata") or {}),
        }
        result.append(
            CatalogItem(
                system="pocket",
                id=str(bundle_id),
                kind="bundle",
                artifacts=tuple(artifacts),
                aliases=tuple(entry.get("aliases") or ()),
                sample_rate=entry.get("sample_rate"),
                metadata=metadata,
            )
        )
    return result


def _select_pocket_profile(item: CatalogItem, quality: str | None) -> CatalogItem:
    """Select Pocket artifacts based on quality profile."""
    profiles = item.metadata.get("profiles") or {}
    selected_quality = quality or "int8"
    if selected_quality not in profiles and selected_quality != "int8":
        available = ", ".join(sorted(profiles.keys())) if profiles else "none"
        raise CatalogError(
            f"Unknown Pocket profile {selected_quality!r} for {item.id!r}; "
            f"available profiles: {available}"
        )
    profile = profiles.get(selected_quality) or {}
    result_artifacts: list[Artifact] = []
    for artifact in item.artifacts:
        if artifact.role not in POCKET_QUALIFIED_ROLES:
            result_artifacts.append(artifact)
            continue
        target_quality = profile.get(artifact.role)
        if target_quality is None:
            raise CatalogError(
                f"Profile {selected_quality!r} does not specify quality for role {artifact.role!r}"
            )
        matching = [
            a for a in item.artifacts if a.role == artifact.role and a.quality == target_quality
        ]
        if not matching:
            raise AssetNotFoundError(
                f"{item.ref} has no {artifact.role} artifact with quality={target_quality!r}"
            )
        if not any(a.role == artifact.role for a in result_artifacts):
            result_artifacts.append(matching[0])
    return CatalogItem(
        system=item.system,
        id=item.id,
        kind=item.kind,
        artifacts=tuple(result_artifacts),
        aliases=item.aliases,
        sample_rate=item.sample_rate,
        voices=item.voices,
        default_voice=item.default_voice,
        metadata={**item.metadata, "selected_quality": selected_quality},
    )


def filter_items(
    items: Iterable[CatalogItem],
    *,
    language: str | None = None,
    quality: str | None = None,
) -> list[CatalogItem]:
    result = list(items)
    if language:
        lang = language.lower().replace("-", "_")
        result = [
            item
            for item in result
            if str((item.metadata.get("language") or {}).get("code", ""))
            .lower()
            .replace("-", "_")
            .startswith(lang)
            or any(
                str(code).lower().replace("-", "_").startswith(lang)
                for code in item.metadata.get("language_codes", ())
            )
        ]
    if quality:
        result = [
            item
            for item in result
            if item.metadata.get("quality") == quality
            or any(a.role == "model" and a.quality == quality for a in item.artifacts)
        ]
    return result
