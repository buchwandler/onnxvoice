from __future__ import annotations

import json
import os
import time
import urllib.request
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from platformdirs import user_cache_path

from .errors import AssetNotFoundError, CatalogError, OfflineError
from .types import Artifact, CatalogItem

DEFAULT_SOURCES = {
    "piper": "https://raw.githubusercontent.com/buchwandler/piper-onnx-voices/main/catalog/voices.json",
    "kokoro": "https://raw.githubusercontent.com/buchwandler/kokoro-onnx-models/main/catalog/models.json",
}


@dataclass(slots=True)
class CatalogClient:
    cache_dir: Path | None = None
    sources: dict[str, str] | None = None
    ttl_seconds: int = 24 * 60 * 60
    offline: bool = False

    def __post_init__(self) -> None:
        self.cache_dir = Path(self.cache_dir or user_cache_path("onnxvoice")) / "catalogs"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
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
        return self.cache_dir / f"{system}.json"

    def _read_source(self, source: str) -> bytes:
        if source.startswith(("http://", "https://")):
            if self.offline:
                raise OfflineError(f"Catalog source requires network access: {source}")
            request = urllib.request.Request(source, headers={"User-Agent": "onnxvoice/0"})
            with urllib.request.urlopen(request, timeout=30) as response:
                return response.read()
        return Path(source).expanduser().read_bytes()

    def load_raw(self, system: str, *, refresh: bool = False) -> dict[str, Any]:
        system = system.lower()
        if not self.sources or system not in self.sources:
            raise CatalogError(f"No catalog source configured for system {system!r}")
        cache_path = self._cache_path(system)
        fresh = (
            cache_path.exists() and (time.time() - cache_path.stat().st_mtime) < self.ttl_seconds
        )
        if cache_path.exists() and (fresh or self.offline) and not refresh:
            return json.loads(cache_path.read_text(encoding="utf-8"))
        payload = self._read_source(self.sources[system])
        try:
            data = json.loads(payload)
        except json.JSONDecodeError as exc:
            raise CatalogError(f"Invalid JSON in {system} catalog") from exc
        tmp = cache_path.with_suffix(".json.tmp")
        tmp.write_bytes(payload)
        os.replace(tmp, cache_path)
        return data

    def list(self, system: str, *, refresh: bool = False) -> list[CatalogItem]:
        system = system.lower()
        raw = self.load_raw(system, refresh=refresh)
        if system == "piper":
            return _parse_piper(raw)
        if system == "kokoro":
            return _parse_kokoro(raw)
        raise CatalogError(f"No parser for system {system!r}")

    def resolve(
        self, ref: str, *, refresh: bool = False, quality: str | None = None
    ) -> CatalogItem:
        system, item_id = parse_ref(ref)
        items = self.list(system, refresh=refresh)
        for item in items:
            if item.id == item_id or item_id in item.aliases:
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
        raise AssetNotFoundError(f"Unknown catalog item: {ref}")


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
                    metadata={"path": raw.get("path")},
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
                },
            )
        )
    return result


def _parse_kokoro(data: dict[str, Any]) -> list[CatalogItem]:
    models = data.get("models")
    if not isinstance(models, dict):
        raise CatalogError("Kokoro catalog is missing the 'models' mapping")
    result: list[CatalogItem] = []
    for model_id, entry in models.items():
        distributions = [d for d in entry.get("distributions", ()) if d.get("runtime_ready", True)]
        if not distributions:
            continue
        distribution = distributions[0]
        artifacts: list[Artifact] = []
        for raw in distribution.get("artifacts", ()):
            artifacts.append(
                Artifact(
                    role=str(raw.get("role") or raw.get("id") or "artifact"),
                    filename=str(raw.get("local_name") or raw.get("id")),
                    url=raw.get("url"),
                    size=raw.get("size"),
                    sha256=raw.get("sha256"),
                    quality=raw.get("quality"),
                    metadata={
                        "id": raw.get("id"),
                        "format": raw.get("format"),
                        "handling": raw.get("handling"),
                    },
                )
            )
        runtime = entry.get("runtime") or {}
        result.append(
            CatalogItem(
                system="kokoro",
                id=str(model_id),
                kind="model",
                artifacts=tuple(artifacts),
                sample_rate=entry.get("sample_rate"),
                voices=tuple(runtime.get("voices") or ()),
                default_voice=runtime.get("default_voice"),
                metadata={
                    "model_version": entry.get("model_version"),
                    "frontend": entry.get("frontend"),
                    "language_codes": entry.get("language_codes") or [],
                    "runtime": runtime,
                    "distribution_id": distribution.get("id"),
                    "release_tag": distribution.get("release_tag"),
                },
            )
        )
    return result


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
