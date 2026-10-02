"""Managed Pocket reference-voice prompts: catalog, resolution, fetch, and cache.

Reference voice prompts are upstream WAV assets (for example
``alba-mackenna/casual.wav`` from ``kyutai/tts-voices``). They are cataloged with a
pinned revision, size, SHA-256, and license provenance, and are lazily cached under
the OnnxVoice cache root. Predefined ``.safetensors`` voice states are a different
asset class and are handled by :mod:`onnxvoice.systems.pocket`.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, NoReturn
from urllib.parse import quote, unquote, urlparse

from platformdirs import user_cache_path

from .catalog import CatalogClient
from .checksums import verify_file
from .errors import (
    AssetAccessError,
    AssetDownloadError,
    AssetNotFoundError,
    IntegrityError,
    InvalidVoicePromptRefError,
    OfflineError,
    UnknownVoicePromptError,
    VoicePromptAccessError,
    VoicePromptCatalogError,
    VoicePromptDownloadError,
    VoicePromptFormatError,
    VoicePromptIntegrityError,
    VoicePromptNotFoundError,
    VoicePromptOfflineError,
)
from .huggingface import HuggingFaceSource, download_huggingface_file
from .store import FileLock, ProgressCallback
from .types import AssetProgress, validate_relative_path

PROMPT_REF_PREFIX = "kyutai-tts-voices:"
PROMPT_CATALOG_SYSTEM = "pocket_voice_prompts"
PROMPT_CATALOG_KIND = "pocket-voice-prompt-catalog"
PROMPT_CACHE_NAME = "pocket-voice-prompts"
DEFAULT_PROMPT_CATALOG_SOURCE = "https://raw.githubusercontent.com/buchwandler/pocket-onnx-bundles/main/catalog/voice-prompts.json"
SUPPORTED_PROMPT_FORMATS = frozenset({"wav"})

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_SHA_RE = re.compile(r"^[0-9a-f]{40}$")
_REPOSITORY_RE = re.compile(r"^[^/\\\s]+/[^/\\\s]+$")
_HF_RESOLVE_RE = re.compile(r"^/([^/]+)/([^/]+)/resolve/([^/]+)/(.+)$")

_PROMPT_FIELDS = frozenset(
    {
        "id",
        "dataset",
        "variant",
        "base_id",
        "format",
        "source_path",
        "source_repository",
        "source_revision",
        "url",
        "size",
        "sha256",
        "license",
        "license_note",
        "license_evidence",
        "metadata",
    }
)
_SOURCE_FIELDS = frozenset(
    {
        "provider",
        "repository",
        "requested_revision",
        "revision",
        "prompt_count",
        "license",
        "snapshot_note",
    }
)


@dataclass(frozen=True, slots=True)
class PocketVoicePrompt:
    """One cataloged reference-voice prompt with pinned provenance and integrity."""

    ref: str
    id: str
    source_path: str
    format: str
    source_repository: str
    source_revision: str
    url: str
    size: int
    sha256: str
    license: str
    dataset: str
    variant: str
    base_id: str | None = None
    license_note: str | None = None
    license_evidence: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def source(self) -> HuggingFaceSource:
        """Return the pinned Hugging Face source for this prompt."""
        return HuggingFaceSource(self.source_repository, self.source_revision, self.source_path)


def prompt_ref(prompt_id: str) -> str:
    """Return the canonical reference for one prompt ID."""
    return f"{PROMPT_REF_PREFIX}{prompt_id}"


def parse_prompt_ref(ref: str) -> str:
    """Parse a canonical ``kyutai-tts-voices:<id>`` reference into its prompt ID."""
    if not isinstance(ref, str) or not ref or ref != ref.strip():
        raise InvalidVoicePromptRefError(f"Invalid voice prompt reference: {ref!r}")
    if not ref.startswith(PROMPT_REF_PREFIX):
        raise InvalidVoicePromptRefError(
            f"Voice prompt reference must start with {PROMPT_REF_PREFIX!r}: {ref!r}"
        )
    prompt_id = ref.removeprefix(PROMPT_REF_PREFIX)
    if not prompt_id or prompt_id != prompt_id.strip():
        raise InvalidVoicePromptRefError(f"Invalid voice prompt reference: {ref!r}")
    _safe_prompt_path(prompt_id, "voice prompt id", error=InvalidVoicePromptRefError)
    return prompt_id


def pinned_source_url(repository: str, revision: str, path: str) -> str:
    """Return the revision-pinned Hugging Face download URL for one repository file."""
    repo = "/".join(quote(part, safe="") for part in repository.split("/"))
    encoded_revision = quote(revision, safe="")
    encoded_path = quote(path, safe="/")
    return f"https://huggingface.co/{repo}/resolve/{encoded_revision}/{encoded_path}?download=true"


def parse_prompt_catalog(data: Any) -> tuple[PocketVoicePrompt, ...]:
    """Parse and validate a voice-prompt catalog into typed prompt records."""
    if not isinstance(data, Mapping):
        raise VoicePromptCatalogError("Voice prompt catalog must be an object")
    if set(data) != {"schema", "kind", "source", "prompts"}:
        raise VoicePromptCatalogError(
            "Voice prompt catalog must have exactly schema, kind, source, prompts"
        )
    if data.get("schema") != 1:
        raise VoicePromptCatalogError("Voice prompt catalog schema must be 1")
    if data.get("kind") != PROMPT_CATALOG_KIND:
        raise VoicePromptCatalogError(f"Unexpected voice prompt catalog kind: {data.get('kind')!r}")
    source = data.get("source")
    if not isinstance(source, Mapping) or set(source) != _SOURCE_FIELDS:
        raise VoicePromptCatalogError("Voice prompt catalog source has unexpected fields")
    if source.get("provider") != "huggingface":
        raise VoicePromptCatalogError("Voice prompt catalog provider must be huggingface")
    repository = source.get("repository")
    if not isinstance(repository, str) or _REPOSITORY_RE.fullmatch(repository) is None:
        raise VoicePromptCatalogError("Voice prompt catalog repository is invalid")
    requested_revision = source.get("requested_revision")
    if not isinstance(requested_revision, str) or not requested_revision:
        raise VoicePromptCatalogError("Voice prompt catalog requested_revision is required")
    revision = source.get("revision")
    if not isinstance(revision, str) or _SHA_RE.fullmatch(revision) is None:
        raise VoicePromptCatalogError(
            "Voice prompt catalog revision must be a lowercase 40-character SHA"
        )
    if not isinstance(source.get("license"), str) or not source["license"]:
        raise VoicePromptCatalogError("Voice prompt catalog license is required")
    if source.get("snapshot_note") is not None and not isinstance(source["snapshot_note"], str):
        raise VoicePromptCatalogError("Voice prompt catalog snapshot_note must be a string")

    raw_prompts = data.get("prompts")
    if not isinstance(raw_prompts, Sequence) or isinstance(raw_prompts, (str, bytes)):
        raise VoicePromptCatalogError("Voice prompt catalog prompts must be a sequence")
    prompts = tuple(
        _parse_prompt_record(raw, repository=repository, revision=revision) for raw in raw_prompts
    )
    ids = [prompt.id for prompt in prompts]
    _reject_duplicates(ids, "voice prompt id")
    _reject_duplicates([prompt.source_path for prompt in prompts], "voice prompt source path")
    prompt_count = source.get("prompt_count")
    if type(prompt_count) is not int or prompt_count != len(prompts):
        raise VoicePromptCatalogError(
            f"Voice prompt count mismatch: source declares {prompt_count!r}, "
            f"catalog holds {len(prompts)}"
        )
    return tuple(sorted(prompts, key=lambda prompt: prompt.id))


def _reject_duplicates(values: list[str], label: str) -> None:
    seen: set[str] = set()
    duplicates: set[str] = set()
    for value in values:
        if value in seen:
            duplicates.add(value)
        seen.add(value)
    if duplicates:
        raise VoicePromptCatalogError(f"Duplicate {label}: {', '.join(sorted(duplicates))}")


def _safe_prompt_path(value: Any, label: str, *, error: type[Exception] = VoicePromptCatalogError):
    if not isinstance(value, str):
        raise error(f"{label} must be a string")
    try:
        validate_relative_path(value, field_name=label)
    except ValueError as exc:
        raise error(f"Unsafe {label}: {value!r}") from exc
    return value


def _required_text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise VoicePromptCatalogError(f"{label} is required")
    return value


def _optional_text(value: Any, label: str) -> str | None:
    if value is None:
        return None
    return _required_text(value, label)


def _parse_prompt_record(raw: Any, *, repository: str, revision: str) -> PocketVoicePrompt:
    if not isinstance(raw, Mapping):
        raise VoicePromptCatalogError("Voice prompt record must be an object")
    if set(raw) != _PROMPT_FIELDS:
        raise VoicePromptCatalogError("Voice prompt record has unexpected fields")
    prompt_id = _safe_prompt_path(raw.get("id"), "voice prompt id")
    source_path = _safe_prompt_path(raw.get("source_path"), "voice prompt source path")
    format_name = raw.get("format")
    if format_name not in SUPPORTED_PROMPT_FORMATS:
        raise VoicePromptFormatError(
            f"Voice prompt {prompt_id!r} uses unsupported format {format_name!r}; "
            f"supported formats: {', '.join(sorted(SUPPORTED_PROMPT_FORMATS))}"
        )
    if not source_path.endswith(".wav") or prompt_id != source_path.removesuffix(".wav"):
        raise VoicePromptCatalogError(
            f"{prompt_id}: id must be the source path without its .wav suffix"
        )
    if raw.get("source_repository") != repository:
        raise VoicePromptCatalogError(f"{prompt_id}: source_repository is not the catalog source")
    if raw.get("source_revision") != revision:
        raise VoicePromptCatalogError(
            f"{prompt_id}: source_revision is not the pinned catalog revision"
        )
    if raw.get("url") != pinned_source_url(repository, revision, source_path):
        raise VoicePromptCatalogError(f"{prompt_id}: url is not pinned to its source")
    size = raw.get("size")
    if type(size) is not int or size <= 0:
        raise VoicePromptCatalogError(f"{prompt_id}: size must be positive")
    sha256 = raw.get("sha256")
    if not isinstance(sha256, str) or _SHA256_RE.fullmatch(sha256) is None:
        raise VoicePromptCatalogError(f"{prompt_id}: sha256 must be 64-character hex")
    license_name = _required_text(raw.get("license"), f"{prompt_id}: license")
    license_note = _optional_text(raw.get("license_note"), f"{prompt_id}: license_note")
    if license_name == "NOASSERTION" and license_note is None:
        raise VoicePromptCatalogError(
            f"{prompt_id}: NOASSERTION license requires an explicit license_note"
        )
    dataset = _required_text(raw.get("dataset"), f"{prompt_id}: dataset")
    if dataset != prompt_id.split("/", 1)[0]:
        raise VoicePromptCatalogError(f"{prompt_id}: dataset must be the top-level prompt group")
    variant = _required_text(raw.get("variant"), f"{prompt_id}: variant")
    base_id = raw.get("base_id")
    if base_id is not None:
        base_id = _safe_prompt_path(base_id, "voice prompt base_id")
        if base_id == prompt_id:
            raise VoicePromptCatalogError(f"{prompt_id}: base_id must differ from the prompt id")
    if variant != "original" and base_id is None:
        raise VoicePromptCatalogError(f"{prompt_id}: variant {variant!r} requires a base_id")
    metadata = raw.get("metadata")
    if not isinstance(metadata, Mapping):
        raise VoicePromptCatalogError(f"{prompt_id}: metadata must be an object")
    return PocketVoicePrompt(
        ref=prompt_ref(prompt_id),
        id=prompt_id,
        source_path=source_path,
        format=format_name,
        source_repository=repository,
        source_revision=revision,
        url=pinned_source_url(repository, revision, source_path),
        size=size,
        sha256=sha256,
        license=license_name,
        license_note=license_note,
        license_evidence=_optional_text(
            raw.get("license_evidence"), f"{prompt_id}: license_evidence"
        ),
        dataset=dataset,
        variant=variant,
        base_id=base_id,
        metadata=dict(metadata),
    )


class PocketVoicePrompts:
    """Resolve, list, and fetch cataloged Pocket reference-voice prompts."""

    def __init__(
        self,
        *,
        cache_dir: str | Path | None = None,
        offline: bool = False,
        catalog_sources: Mapping[str, str] | None = None,
    ) -> None:
        """Create a prompt store on one cache root and catalog source set."""
        self.root = Path(
            cache_dir or os.environ.get("ONNXVOICE_CACHE_DIR") or user_cache_path("onnxvoice")
        )
        self.offline = offline
        self.cache_dir = self.root / PROMPT_CACHE_NAME
        sources = {PROMPT_CATALOG_SYSTEM: DEFAULT_PROMPT_CATALOG_SOURCE}
        if catalog_sources:
            sources.update(catalog_sources)
        self._catalog = CatalogClient(cache_dir=self.root, sources=sources, offline=offline)

    def list(
        self,
        *,
        dataset: str | None = None,
        variant: str | None = None,
        license: str | None = None,
        refresh: bool = False,
        progress: ProgressCallback | None = None,
    ) -> tuple[PocketVoicePrompt, ...]:
        """List cataloged prompts, optionally filtered by dataset, variant, or license."""
        prompts = self._prompts(refresh=refresh, progress=progress)
        if dataset is not None:
            prompts = tuple(prompt for prompt in prompts if prompt.dataset == dataset)
        if variant is not None:
            prompts = tuple(prompt for prompt in prompts if prompt.variant == variant)
        if license is not None:
            prompts = tuple(prompt for prompt in prompts if prompt.license == license)
        return prompts

    def resolve(
        self,
        ref: str,
        *,
        refresh: bool = False,
        progress: ProgressCallback | None = None,
    ) -> PocketVoicePrompt:
        """Resolve one canonical prompt reference to its typed catalog record."""
        prompt_id = parse_prompt_ref(ref)
        for prompt in self._prompts(refresh=refresh, progress=progress):
            if prompt.id == prompt_id:
                return prompt
        raise UnknownVoicePromptError(
            f"Unknown voice prompt {ref!r}. Discover prompts with list_pocket_voice_prompts()."
        )

    def resolve_source(self, value: str, *, refresh: bool = False) -> PocketVoicePrompt:
        """Resolve an ``hf://`` or pinned ``https://`` prompt URL through the catalog.

        Raw URLs never weaken the catalog contract: the URL must name a cataloged
        asset, and the pinned catalog revision, size, SHA-256, and license apply.
        """
        prompts = self._prompts(refresh=refresh)
        if isinstance(value, str) and value.startswith("hf://"):
            parts = value.removeprefix("hf://").split("/", 2)
            if (
                len(parts) != 3
                or not parts[2]
                or _REPOSITORY_RE.fullmatch("/".join(parts[:2])) is None
            ):
                raise InvalidVoicePromptRefError(f"Invalid hf:// prompt URL: {value!r}")
            return self._match_source(
                prompts, value, repository="/".join(parts[:2]), source_path=parts[2]
            )
        parsed = urlparse(value) if isinstance(value, str) else None
        if parsed is None or parsed.scheme != "https" or parsed.netloc != "huggingface.co":
            raise InvalidVoicePromptRefError(f"Unsupported voice prompt URL: {value!r}")
        match = _HF_RESOLVE_RE.fullmatch(parsed.path)
        if match is None:
            raise InvalidVoicePromptRefError(f"Unsupported voice prompt URL: {value!r}")
        owner, name, revision, path = match.groups()
        if _SHA_RE.fullmatch(revision) is None:
            raise InvalidVoicePromptRefError(
                f"Voice prompt URL is not pinned to an immutable revision: {value!r}"
            )
        return self._match_source(
            prompts,
            value,
            repository=f"{owner}/{name}",
            revision=revision,
            source_path=unquote(path),
        )

    def fetch(
        self,
        ref: str,
        *,
        refresh: bool = False,
        force: bool = False,
        progress: ProgressCallback | None = None,
    ) -> Path:
        """Fetch one prompt into the OnnxVoice cache and return its verified local path."""
        prompt = self.resolve(ref, refresh=refresh, progress=progress)
        return self._fetch(prompt, force=force, progress=progress)

    def _prompts(
        self, *, refresh: bool, progress: ProgressCallback | None = None
    ) -> tuple[PocketVoicePrompt, ...]:
        try:
            data = self._catalog.load_raw(PROMPT_CATALOG_SYSTEM, refresh=refresh, progress=progress)
        except OfflineError as exc:
            raise VoicePromptOfflineError(
                f"Voice prompt catalog {PROMPT_CATALOG_SYSTEM!r} is not cached and offline "
                "mode is enabled. Load it once with network access enabled."
            ) from exc
        return parse_prompt_catalog(data)

    @staticmethod
    def _match_source(
        prompts: Sequence[PocketVoicePrompt],
        value: str,
        *,
        repository: str,
        source_path: str,
        revision: str | None = None,
    ) -> PocketVoicePrompt:
        for prompt in prompts:
            if prompt.source_repository != repository or prompt.source_path != source_path:
                continue
            if revision is not None and prompt.source_revision != revision:
                continue
            return prompt
        raise UnknownVoicePromptError(
            f"No cataloged voice prompt matches {value!r}. Raw URLs must name a "
            "revision-pinned catalog asset; use a kyutai-tts-voices: reference instead."
        )

    def _fetch(
        self, prompt: PocketVoicePrompt, *, force: bool, progress: ProgressCallback | None
    ) -> Path:
        key = {
            "system": "pocket",
            "asset_kind": "voice_prompt",
            "provider": "huggingface",
            "repository": prompt.source_repository,
            "revision": prompt.source_revision,
            "source_path": prompt.source_path,
            "expected_size": prompt.size,
            "expected_sha256": prompt.sha256,
        }
        digest = hashlib.sha256(
            json.dumps(key, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        cache_dir = self.cache_dir / digest[:2] / digest
        asset_path = cache_dir / Path(prompt.source_path).name
        record_path = cache_dir / "record.json"
        lock_path = self.cache_dir / "locks" / f"{digest}.lock"
        with FileLock(lock_path, timeout=600.0):
            if self._is_cached(prompt, asset_path, record_path, key) and not force:
                self._emit(
                    progress,
                    AssetProgress(
                        "artifact_cached",
                        prompt.ref,
                        prompt.source_path,
                        completed=prompt.size,
                        total=prompt.size,
                        target=str(asset_path),
                    ),
                )
                return asset_path
            if self.offline:
                if asset_path.exists() or record_path.exists():
                    raise VoicePromptIntegrityError(
                        f"Cached voice prompt {prompt.ref} failed integrity validation and "
                        "cannot be repaired while offline."
                    )
                raise VoicePromptOfflineError(
                    f"Voice prompt {prompt.ref} is not cached and offline mode is enabled. "
                    "Fetch it once with network access enabled."
                )
            self._download(
                prompt,
                key=key,
                cache_dir=cache_dir,
                asset_path=asset_path,
                record_path=record_path,
                progress=progress,
            )
            return asset_path

    @staticmethod
    def _is_cached(
        prompt: PocketVoicePrompt, asset_path: Path, record_path: Path, key: Mapping[str, Any]
    ) -> bool:
        if not asset_path.is_file() or not record_path.is_file():
            return False
        try:
            record = json.loads(record_path.read_text(encoding="utf-8"))
            if not isinstance(record, dict) or record.get("key") != dict(key):
                return False
            verify_file(asset_path, expected_size=prompt.size, sha256=prompt.sha256)
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError, IntegrityError):
            return False
        return True

    def _download(
        self,
        prompt: PocketVoicePrompt,
        *,
        key: Mapping[str, Any],
        cache_dir: Path,
        asset_path: Path,
        record_path: Path,
        progress: ProgressCallback | None,
    ) -> None:
        self._emit(
            progress,
            AssetProgress(
                "download_started",
                prompt.ref,
                prompt.source_path,
                target=str(asset_path),
                total=prompt.size,
            ),
        )
        cache_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="onnxvoice-prompt-") as download_dir:
            try:
                downloaded = download_huggingface_file(
                    prompt.source, local_dir=Path(download_dir), offline=self.offline
                )
            except (OfflineError, AssetAccessError, AssetNotFoundError, AssetDownloadError) as exc:
                _raise_prompt_download_error(prompt, exc)
            if not downloaded.is_file():
                raise VoicePromptDownloadError(
                    f"Downloaded voice prompt {prompt.ref!r} is missing from its staging directory"
                )
            fd, temporary_name = tempfile.mkstemp(prefix=".prompt.", dir=cache_dir)
            os.close(fd)
            temporary = Path(temporary_name)
            try:
                _copy_with_progress(downloaded, temporary, prompt, progress)
                self._emit(
                    progress,
                    AssetProgress(
                        "verify_started", prompt.ref, prompt.source_path, target=str(asset_path)
                    ),
                )
                try:
                    verify_file(temporary, expected_size=prompt.size, sha256=prompt.sha256)
                except IntegrityError as exc:
                    raise VoicePromptIntegrityError(
                        f"Voice prompt {prompt.ref} failed size/SHA-256 verification at its "
                        "pinned upstream revision"
                    ) from exc
                self._emit(
                    progress,
                    AssetProgress(
                        "verify_completed", prompt.ref, prompt.source_path, target=str(asset_path)
                    ),
                )
                os.replace(temporary, asset_path)
                record = {"key": dict(key), "size": prompt.size, "sha256": prompt.sha256}
                fd, temporary_name = tempfile.mkstemp(prefix=".record.", dir=cache_dir)
                os.close(fd)
                temporary_record = Path(temporary_name)
                try:
                    temporary_record.write_text(
                        json.dumps(record, sort_keys=True) + "\n", encoding="utf-8"
                    )
                    os.replace(temporary_record, record_path)
                finally:
                    temporary_record.unlink(missing_ok=True)
            finally:
                temporary.unlink(missing_ok=True)
        self._emit(
            progress,
            AssetProgress(
                "download_completed",
                prompt.ref,
                prompt.source_path,
                completed=prompt.size,
                total=prompt.size,
                target=str(asset_path),
            ),
        )

    @staticmethod
    def _emit(progress: ProgressCallback | None, event: AssetProgress) -> None:
        if progress is not None:
            progress(event)


def _copy_with_progress(
    source: Path, destination: Path, prompt: PocketVoicePrompt, progress: ProgressCallback | None
) -> None:
    completed = 0
    with source.open("rb") as reader, destination.open("wb") as writer:
        while chunk := reader.read(1024 * 1024):
            writer.write(chunk)
            completed += len(chunk)
            if progress is not None:
                progress(
                    AssetProgress(
                        "download_progress",
                        prompt.ref,
                        prompt.source_path,
                        completed=completed,
                        total=prompt.size,
                    )
                )


def _raise_prompt_download_error(prompt: PocketVoicePrompt, exc: Exception) -> NoReturn:
    label = f"{prompt.source_repository}@{prompt.source_revision}:{prompt.source_path}"
    if isinstance(exc, OfflineError):
        raise VoicePromptOfflineError(
            f"Voice prompt {prompt.ref} is not available locally and offline mode is enabled"
        ) from exc
    if isinstance(exc, AssetAccessError):
        raise VoicePromptAccessError(
            f"Voice prompt asset {label} requires authentication or repository access. "
            "Authenticate with 'hf auth login' or HF_TOKEN, then retry."
        ) from exc
    if isinstance(exc, AssetNotFoundError):
        raise VoicePromptNotFoundError(
            f"Voice prompt asset {label} is not present at its pinned revision"
        ) from exc
    if isinstance(exc, AssetDownloadError):
        raise VoicePromptDownloadError(f"Could not download voice prompt asset {label}") from exc
    raise VoicePromptDownloadError(f"Could not download voice prompt asset {label}") from exc


__all__ = [
    "DEFAULT_PROMPT_CATALOG_SOURCE",
    "PROMPT_CATALOG_KIND",
    "PROMPT_CATALOG_SYSTEM",
    "PROMPT_REF_PREFIX",
    "SUPPORTED_PROMPT_FORMATS",
    "PocketVoicePrompt",
    "PocketVoicePrompts",
    "parse_prompt_catalog",
    "parse_prompt_ref",
    "pinned_source_url",
    "prompt_ref",
]
