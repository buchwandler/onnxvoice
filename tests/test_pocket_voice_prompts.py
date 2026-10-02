"""Tests for managed Pocket voice-prompt parsing, resolution, and fetching."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from onnxvoice.errors import (
    AssetAuthenticationError,
    AssetDownloadError,
    AssetNotFoundError,
    InvalidVoicePromptRefError,
    UnknownVoicePromptError,
    VoicePromptAccessError,
    VoicePromptCatalogError,
    VoicePromptDownloadError,
    VoicePromptFormatError,
    VoicePromptIntegrityError,
    VoicePromptNotFoundError,
    VoicePromptOfflineError,
)
from onnxvoice.pocket_voice_prompts import (
    PocketVoicePrompt,
    PocketVoicePrompts,
    parse_prompt_catalog,
    parse_prompt_ref,
    pinned_source_url,
)

REPOSITORY = "kyutai/tts-voices"
REVISION = "a" * 40
EVIDENCE = f"https://huggingface.co/{REPOSITORY}/blob/{REVISION}/README.md"
PAYLOAD = b"fake wav payload"
REF = "kyutai-tts-voices:alba-mackenna/casual"


def _record(prompt_id: str = "alba-mackenna/casual", **overrides: Any) -> dict[str, Any]:
    source_path = overrides.pop("source_path", f"{prompt_id}.wav")
    record: dict[str, Any] = {
        "id": prompt_id,
        "dataset": prompt_id.split("/", 1)[0],
        "variant": "original",
        "base_id": None,
        "format": "wav",
        "source_path": source_path,
        "source_repository": REPOSITORY,
        "source_revision": REVISION,
        "url": pinned_source_url(REPOSITORY, REVISION, source_path),
        "size": len(PAYLOAD),
        "sha256": hashlib.sha256(PAYLOAD).hexdigest(),
        "license": "CC-BY-4.0",
        "license_note": None,
        "license_evidence": EVIDENCE,
        "metadata": {},
    }
    record.update(overrides)
    return record


def _catalog(*records: dict[str, Any]) -> dict[str, Any]:
    prompts = list(records) or [_record()]
    return {
        "schema": 1,
        "kind": "pocket-voice-prompt-catalog",
        "source": {
            "provider": "huggingface",
            "repository": REPOSITORY,
            "requested_revision": "main",
            "revision": REVISION,
            "prompt_count": len(prompts),
            "license": "mixed",
            "snapshot_note": None,
        },
        "prompts": prompts,
    }


def _store(
    tmp_path: Path,
    catalog: dict[str, Any] | None = None,
    *,
    source: Path | None = None,
    **kwargs: Any,
) -> PocketVoicePrompts:
    catalog_path = source or tmp_path / "voice-prompts.json"
    if source is None:
        catalog_path.write_text(json.dumps(catalog or _catalog()), encoding="utf-8")
    return PocketVoicePrompts(
        cache_dir=tmp_path / "cache",
        catalog_sources={"pocket_voice_prompts": str(catalog_path)},
        **kwargs,
    )


def _fake_download(payload: bytes = PAYLOAD) -> Callable[..., Path]:
    def download(source: Any, *, local_dir: Path, offline: bool) -> Path:
        target = local_dir / Path(source.path).name
        target.write_bytes(payload)
        return target

    return download


class TestPromptRefs:
    def test_parses_nested_prompt_ids(self) -> None:
        assert parse_prompt_ref(REF) == "alba-mackenna/casual"
        assert (
            parse_prompt_ref("kyutai-tts-voices:ears/p050/freeform_speech_01")
            == "ears/p050/freeform_speech_01"
        )

    @pytest.mark.parametrize(
        "ref",
        [
            "",
            "alba-mackenna/casual",
            "kyutai-tts-voices:",
            "kyutai-tts-voices: casual",
            " system:asset/voice",
            "system:asset/voice",
        ],
    )
    def test_rejects_malformed_ref(self, ref: str) -> None:
        with pytest.raises(InvalidVoicePromptRefError):
            parse_prompt_ref(ref)

    @pytest.mark.parametrize(
        "ref",
        [
            "kyutai-tts-voices:../secrets",
            "kyutai-tts-voices:alba-mackenna/../../etc/passwd",
            "kyutai-tts-voices:/etc/passwd",
            "kyutai-tts-voices:alba\\mackenna",
        ],
    )
    def test_rejects_path_traversal(self, ref: str) -> None:
        with pytest.raises(InvalidVoicePromptRefError, match="Unsafe"):
            parse_prompt_ref(ref)


class TestPromptCatalogParsing:
    def test_parse_returns_typed_prompts(self) -> None:
        prompts = parse_prompt_catalog(_catalog())
        assert len(prompts) == 1
        prompt = prompts[0]
        assert isinstance(prompt, PocketVoicePrompt)
        assert prompt.ref == REF
        assert prompt.id == "alba-mackenna/casual"
        assert prompt.source_path == "alba-mackenna/casual.wav"
        assert prompt.format == "wav"
        assert prompt.source_repository == REPOSITORY
        assert prompt.source_revision == REVISION
        assert prompt.size == len(PAYLOAD)
        assert prompt.sha256 == hashlib.sha256(PAYLOAD).hexdigest()
        assert prompt.license == "CC-BY-4.0"
        assert prompt.dataset == "alba-mackenna"
        assert prompt.variant == "original"
        assert prompt.base_id is None
        assert prompt.source.path == prompt.source_path

    def test_prompts_are_sorted_by_id(self) -> None:
        catalog = _catalog(
            _record("vctk/p222_001"),
            _record("alba-mackenna/casual"),
            _record("ears/p050/freeform_speech_01"),
        )
        prompts = parse_prompt_catalog(catalog)
        assert [prompt.id for prompt in prompts] == [
            "alba-mackenna/casual",
            "ears/p050/freeform_speech_01",
            "vctk/p222_001",
        ]

    def test_records_carried_fields(self) -> None:
        catalog = _catalog(
            _record(
                "unmute-prod-website/degaulle-2",
                license="NOASSERTION",
                license_note="Upstream README does not establish a definitive license.",
                metadata={"reviewed": True},
            )
        )
        prompt = parse_prompt_catalog(catalog)[0]
        assert prompt.license == "NOASSERTION"
        assert prompt.license_note.startswith("Upstream README")
        assert prompt.license_evidence == EVIDENCE
        assert prompt.metadata == {"reviewed": True}

    @pytest.mark.parametrize(
        ("overrides", "match"),
        [
            ({"format": "mp3"}, "unsupported format"),
            ({"format": "safetensors"}, "unsupported format"),
            ({"size": 0}, "size must be positive"),
            ({"sha256": "not-hex"}, "sha256"),
            ({"license": ""}, "license is required"),
            ({"license": "NOASSERTION"}, "license_note"),
            ({"source_revision": "b" * 40}, "source_revision"),
            ({"source_repository": "kyutai/pocket-tts"}, "source_repository"),
            ({"url": "https://example.com/casual.wav"}, "url is not pinned"),
            ({"dataset": "wrong-group"}, "dataset"),
            ({"variant": "enhanced"}, "base_id"),
            ({"base_id": "alba-mackenna/casual"}, "base_id must differ"),
            ({"source_path": "vctk/p222_001.wav"}, "source path"),
            ({"id": "alba-mackenna/../casual"}, "Unsafe"),
        ],
    )
    def test_rejects_invalid_records(self, overrides: dict[str, Any], match: str) -> None:
        with pytest.raises((VoicePromptCatalogError, VoicePromptFormatError), match=match):
            parse_prompt_catalog(_catalog(_record(**overrides)))

    def test_rejects_duplicate_ids(self) -> None:
        with pytest.raises(VoicePromptCatalogError, match="Duplicate voice prompt id"):
            parse_prompt_catalog(_catalog(_record(), _record()))

    def test_rejects_prompt_count_mismatch(self) -> None:
        catalog = _catalog()
        catalog["source"]["prompt_count"] = 2
        with pytest.raises(VoicePromptCatalogError, match="count mismatch"):
            parse_prompt_catalog(catalog)

    @pytest.mark.parametrize(
        ("catalog", "match"),
        [
            ([], "must be an object"),
            (_catalog() | {"extra": 1}, "exactly schema, kind, source, prompts"),
            (_catalog() | {"schema": 2}, "schema must be 1"),
            (_catalog() | {"kind": "other"}, "kind"),
        ],
    )
    def test_rejects_invalid_catalog_shape(self, catalog: Any, match: str) -> None:
        with pytest.raises(VoicePromptCatalogError, match=match):
            parse_prompt_catalog(catalog)


class TestListingAndResolution:
    def test_list_all_prompts(self, tmp_path: Path) -> None:
        catalog = _catalog(
            _record("alba-mackenna/casual"),
            _record("vctk/p222_001", license="CC-BY-4.0"),
            _record(
                "expresso/en/ex04_enhanced",
                source_path="expresso/en/ex04_enhanced.wav",
                variant="enhanced",
                base_id="expresso/en/ex04",
                license="CC-BY-NC-4.0",
                license_note="Non-commercial.",
            ),
        )
        prompts = _store(tmp_path, catalog).list()
        assert [prompt.id for prompt in prompts] == [
            "alba-mackenna/casual",
            "expresso/en/ex04_enhanced",
            "vctk/p222_001",
        ]

    def test_filter_by_dataset(self, tmp_path: Path) -> None:
        catalog = _catalog(_record("vctk/p222_001"), _record("vctk/p222_002"), _record())
        store = _store(tmp_path, catalog)
        assert [p.id for p in store.list(dataset="vctk")] == ["vctk/p222_001", "vctk/p222_002"]
        assert store.list(dataset="missing") == ()

    def test_filter_by_variant(self, tmp_path: Path) -> None:
        catalog = _catalog(
            _record(),
            _record(
                "alba-mackenna/casual_enhanced",
                variant="enhanced",
                base_id="alba-mackenna/casual",
            ),
        )
        store = _store(tmp_path, catalog)
        enhanced = store.list(variant="enhanced")
        assert [prompt.id for prompt in enhanced] == ["alba-mackenna/casual_enhanced"]
        assert enhanced[0].base_id == "alba-mackenna/casual"
        assert [p.id for p in store.list(variant="original")] == ["alba-mackenna/casual"]

    def test_filter_by_license(self, tmp_path: Path) -> None:
        catalog = _catalog(
            _record(),
            _record(
                "ears/p050/freeform_speech_01",
                license="CC-BY-NC-4.0",
                license_note="Non-commercial.",
            ),
        )
        store = _store(tmp_path, catalog)
        assert [p.id for p in store.list(license="CC-BY-NC-4.0")] == [
            "ears/p050/freeform_speech_01"
        ]

    def test_resolve_valid_ref(self, tmp_path: Path) -> None:
        prompt = _store(tmp_path).resolve(REF)
        assert prompt.ref == REF
        assert prompt.source_revision == REVISION

    def test_reject_unknown_ref(self, tmp_path: Path) -> None:
        with pytest.raises(UnknownVoicePromptError):
            _store(tmp_path).resolve("kyutai-tts-voices:missing/voice")

    def test_reject_malformed_ref(self, tmp_path: Path) -> None:
        with pytest.raises(InvalidVoicePromptRefError):
            _store(tmp_path).resolve("alba-mackenna/casual")


class TestDirectResolver:
    def test_resolves_hf_url(self, tmp_path: Path) -> None:
        prompt = _store(tmp_path).resolve_source("hf://kyutai/tts-voices/alba-mackenna/casual.wav")
        assert prompt.ref == REF
        assert prompt.source_revision == REVISION

    def test_resolves_pinned_https_url(self, tmp_path: Path) -> None:
        prompt = _store(tmp_path).resolve_source(
            pinned_source_url(REPOSITORY, REVISION, "alba-mackenna/casual.wav")
        )
        assert prompt.ref == REF

    @pytest.mark.parametrize(
        "value",
        [
            "https://huggingface.co/kyutai/tts-voices/resolve/main/alba-mackenna/casual.wav",
            "s3://bucket/casual.wav",
            "https://example.com/casual.wav",
            "hf://not-a-repository",
        ],
    )
    def test_rejects_unpinned_or_unsupported_urls(self, tmp_path: Path, value: str) -> None:
        with pytest.raises(InvalidVoicePromptRefError):
            _store(tmp_path).resolve_source(value)

    def test_rejects_uncataloged_url(self, tmp_path: Path) -> None:
        with pytest.raises(UnknownVoicePromptError):
            _store(tmp_path).resolve_source("hf://kyutai/tts-voices/voice-zero/missing.wav")


class TestPromptFetch:
    def test_download_uses_exact_pinned_source(self, tmp_path: Path) -> None:
        store = _store(tmp_path)
        with patch(
            "onnxvoice.pocket_voice_prompts.download_huggingface_file",
            side_effect=_fake_download(),
        ) as download:
            path = store.fetch(REF)

        source = download.call_args.args[0]
        assert source.repository == REPOSITORY
        assert source.revision == REVISION
        assert source.path == "alba-mackenna/casual.wav"
        assert download.call_args.kwargs["offline"] is False
        assert path.read_bytes() == PAYLOAD

    def test_size_verification(self, tmp_path: Path) -> None:
        with (
            patch(
                "onnxvoice.pocket_voice_prompts.download_huggingface_file",
                side_effect=_fake_download(b"short"),
            ),
            pytest.raises(VoicePromptIntegrityError, match="verification"),
        ):
            _store(tmp_path).fetch(REF)

    def test_sha256_verification(self, tmp_path: Path) -> None:
        tampered = b"x" * len(PAYLOAD)
        with (
            patch(
                "onnxvoice.pocket_voice_prompts.download_huggingface_file",
                side_effect=_fake_download(tampered),
            ),
            pytest.raises(VoicePromptIntegrityError),
        ):
            _store(tmp_path).fetch(REF)

    def test_progress_callback(self, tmp_path: Path) -> None:
        events = []
        with patch(
            "onnxvoice.pocket_voice_prompts.download_huggingface_file",
            side_effect=_fake_download(),
        ):
            _store(tmp_path).fetch(REF, progress=events.append)
            _store(tmp_path, source=tmp_path / "voice-prompts.json").fetch(
                REF, progress=events.append
            )
        phases = [event.phase for event in events]
        assert phases.index("download_started") < phases.index("verify_started")
        assert phases.index("verify_started") < phases.index("verify_completed")
        assert phases.index("verify_completed") < phases.index("download_completed")
        assert "download_progress" in phases
        assert "artifact_cached" in phases

    @pytest.mark.parametrize(
        ("error", "expected", "match"),
        [
            (AssetAuthenticationError("gated"), VoicePromptAccessError, "authentication"),
            (AssetNotFoundError("gone"), VoicePromptNotFoundError, "not present"),
            (AssetDownloadError("boom"), VoicePromptDownloadError, "Could not download"),
        ],
    )
    def test_download_typed_errors(
        self, tmp_path: Path, error: Exception, expected: type[Exception], match: str
    ) -> None:
        with (
            patch("onnxvoice.pocket_voice_prompts.download_huggingface_file", side_effect=error),
            pytest.raises(expected, match=match),
        ):
            _store(tmp_path).fetch(REF)

    def test_offline_cache_hit(self, tmp_path: Path) -> None:
        online = _store(tmp_path)
        with patch(
            "onnxvoice.pocket_voice_prompts.download_huggingface_file",
            side_effect=_fake_download(),
        ) as download:
            cached = online.fetch(REF)
        assert cached.read_bytes() == PAYLOAD

        offline = _store(tmp_path, offline=True)
        with patch("onnxvoice.pocket_voice_prompts.download_huggingface_file") as download:
            assert offline.fetch(REF) == cached
        download.assert_not_called()

    def test_offline_cache_miss(self, tmp_path: Path) -> None:
        with (
            patch("onnxvoice.pocket_voice_prompts.download_huggingface_file") as download,
            pytest.raises(VoicePromptOfflineError, match="not cached"),
        ):
            _store(tmp_path, offline=True).fetch(REF)
        download.assert_not_called()


class TestManagerApi:
    def test_manager_exposes_prompt_api(self, tmp_path: Path) -> None:
        from onnxvoice.manager import OnnxVoice

        catalog_path = tmp_path / "voice-prompts.json"
        catalog_path.write_text(json.dumps(_catalog()), encoding="utf-8")
        manager = OnnxVoice(
            cache_dir=tmp_path / "cache",
            catalog_sources={"pocket_voice_prompts": str(catalog_path)},
        )
        assert [prompt.ref for prompt in manager.list_pocket_voice_prompts()] == [REF]
        assert manager.resolve_pocket_voice_prompt(REF).id == "alba-mackenna/casual"
        with patch(
            "onnxvoice.pocket_voice_prompts.download_huggingface_file",
            side_effect=_fake_download(),
        ):
            path = manager.fetch_pocket_voice_prompt(REF)
        assert path.read_bytes() == PAYLOAD
