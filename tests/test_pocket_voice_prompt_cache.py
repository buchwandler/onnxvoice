"""Tests for the OnnxVoice-owned Pocket voice-prompt cache."""

from __future__ import annotations

import hashlib
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from onnxvoice.errors import VoicePromptIntegrityError, VoicePromptOfflineError
from onnxvoice.pocket_voice_prompts import PocketVoicePrompts, pinned_source_url

REPOSITORY = "kyutai/tts-voices"
REVISION = "a" * 40
EVIDENCE = f"https://huggingface.co/{REPOSITORY}/blob/{REVISION}/README.md"
PAYLOAD = b"fake wav payload"
REF = "kyutai-tts-voices:alba-mackenna/casual"


def _record(**overrides: Any) -> dict[str, Any]:
    source_path = overrides.pop("source_path", "alba-mackenna/casual.wav")
    record: dict[str, Any] = {
        "id": "alba-mackenna/casual",
        "dataset": "alba-mackenna",
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


def _catalog(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema": 1,
        "kind": "pocket-voice-prompt-catalog",
        "source": {
            "provider": "huggingface",
            "repository": REPOSITORY,
            "requested_revision": "main",
            "revision": REVISION,
            "prompt_count": 1,
            "license": "mixed",
            "snapshot_note": None,
        },
        "prompts": [record],
    }


def _store(
    tmp_path: Path, record: dict[str, Any] | None = None, **kwargs: Any
) -> PocketVoicePrompts:
    catalog_path = (
        tmp_path / f"voice-prompts-{len(list(tmp_path.glob('voice-prompts-*.json')))}.json"
    )
    catalog_path.write_text(json.dumps(_catalog(record or _record())), encoding="utf-8")
    return PocketVoicePrompts(
        cache_dir=tmp_path / "cache",
        catalog_sources={"pocket_voice_prompts": str(catalog_path)},
        **kwargs,
    )


def _fake_download(payload: bytes = PAYLOAD):
    def download(source: Any, *, local_dir: Path, offline: bool) -> Path:
        target = local_dir / Path(source.path).name
        target.write_bytes(payload)
        return target

    return download


def _cached_paths(tmp_path: Path) -> set[Path]:
    root = tmp_path / "cache" / "pocket-voice-prompts"
    return {path.parent for path in root.glob("*/*/record.json")}


def test_cache_hit_avoids_redownload(tmp_path: Path) -> None:
    store = _store(tmp_path)
    with patch(
        "onnxvoice.pocket_voice_prompts.download_huggingface_file",
        side_effect=_fake_download(),
    ) as download:
        first = store.fetch(REF)
        second = store.fetch(REF)
    assert first == second
    assert first.read_bytes() == PAYLOAD
    download.assert_called_once()


def test_force_refetches_into_the_same_entry(tmp_path: Path) -> None:
    store = _store(tmp_path)
    with patch(
        "onnxvoice.pocket_voice_prompts.download_huggingface_file",
        side_effect=_fake_download(),
    ) as download:
        first = store.fetch(REF)
        second = store.fetch(REF, force=True)
    assert first == second
    assert download.call_count == 2


def test_cache_identity_includes_integrity_fields(tmp_path: Path) -> None:
    first = _store(tmp_path)
    with patch(
        "onnxvoice.pocket_voice_prompts.download_huggingface_file",
        side_effect=_fake_download(),
    ):
        first.fetch(REF)
    other_payload = b"other wav payload"
    second = _store(
        tmp_path,
        _record(
            size=len(other_payload),
            sha256=hashlib.sha256(other_payload).hexdigest(),
        ),
    )
    with patch(
        "onnxvoice.pocket_voice_prompts.download_huggingface_file",
        side_effect=_fake_download(other_payload),
    ):
        second.fetch(REF, refresh=True)
    entries = _cached_paths(tmp_path)
    assert len(entries) == 2


def test_corrupt_cache_entry_is_repaired(tmp_path: Path) -> None:
    store = _store(tmp_path)
    with patch(
        "onnxvoice.pocket_voice_prompts.download_huggingface_file",
        side_effect=_fake_download(),
    ) as download:
        cached = store.fetch(REF)
        cached.write_bytes(b"corrupted")
        repaired = store.fetch(REF)
    assert download.call_count == 2
    assert repaired.read_bytes() == PAYLOAD


def test_corrupt_cache_entry_is_rejected_offline(tmp_path: Path) -> None:
    online = _store(tmp_path)
    with patch(
        "onnxvoice.pocket_voice_prompts.download_huggingface_file",
        side_effect=_fake_download(),
    ):
        cached = online.fetch(REF)
    cached.write_bytes(b"corrupted")
    offline = _store(tmp_path, offline=True)
    with (
        patch("onnxvoice.pocket_voice_prompts.download_huggingface_file") as download,
        pytest.raises(VoicePromptIntegrityError, match="offline"),
    ):
        offline.fetch(REF)
    download.assert_not_called()


def test_offline_cache_hit_needs_no_network(tmp_path: Path) -> None:
    online = _store(tmp_path)
    with patch(
        "onnxvoice.pocket_voice_prompts.download_huggingface_file",
        side_effect=_fake_download(),
    ):
        cached = online.fetch(REF)
    offline = _store(tmp_path, offline=True)
    with patch("onnxvoice.pocket_voice_prompts.download_huggingface_file") as download:
        assert offline.fetch(REF) == cached
    download.assert_not_called()


def test_offline_cache_miss_is_typed_and_actionable(tmp_path: Path) -> None:
    with (
        patch("onnxvoice.pocket_voice_prompts.download_huggingface_file") as download,
        pytest.raises(VoicePromptOfflineError, match="network access"),
    ):
        _store(tmp_path, offline=True).fetch(REF)
    download.assert_not_called()


def test_concurrent_fetch_downloads_once_and_never_corrupts(tmp_path: Path) -> None:
    store = _store(tmp_path)

    def slow_download(source: Any, *, local_dir: Path, offline: bool) -> Path:
        time.sleep(0.2)
        target = local_dir / Path(source.path).name
        target.write_bytes(PAYLOAD)
        return target

    with (
        patch(
            "onnxvoice.pocket_voice_prompts.download_huggingface_file",
            side_effect=slow_download,
        ) as download,
        ThreadPoolExecutor(max_workers=4) as pool,
    ):
        paths = list(pool.map(lambda _: store.fetch(REF), range(4)))

    assert download.call_count == 1
    assert set(paths) == {paths[0]}
    assert paths[0].read_bytes() == PAYLOAD
