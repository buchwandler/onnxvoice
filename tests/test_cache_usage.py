"""Tests for cache usage reporting and GC byte tracking."""

from __future__ import annotations

import hashlib
import json
import os
from argparse import Namespace
from types import SimpleNamespace

import pytest

import onnxvoice.cli as cli
from onnxvoice.errors import NotInstalledError
from onnxvoice.manager import OnnxVoice
from onnxvoice.store import AssetStore
from onnxvoice.types import Artifact, CatalogItem

# ---------------------------------------------------------------------------
# Cache usage tests
# ---------------------------------------------------------------------------


class TestCacheUsage:
    def _make_item(self, system="test", item_id="one", content=b"model-data"):
        sha = hashlib.sha256(content).hexdigest()
        return CatalogItem(
            system=system,
            id=item_id,
            kind="model",
            artifacts=(Artifact("model", "model.onnx", "file:///dev/null", len(content), sha),),
        )

    def test_empty_cache_usage(self, tmp_path):
        store = AssetStore(tmp_path / "cache")
        usage = store.usage()
        assert usage.installation_count == 0
        assert usage.install_logical_bytes == 0
        assert usage.orphan_blob_count == 0
        assert usage.auxiliary_bytes == 0
        assert usage.pocket_voice_state_count == 0
        assert usage.pocket_voice_state_bytes == 0
        assert usage.orphan_auxiliary_count == 0
        assert usage.orphan_auxiliary_bytes == 0

    def test_full_root_scan_reports_categories_filesystem_and_unmanaged_paths(
        self, tmp_path, monkeypatch
    ):
        cache_root = tmp_path / "cache"
        store = AssetStore(cache_root)
        blob_path = store.blobs / "aa" / "digest"
        blob_path.parent.mkdir(parents=True)
        blob_path.write_bytes(b"blob")

        install_path = store.installs / "test" / "untracked" / "model.onnx"
        install_path.parent.mkdir(parents=True)
        install_path.write_bytes(b"install")
        catalog_path = cache_root / "catalogs" / "index.json"
        catalog_path.parent.mkdir(parents=True)
        catalog_path.write_bytes(b"cat")
        lock_path = cache_root / "locks" / "gc.lock"
        lock_path.parent.mkdir(parents=True)
        lock_path.write_bytes(b"lock")
        state_path = cache_root / "pocket-voice-states" / "orphan" / "state.bin"
        state_path.parent.mkdir(parents=True)
        state_path.write_bytes(b"state")
        legacy_path = cache_root / "pocket-voice-prompts" / "old.bin"
        legacy_path.parent.mkdir(parents=True)
        legacy_path.write_bytes(b"legacy")
        unknown_path = cache_root / "my-important-files" / "data.bin"
        unknown_path.parent.mkdir(parents=True)
        unknown_path.write_bytes(b"other")
        try:
            (unknown_path.parent / "shared-blob").hardlink_to(blob_path)
            has_hardlink = True
        except Exception:
            has_hardlink = False

        outside = tmp_path / "outside"
        outside.mkdir()
        (outside / "must-not-count.bin").write_bytes(b"outside-data")
        try:
            (cache_root / "outside-link").symlink_to(outside, target_is_directory=True)
            has_symlink = True
        except Exception:
            has_symlink = False

        monkeypatch.setattr(
            "onnxvoice.store.shutil.disk_usage",
            lambda _path: SimpleNamespace(total=1000, used=400, free=600),
        )
        usage = store.usage()

        expected_apparent = sum(
            path.stat().st_size
            for path in (
                blob_path,
                install_path,
                catalog_path,
                lock_path,
                state_path,
                legacy_path,
                unknown_path,
            )
        ) + (blob_path.stat().st_size if has_hardlink else 0)
        assert usage.cache_apparent_bytes == expected_apparent
        assert usage.cache_unique_apparent_bytes == expected_apparent - (
            blob_path.stat().st_size if has_hardlink else 0
        )
        assert usage.cache_allocated_bytes >= 0
        assert (
            sum(item.allocated_bytes for item in usage.top_level_usage)
            == usage.cache_allocated_bytes
        )
        assert usage.filesystem_total_bytes == 1000
        assert usage.filesystem_used_bytes == 400
        assert usage.filesystem_free_bytes == 600
        assert usage.install_apparent_bytes == install_path.stat().st_size
        assert usage.catalog_bytes == catalog_path.stat().st_size
        assert usage.lock_bytes == lock_path.stat().st_size
        assert usage.auxiliary_bytes == state_path.stat().st_size
        assert usage.legacy_cache_bytes == legacy_path.stat().st_size
        assert usage.other_cache_bytes == unknown_path.stat().st_size
        assert cache_root / "my-important-files" in usage.unmanaged_paths
        if has_symlink:
            assert cache_root / "outside-link" in usage.unmanaged_paths

    def test_scan_tolerates_file_disappearing_during_stat(self, tmp_path, monkeypatch):
        store = AssetStore(tmp_path / "cache")
        missing = store.root / "unmanaged" / "vanishing.bin"
        missing.parent.mkdir()
        missing.write_bytes(b"gone")
        real_stat = os.stat

        def disappearing_stat(path, *args, **kwargs):
            if os.fspath(path) == os.fspath(missing):
                raise FileNotFoundError(os.fspath(path))
            return real_stat(path, *args, **kwargs)

        monkeypatch.setattr("onnxvoice.store.os.stat", disappearing_stat)
        usage = store.usage()

        assert usage.cache_apparent_bytes == 0
        assert any(os.fspath(missing) in error for error in usage.scan_errors)

    def test_pocket_auxiliary_metrics_and_cache_cli(self, tmp_path, monkeypatch, capsys):
        state_payload = b"pocket voice state bytes"
        state_sha = hashlib.sha256(state_payload).hexdigest()
        source_record = {
            "name": "alba",
            "source": {
                "provider": "huggingface",
                "repository": "kyutai/pocket-tts",
                "revision": "d" * 40,
                "path": "languages/english_2026-04/embeddings/alba.safetensors",
            },
            "size": len(state_payload),
            "sha256": state_sha,
        }
        bundle_file = tmp_path / "bundle.json"
        bundle_file.write_bytes(b"bundle")
        bundle_sha = hashlib.sha256(b"bundle").hexdigest()
        item = CatalogItem(
            system="pocket",
            id="english_2026-04",
            kind="bundle",
            artifacts=(
                Artifact(
                    "bundle_metadata",
                    bundle_file.name,
                    bundle_file.as_uri(),
                    bundle_file.stat().st_size,
                    bundle_sha,
                ),
            ),
            metadata={"predefined_voice_names": ["alba"], "voice_states": [source_record]},
        )
        store = AssetStore(tmp_path / "cache")
        store.install(item)
        key = {
            "system": "pocket",
            "asset_kind": "predefined_voice_state",
            "model_repo": "kyutai/pocket-tts",
            "model_revision": "d" * 40,
            "asset_path": "languages/english_2026-04/embeddings/alba.safetensors",
            "bundle_id": "english_2026-04",
            "voice_name": "alba",
            "expected_size": len(state_payload),
            "expected_sha256": state_sha,
        }
        digest = hashlib.sha256(
            json.dumps(key, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        state_dir = store.root / "pocket-voice-states" / digest[:2] / digest
        state_dir.mkdir(parents=True)
        state_path = state_dir / "alba.safetensors"
        state_path.write_bytes(state_payload)
        metadata_path = state_dir / "alba.json"
        metadata_path.write_text(
            json.dumps(
                {"key": key, "size": len(state_payload), "sha256": state_sha},
                sort_keys=True,
            ),
            encoding="utf-8",
        )

        referenced_usage = store.usage()
        assert (
            referenced_usage.auxiliary_bytes
            == state_path.stat().st_size + metadata_path.stat().st_size
        )
        assert referenced_usage.pocket_voice_state_count == 1
        assert referenced_usage.pocket_voice_state_bytes == len(state_payload)
        assert referenced_usage.orphan_auxiliary_count == 0
        store.remove("pocket", "english_2026-04")

        monkeypatch.setattr(cli, "_manager", lambda args: SimpleNamespace(store=store))
        assert cli._cmd_cache_info(Namespace(format="json")) == 0
        json_usage = json.loads(capsys.readouterr().out)
        assert (
            json_usage["auxiliary_bytes"]
            == state_path.stat().st_size + metadata_path.stat().st_size
        )
        assert json_usage["pocket_voice_state_count"] == 1
        assert json_usage["pocket_voice_state_bytes"] == len(state_payload)
        assert json_usage["orphan_auxiliary_count"] == 1
        assert (
            json_usage["orphan_auxiliary_bytes"]
            == state_path.stat().st_size + metadata_path.stat().st_size
        )
        assert isinstance(json_usage["filesystem_free_bytes"], int)
        assert isinstance(json_usage["cache"]["allocated_bytes"], int)
        assert isinstance(json_usage["orphan_auxiliary_bytes"], int)

        assert cli._cmd_cache_info(Namespace(format="table")) == 0
        text_usage = capsys.readouterr().out
        assert "Pocket voice states" in text_usage
        assert "Reclaimable Pocket" in text_usage

        assert cli._cmd_cache_gc(Namespace(dry_run=True)) == 0
        assert "Would remove 1 Pocket voice states" in capsys.readouterr().out
        assert cli._cmd_cache_gc(Namespace(dry_run=False)) == 0
        gc_output = capsys.readouterr().out
        assert "Removed 1 Pocket voice states" in gc_output
        assert not state_path.exists()

    def test_install_logical_bytes(self, tmp_path):
        payload = b"x" * 1000
        source = tmp_path / "model.onnx"
        source.write_bytes(payload)
        sha = hashlib.sha256(payload).hexdigest()
        item = CatalogItem(
            system="test",
            id="one",
            kind="model",
            artifacts=(Artifact("model", source.name, source.as_uri(), len(payload), sha),),
        )
        store = AssetStore(tmp_path / "cache")
        store.install(item)
        usage = store.usage()
        assert usage.installation_count == 1
        assert usage.install_logical_bytes == 1000

    def test_blob_apparent_bytes(self, tmp_path):
        payload = b"x" * 1000
        source = tmp_path / "model.onnx"
        source.write_bytes(payload)
        sha = hashlib.sha256(payload).hexdigest()
        item = CatalogItem(
            system="test",
            id="one",
            kind="model",
            artifacts=(Artifact("model", source.name, source.as_uri(), len(payload), sha),),
        )
        store = AssetStore(tmp_path / "cache")
        store.install(item)
        usage = store.usage()
        assert usage.blob_apparent_bytes == 1000

    def test_unique_file_bytes_no_double_count(self, tmp_path):
        """Hard-linked files should be counted once in unique_file_bytes."""
        payload = b"x" * 1000
        source = tmp_path / "model.onnx"
        source.write_bytes(payload)
        sha = hashlib.sha256(payload).hexdigest()
        item = CatalogItem(
            system="test",
            id="one",
            kind="model",
            artifacts=(Artifact("model", source.name, source.as_uri(), len(payload), sha),),
        )
        store = AssetStore(tmp_path / "cache")
        store.install(item)
        usage = store.usage()

        # If hard links are supported, unique should equal blob size
        # If copy fallback was used, unique might be 2x
        blob_path = store.blobs / sha[:2] / sha
        inst_path = store.installs / "test" / "one" / source.name

        if blob_path.exists() and inst_path.exists():
            blob_stat = blob_path.stat()
            inst_stat = inst_path.stat()
            same_inode = (
                blob_stat.st_ino == inst_stat.st_ino and blob_stat.st_dev == inst_stat.st_dev
            )
            if same_inode:
                assert usage.unique_file_bytes == 1000
                assert usage.hardlinked_install_artifact_count == 1
                assert usage.hardlinked_install_bytes == 1000
                assert usage.copied_install_artifact_count == 0
            else:
                # Copy fallback: both counted separately
                assert usage.unique_file_bytes == 2000
                assert usage.copied_install_artifact_count == 1
                assert usage.copied_install_bytes == 1000
                assert usage.hardlinked_install_artifact_count == 0

    def test_orphan_blob_detection(self, tmp_path):
        payload = b"x" * 1000
        source = tmp_path / "model.onnx"
        source.write_bytes(payload)
        sha = hashlib.sha256(payload).hexdigest()
        item = CatalogItem(
            system="test",
            id="one",
            kind="model",
            artifacts=(Artifact("model", source.name, source.as_uri(), len(payload), sha),),
        )
        store = AssetStore(tmp_path / "cache")
        store.install(item)
        store.remove("test", "one")

        usage = store.usage()
        assert usage.orphan_blob_count == 1
        assert usage.orphan_blob_bytes == 1000
        assert usage.installation_count == 0


# ---------------------------------------------------------------------------
# GC report tests
# ---------------------------------------------------------------------------


class TestGcReport:
    def test_gc_returns_report(self, tmp_path):
        payload = b"x" * 1000
        source = tmp_path / "model.onnx"
        source.write_bytes(payload)
        sha = hashlib.sha256(payload).hexdigest()
        item = CatalogItem(
            system="test",
            id="one",
            kind="model",
            artifacts=(Artifact("model", source.name, source.as_uri(), len(payload), sha),),
        )
        store = AssetStore(tmp_path / "cache")
        store.install(item)
        store.remove("test", "one")

        report = store.gc()
        assert report.removed_blobs == 1
        assert report.removed_bytes == 1000

    def test_gc_plan_is_shared_and_stale_plan_is_revalidated(self, tmp_path):
        payload = b"plan-revalidation"
        source = tmp_path / "model.onnx"
        source.write_bytes(payload)
        sha = hashlib.sha256(payload).hexdigest()
        item = CatalogItem(
            system="test",
            id="planned",
            kind="model",
            artifacts=(Artifact("model", source.name, source.as_uri(), len(payload), sha),),
        )
        store = AssetStore(tmp_path / "cache")
        blob_path = store.blobs / sha[:2] / sha
        blob_path.parent.mkdir(parents=True)
        blob_path.write_bytes(payload)

        stale_plan = store.plan_gc()
        assert len(stale_plan.candidates) == 1
        assert stale_plan.candidates[0].category == "orphan_blob"
        assert stale_plan.apparent_bytes == len(payload)

        store.install(item)
        report = store.gc(plan=stale_plan)
        assert report.removed_blobs == 0
        assert blob_path.exists()

        store.remove("test", "planned")
        current_plan = store.plan_gc()
        report = store.gc(plan=current_plan)
        assert report.removed_blobs == 1
        assert report.removed_apparent_bytes == len(payload)
        assert report.removed_allocated_bytes == current_plan.allocated_bytes
        assert not blob_path.exists()

    def test_gc_legacy_cleanup_is_opt_in_and_does_not_touch_unknown_paths(self, tmp_path):
        store = AssetStore(tmp_path / "cache")
        legacy_file = store.root / "pocket-voice-prompts" / "nested" / "old.bin"
        legacy_file.parent.mkdir(parents=True)
        legacy_file.write_bytes(b"legacy")
        unknown_file = store.root / "my-important-files" / "keep.bin"
        unknown_file.parent.mkdir(parents=True)
        unknown_file.write_bytes(b"keep")

        standard_plan = store.plan_gc()
        assert not standard_plan.candidates
        assert legacy_file.exists()
        assert unknown_file.exists()
        assert store.gc().removed_candidates == 0

        legacy_plan = store.plan_gc(include_legacy=True)
        assert [candidate.category for candidate in legacy_plan.candidates] == ["legacy_cache"]
        assert legacy_plan.apparent_bytes == len(b"legacy")
        report = store.gc(plan=legacy_plan)
        assert report.removed_legacy_count == 1
        assert report.removed_legacy_bytes == len(b"legacy")
        assert not legacy_file.exists()
        assert unknown_file.exists()

    def test_gc_no_orphans(self, tmp_path):
        store = AssetStore(tmp_path / "cache")
        report = store.gc()
        assert report.removed_blobs == 0
        assert report.removed_bytes == 0

    def test_gc_preserves_referenced_blobs(self, tmp_path):
        payload = b"x" * 1000
        source = tmp_path / "model.onnx"
        source.write_bytes(payload)
        sha = hashlib.sha256(payload).hexdigest()
        item = CatalogItem(
            system="test",
            id="one",
            kind="model",
            artifacts=(Artifact("model", source.name, source.as_uri(), len(payload), sha),),
        )
        store = AssetStore(tmp_path / "cache")
        store.install(item)

        # Don't remove the installation - blob should be preserved
        report = store.gc()
        assert report.removed_blobs == 0

    def test_gc_shared_blob_not_removed(self, tmp_path):
        """A blob shared by two installations should not be GC'd when only one is removed."""
        payload = b"shared-data"
        source = tmp_path / "model.onnx"
        source.write_bytes(payload)
        sha = hashlib.sha256(payload).hexdigest()
        artifact = Artifact("model", source.name, source.as_uri(), len(payload), sha)

        item1 = CatalogItem(system="test", id="one", kind="model", artifacts=(artifact,))
        item2 = CatalogItem(system="test", id="two", kind="model", artifacts=(artifact,))

        store = AssetStore(tmp_path / "cache")
        store.install(item1)
        store.install(item2)

        # Remove only one
        store.remove("test", "one")
        report = store.gc()
        assert report.removed_blobs == 0  # Still referenced by "two"

        # Now remove the other
        store.remove("test", "two")
        report = store.gc()
        assert report.removed_blobs == 1

    def test_removal_planning_distinguishes_copies_and_shared_blob_gc(self, tmp_path, monkeypatch):
        def fail_link(*_args, **_kwargs):
            raise OSError("hard links disabled for this test")

        monkeypatch.setattr(os, "link", fail_link, raising=False)
        payload = b"shared-removal-plan"
        source = tmp_path / "model.onnx"
        source.write_bytes(payload)
        sha = hashlib.sha256(payload).hexdigest()
        artifact = Artifact("model", source.name, source.as_uri(), len(payload), sha)
        store = AssetStore(tmp_path / "cache")
        store.install(CatalogItem(system="test", id="one", kind="model", artifacts=(artifact,)))
        store.install(CatalogItem(system="test", id="two", kind="model", artifacts=(artifact,)))
        installations = store.installed()

        single_plan = store.plan_remove((installations[0],), gc=True)
        assert single_plan.immediate_reclaimable_bytes >= len(payload)
        assert single_plan.gc_reclaimable_bytes == 0
        combined_plan = store.plan_remove(tuple(installations), gc=True)
        assert combined_plan.logical_bytes == 2 * len(payload)
        assert combined_plan.immediate_reclaimable_bytes >= 2 * len(payload)
        assert combined_plan.gc_reclaimable_bytes == len(payload)
        assert combined_plan.total_reclaimable_bytes == (
            combined_plan.immediate_reclaimable_bytes + combined_plan.gc_reclaimable_bytes
        )
        assert all(installation.path.exists() for installation in installations)

        first = store.remove("test", "one")
        second = store.remove("test", "two")
        assert first.immediate_reclaimable_bytes >= len(payload)
        assert second.immediate_reclaimable_bytes >= len(payload)
        gc_report = store.gc()
        assert gc_report.removed_bytes == len(payload)

    def test_manager_removal_plan_validates_variants_and_all_refs_before_mutation(self, tmp_path):
        manager = OnnxVoice(cache_dir=tmp_path / "cache")
        payload = b"variant-removal-plan"
        source = tmp_path / "variant.onnx"
        source.write_bytes(payload)
        artifact = Artifact(
            "model",
            source.name,
            source.as_uri(),
            len(payload),
            hashlib.sha256(payload).hexdigest(),
        )
        item = CatalogItem(system="test", id="voice", kind="model", artifacts=(artifact,))
        manager.store.install(manager._with_selection_identity(item, quality="low"))
        manager.store.install(manager._with_selection_identity(item, quality="high"))

        with pytest.raises(ValueError, match="Multiple installations"):
            manager.plan_remove(("test:voice",), gc=True)
        with pytest.raises(NotInstalledError, match="test:missing"):
            manager.plan_remove(("test:voice", "test:missing"), all_variants=True)
        assert len(manager.store.installed()) == 2

        selected_plan = manager.plan_remove(("test:voice",), quality="low", gc=True)
        assert len(selected_plan.installations) == 1
        assert selected_plan.installations[0].storage_id == manager._selection_storage_id(
            "voice", quality="low"
        )
        all_plan = manager.plan_remove(("test:voice",), all_variants=True, gc=True)
        assert len(all_plan.installations) == 2
        assert len(manager.store.installed()) == 2

    def test_cli_remove_preview_and_actual_gc_report_reclaimed_bytes(
        self, tmp_path, monkeypatch, capsys
    ):
        manager = OnnxVoice(cache_dir=tmp_path / "cache")
        payload = b"cli-removal-data"
        source = tmp_path / "cli-model.onnx"
        source.write_bytes(payload)
        sha = hashlib.sha256(payload).hexdigest()
        item = CatalogItem(
            system="test",
            id="cli-model",
            kind="model",
            artifacts=(Artifact("model", source.name, source.as_uri(), len(payload), sha),),
        )
        installation = manager.store.install(item)
        blob_path = manager.store.blobs / sha[:2] / sha
        monkeypatch.setattr(cli, "_manager", lambda _args: manager)

        args = Namespace(
            ref=["test:cli-model"],
            quality=None,
            distribution=None,
            all_variants=False,
            gc=True,
            yes=False,
            dry_run=True,
        )
        assert cli._cmd_remove(args) == 0
        preview = capsys.readouterr().out
        assert "Installation logical size:" in preview
        assert "Immediate unique bytes freed:" in preview
        assert "Newly orphaned blobs: 1" in preview
        assert "GC reclaimable after removal:" in preview
        assert installation.path.exists()
        assert blob_path.exists()

        invalid_args = Namespace(**{**vars(args), "ref": ["test:cli-model", "test:missing"]})
        assert cli._cmd_remove(invalid_args) == 2
        capsys.readouterr()
        assert installation.path.exists()
        assert blob_path.exists()

        actual_args = Namespace(**{**vars(args), "dry_run": False})
        assert cli._cmd_remove(actual_args) == 0
        actual = capsys.readouterr().out
        assert "Immediate bytes reclaimed:" in actual
        assert "GC bytes reclaimed:" in actual
        assert "Total bytes reclaimed:" in actual
        assert not installation.path.exists()
        assert not blob_path.exists()

    def test_cli_legacy_gc_flag_and_cache_details(self, tmp_path, monkeypatch, capsys):
        manager = OnnxVoice(cache_dir=tmp_path / "cache")
        legacy = manager.store.root / "pocket-voice-prompts" / "retired.bin"
        legacy.parent.mkdir(parents=True)
        legacy.write_bytes(b"legacy")
        unknown = manager.store.root / "my-important-files" / "keep.bin"
        unknown.parent.mkdir(parents=True)
        unknown.write_bytes(b"keep")
        monkeypatch.setattr(cli, "_manager", lambda _args: manager)

        assert cli._cmd_cache_info(Namespace(format="table", details=True)) == 0
        details = capsys.readouterr().out
        assert "Filesystem free" in details
        assert "Cache disk usage" in details
        assert "Reclaimable now" in details
        assert "pocket-voice-prompts" in details
        assert "my-important-files" in details

        assert cli._cmd_cache_gc(Namespace(dry_run=True, include_legacy=False)) == 0
        assert "No orphan cache files" in capsys.readouterr().out
        assert legacy.exists() and unknown.exists()
        assert cli._cmd_cache_gc(Namespace(dry_run=True, include_legacy=True)) == 0
        assert "legacy cache files" in capsys.readouterr().out
        assert cli._cmd_cache_gc(Namespace(dry_run=False, include_legacy=True)) == 0
        assert "legacy cache files" in capsys.readouterr().out
        assert not legacy.exists()
        assert unknown.exists()

    def test_installed_sorting_applies_before_json_serialization(self, monkeypatch, capsys):
        records = [
            SimpleNamespace(
                installed=True,
                system="z-system",
                kind="model",
                id="small",
                language_codes=(),
                gender="unknown",
                quality=None,
                distribution=None,
                size_bytes=10,
                version=None,
                installation=SimpleNamespace(path="/cache/small"),
            ),
            SimpleNamespace(
                installed=True,
                system="a-system",
                kind="model",
                id="large",
                language_codes=(),
                gender="unknown",
                quality=None,
                distribution=None,
                size_bytes=100,
                version=None,
                installation=SimpleNamespace(path="/cache/large"),
            ),
        ]
        fake_manager = SimpleNamespace(inventory=lambda **_kwargs: records)
        monkeypatch.setattr(cli, "_manager", lambda _args: fake_manager)
        args = Namespace(
            system=None,
            language=None,
            gender=None,
            kind=None,
            quality=None,
            distribution=None,
            format="json",
            sort="size",
            desc=True,
        )

        assert cli._cmd_installed(args) == 0
        data = json.loads(capsys.readouterr().out)
        assert [item["ID"] for item in data["items"]] == ["large", "small"]

        args.sort = "id"
        args.desc = False
        assert cli._cmd_installed(args) == 0
        data = json.loads(capsys.readouterr().out)
        assert [item["ID"] for item in data["items"]] == ["large", "small"]

        args.format = "table"
        args.sort = "size"
        args.desc = True
        assert cli._cmd_installed(args) == 0
        human = capsys.readouterr().out
        assert human.index("large") < human.index("small")

    def test_uninstall_is_a_remove_alias(self, tmp_path, monkeypatch, capsys):
        manager = OnnxVoice(cache_dir=tmp_path / "cache")
        payload = b"uninstall-alias"
        source = tmp_path / "alias.onnx"
        source.write_bytes(payload)
        sha = hashlib.sha256(payload).hexdigest()
        manager.store.install(
            CatalogItem(
                system="test",
                id="alias",
                kind="model",
                artifacts=(Artifact("model", source.name, source.as_uri(), len(payload), sha),),
            )
        )
        monkeypatch.setattr(cli, "_manager", lambda _args: manager)

        assert cli.main(["uninstall", "test:alias", "--dry-run", "--gc"]) == 0
        assert "Would remove: test:alias" in capsys.readouterr().out
        assert len(manager.store.installed()) == 1

    def test_cli_help_documents_cache_and_inventory_options(self, capsys):
        help_cases = (
            (["cache", "info", "--help"], "--details"),
            (["cache", "gc", "--help"], "--include-legacy"),
            (["installed", "--help"], "--sort"),
        )
        for argv, expected in help_cases:
            with pytest.raises(SystemExit) as exc_info:
                cli.main(argv)
            assert exc_info.value.code == 0
            assert expected in capsys.readouterr().out

        with pytest.raises(SystemExit) as exc_info:
            cli.main(["--help"])
        assert exc_info.value.code == 0
        assert "uninstall" in capsys.readouterr().out
