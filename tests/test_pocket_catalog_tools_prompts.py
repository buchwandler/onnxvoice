"""Tests for the Pocket voice-prompt catalog builder and verifier."""

from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from onnxvoice.catalog_tools.pocket import (
    PROMPT_CATALOG_KIND,
    CatalogError,
    _canonical_json,
    build_prompt_catalog,
    load_prompt_catalog,
    prompt_license,
    verify_prompt_catalog,
)
from onnxvoice.pocket_voice_prompts import parse_prompt_catalog, pinned_source_url

REVISION = "a" * 40
REPOSITORY = "kyutai/tts-voices"
BASE_PATHS = [
    "alba-mackenna/casual.wav",
    "voice-donations/lois.wav",
    "vctk/p222_001.wav",
]


def _tree(paths: list[str]) -> list[dict[str, Any]]:
    return [
        {
            "path": path,
            "size": 1000 + len(path),
            "lfs": {"oid": hashlib.sha256(path.encode("utf-8")).hexdigest()},
        }
        for path in paths
    ]


def _build(paths: list[str] | None = None) -> dict[str, Any]:
    with patch(
        "onnxvoice.catalog_tools.pocket._repository_tree",
        return_value=_tree(paths or BASE_PATHS),
    ):
        return build_prompt_catalog(resolved_revision=REVISION)


class TestBuildPromptCatalog:
    def test_same_revision_is_byte_identical(self) -> None:
        assert _canonical_json(_build()) == _canonical_json(_build())

    def test_upstream_file_order_does_not_affect_output(self) -> None:
        first = _build()
        with patch(
            "onnxvoice.catalog_tools.pocket._repository_tree",
            return_value=list(reversed(_tree(BASE_PATHS))),
        ):
            second = build_prompt_catalog(resolved_revision=REVISION)
        assert _canonical_json(first) == _canonical_json(second)
        assert [record["id"] for record in first["prompts"]] == [
            "alba-mackenna/casual",
            "vctk/p222_001",
            "voice-donations/lois",
        ]

    def test_new_wav_adds_one_deterministic_record(self) -> None:
        before = _build()
        after = _build([*BASE_PATHS, "ears/p050/freeform_speech_01.wav"])
        assert after["source"]["prompt_count"] == before["source"]["prompt_count"] + 1
        added = [record for record in after["prompts"] if record not in before["prompts"]]
        assert [record["id"] for record in added] == ["ears/p050/freeform_speech_01"]
        assert [record for record in after["prompts"] if record not in added] == before["prompts"]

    def test_removed_wav_deterministically_removes_one_record(self) -> None:
        before = _build()
        after = _build(BASE_PATHS[:-1])
        removed = [record for record in before["prompts"] if record not in after["prompts"]]
        assert [record["id"] for record in removed] == ["vctk/p222_001"]
        assert after["source"]["prompt_count"] == before["source"]["prompt_count"] - 1

    def test_ignores_non_wav_files(self) -> None:
        catalog = _build(
            [
                "alba-mackenna/casual.wav",
                "alba-mackenna/casual.safetensors",
                "voice-zero/embeddings.safetensors",
                "README.md",
            ]
        )
        assert [record["id"] for record in catalog["prompts"]] == ["alba-mackenna/casual"]

    def test_classifies_enhanced_variants(self) -> None:
        catalog = _build(["expresso/en/ex04.wav", "expresso/en/ex04_enhanced.wav"])
        records = {record["id"]: record for record in catalog["prompts"]}
        assert records["expresso/en/ex04"]["variant"] == "original"
        assert records["expresso/en/ex04"]["base_id"] is None
        assert records["expresso/en/ex04_enhanced"]["variant"] == "enhanced"
        assert records["expresso/en/ex04_enhanced"]["base_id"] == "expresso/en/ex04"

    def test_records_pin_revision_url_and_evidence(self) -> None:
        record = _build(["alba-mackenna/casual.wav"])["prompts"][0]
        assert record["source_revision"] == REVISION
        assert record["url"] == pinned_source_url(REPOSITORY, REVISION, "alba-mackenna/casual.wav")
        assert record["license_evidence"] == (
            f"https://huggingface.co/{REPOSITORY}/blob/{REVISION}/README.md"
        )

    @pytest.mark.parametrize(
        "path",
        ["cml-tts/de/echo.wav", "unknown-group/voice.wav", "degaulle-2.wav"],
    )
    def test_license_mapping_omission_fails_loudly(self, path: str) -> None:
        with pytest.raises(CatalogError, match="no checked-in license rule"):
            _build([path])

    def test_runtime_parser_accepts_built_catalog(self) -> None:
        prompts = parse_prompt_catalog(_build())
        assert [prompt.id for prompt in prompts] == [
            "alba-mackenna/casual",
            "vctk/p222_001",
            "voice-donations/lois",
        ]


class TestLicenseRules:
    @pytest.mark.parametrize(
        ("path", "license_name", "has_note"),
        [
            ("alba-mackenna/casual.wav", "CC-BY-4.0", False),
            ("cml-tts/fr/bonjour.wav", "CC-BY-4.0", False),
            ("vctk/p222_001.wav", "CC-BY-4.0", False),
            ("voice-donations/lois.wav", "CC0-1.0", False),
            ("voice-zero/zero.wav", "CC0-1.0", False),
            ("ears/p050/freeform_speech_01.wav", "CC-BY-NC-4.0", True),
            ("expresso/en/ex04.wav", "CC-BY-NC-4.0", True),
            ("unmute-prod-website/other.wav", "CC0-1.0", True),
            ("unmute-prod-website/ex04_narration_longform_00001.wav", "CC-BY-NC-4.0", True),
            ("unmute-prod-website/p329_022.wav", "CC-BY-4.0", True),
            ("unmute-prod-website/degaulle-2.wav", "NOASSERTION", True),
        ],
    )
    def test_checked_in_license_decisions(
        self, path: str, license_name: str, has_note: bool
    ) -> None:
        resolved_license, note = prompt_license(path)
        assert resolved_license == license_name
        assert (note is not None) is has_note


class TestVerifyPromptCatalog:
    def test_accepts_built_catalog(self) -> None:
        verify_prompt_catalog(_build())

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("id", "alba-mackenna/other"),
            ("variant", "enhanced"),
            ("base_id", "alba-mackenna/casual"),
            ("format", "mp3"),
            ("source_path", "vctk/p222_001.wav"),
            ("url", "https://example.com/casual.wav"),
            ("sha256", "not-hex"),
            ("license", "CC-BY-NC-4.0"),
            ("license_note", "forged note"),
            ("license_evidence", "https://example.com/README.md"),
            ("metadata", "not-an-object"),
        ],
    )
    def test_rejects_tampered_prompt_records(self, field: str, value: Any) -> None:
        catalog = _build()
        catalog["prompts"][0][field] = value
        with pytest.raises(CatalogError):
            verify_prompt_catalog(catalog)

    def test_rejects_unsorted_prompts(self) -> None:
        catalog = _build()
        catalog["prompts"] = list(reversed(catalog["prompts"]))
        with pytest.raises(CatalogError, match="sorted"):
            verify_prompt_catalog(catalog)

    def test_rejects_prompt_count_mismatch(self) -> None:
        catalog = _build()
        catalog["source"]["prompt_count"] = 2
        with pytest.raises(CatalogError, match="Prompt count mismatch"):
            verify_prompt_catalog(catalog)

    @pytest.mark.parametrize(
        "mutate",
        [
            lambda catalog: catalog.update(schema=2),
            lambda catalog: catalog.update(kind="other"),
            lambda catalog: catalog["source"].update(revision="main"),
            lambda catalog: catalog["prompts"][0].update(extra="field"),
        ],
        ids=["schema", "kind", "revision", "extra-field"],
    )
    def test_rejects_unsupported_shapes(self, mutate: Callable[[dict[str, Any]], None]) -> None:
        catalog = _build()
        mutate(catalog)
        with pytest.raises(CatalogError):
            verify_prompt_catalog(catalog)


class TestPromptCatalogFiles:
    def test_load_prompt_catalog_roundtrip(self, tmp_path: Path) -> None:
        path = tmp_path / "voice-prompts.json"
        path.write_text(json.dumps(_build()), encoding="utf-8")
        assert load_prompt_catalog(path)["source"]["revision"] == REVISION

    def test_load_prompt_catalog_rejects_tampering(self, tmp_path: Path) -> None:
        tampered = copy.deepcopy(_build())
        tampered["prompts"][0]["license"] = "CC-BY-NC-4.0"
        path = tmp_path / "voice-prompts.json"
        path.write_text(json.dumps(tampered), encoding="utf-8")
        with pytest.raises(CatalogError, match="license"):
            load_prompt_catalog(path)

    def test_cli_build_and_verify(self, tmp_path: Path) -> None:
        from onnxvoice import cli

        output = tmp_path / "voice-prompts.json"
        source_output = tmp_path / "voice-prompts-source.json"
        with (
            patch("onnxvoice.catalog_tools.pocket.resolve_revision", return_value=REVISION),
            patch(
                "onnxvoice.catalog_tools.pocket._repository_tree", return_value=_tree(BASE_PATHS)
            ),
        ):
            result = cli.main(
                [
                    "catalog",
                    "pocket",
                    "prompts",
                    "build",
                    "--repository",
                    REPOSITORY,
                    "--revision",
                    "main",
                    "--output",
                    str(output),
                    "--source-output",
                    str(source_output),
                ]
            )
        assert result == 0
        catalog = json.loads(output.read_text(encoding="utf-8"))
        assert catalog["kind"] == PROMPT_CATALOG_KIND
        source = json.loads(source_output.read_text(encoding="utf-8"))
        assert source == {"schema": 1, **catalog["source"]}
        assert (
            cli.main(
                [
                    "catalog",
                    "pocket",
                    "prompts",
                    "verify",
                    "--catalog",
                    str(output),
                    "--source",
                    str(source_output),
                ]
            )
            == 0
        )
