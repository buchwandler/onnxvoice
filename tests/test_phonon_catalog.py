from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest

from onnxvoice.catalog import CatalogClient, _parse_phonon
from onnxvoice.errors import AssetNotFoundError, CatalogError

GRAPHS = {
    "onnx/text_conditioner.onnx": "text_conditioner",
    "onnx/flow_lm_main.onnx": "flow_lm_main",
    "onnx/flow_lm_flow.onnx": "flow_lm_flow",
    "onnx/mimi_decoder.onnx": "mimi_decoder",
}


def _license(model_key: str) -> dict[str, Any]:
    if model_key == "phonon-31m-en":
        return {
            "spdx": "Apache-2.0",
            "uri": "https://www.apache.org/licenses/LICENSE-2.0",
            "commercial_use": True,
            "restrictions": [],
            "attribution_path": "ATTRIBUTION.md",
        }
    return {
        "spdx": "CC-BY-NC-4.0",
        "uri": "https://creativecommons.org/licenses/by-nc/4.0/",
        "legalcode_uri": "https://creativecommons.org/licenses/by-nc/4.0/legalcode.en",
        "warranty_disclaimer_section": "Section 5",
        "commercial_use": False,
        "restrictions": ["non-commercial"],
        "attribution_path": "ATTRIBUTION.md",
    }


def _bundle(
    model_key: str = "phonon-31m-en",
    precision: str = "fp32",
    *,
    tag: str | None = None,
    revision: str | None = None,
) -> dict[str, Any]:
    tag = tag or f"{model_key}-onnx-{precision}-files-v1"
    revision = revision or ("a" if precision == "fp32" else "b") * 40
    paths = [
        "config.json",
        "tokenizer.json",
        *GRAPHS,
        "voices/easton.bin",
        "ATTRIBUTION.md",
        "export-provenance.json",
        "third_party/model/README.md",
    ]
    if model_key == "phonon-40m-en":
        paths.append("third_party/voices/README.md")
    artifacts = [
        {
            "path": path,
            "bytes": 100 + index,
            "sha256": f"{index + 1:064x}",
            "download_url": f"https://github.com/example/{tag}/download/{path.replace('/', '--')}",
        }
        for index, path in enumerate(paths)
    ]
    return {
        "tag": tag,
        "model_key": model_key,
        "precision": precision,
        "source_revision": revision,
        "source_uri": f"https://huggingface.co/gradium/{model_key}/tree/{revision}",
        "runtime": {
            "kind": "phonon-streaming-onnx-v1",
            "graphs": list(GRAPHS),
            "sample_rate": 24000,
            "frame_samples": 1920,
            "frame_rate": 12.5,
            "d_model": 512,
            "latent_dim": 32,
            "num_layers": 6,
            "num_heads": 8,
            "head_dim": 64,
            "mimi_state": [{"name": "cache", "shape": [1, 2], "dtype": "float32"}],
        },
        "generation": {
            "temperature_default": 0.8,
            "eos_threshold_default": 4.0,
            "duration_delta_default": 0.0,
            "frames_after_eos": 5,
            "min_frames_before_eos": 2,
            "max_frames": 500,
        },
        "license": _license(model_key),
        "voices": [
            {
                "name": "easton",
                "display_name": "Easton",
                "voice_id": "LMDIYE0Oa377Qh6Z",
                "locale": "en-US",
                "gender": "male",
                "frames": 125,
            }
        ],
        "default_voice": "easton",
        "languages": ["en"],
        "artifacts": artifacts,
    }


def _catalog(*bundles: dict[str, Any]) -> dict[str, Any]:
    return {"schema_version": 2, "bundles": list(bundles)}


def test_current_partial_catalog_lists_only_published_target_qualities():
    items = _parse_phonon(
        _catalog(_bundle("phonon-31m-en"), _bundle("phonon-40m-en", revision="c" * 40))
    )
    assert [(item.id, item.system, item.kind) for item in items] == [
        ("phonon-31m-en", "phonon", "bundle"),
        ("phonon-40m-en", "phonon", "bundle"),
    ]
    assert [item.metadata["quality_profiles"] for item in items] == [["fp32"], ["fp32"]]
    assert items[0].metadata["commercial_use"] is True
    assert items[1].metadata["commercial_use"] is False
    assert items[1].metadata["license"]["spdx"] == "CC-BY-NC-4.0"


def test_new_precision_aggregates_on_existing_target_and_resolves_only_its_assets(tmp_path: Path):
    fp32 = _bundle("phonon-31m-en", "fp32")
    q4 = _bundle("phonon-31m-en", "q4", tag="phonon-31m-en-onnx-q4-files-v1", revision="b" * 40)
    fp16 = _bundle(
        "phonon-31m-en", "fp16", tag="phonon-31m-en-onnx-fp16-files-v1", revision="c" * 40
    )
    (item,) = _parse_phonon(_catalog(fp32, q4, fp16))
    assert item.id == "phonon-31m-en"
    assert item.metadata["quality_profiles"] == ["fp16", "fp32", "q4"]

    catalog_path = tmp_path / "phonon.json"
    catalog_path.write_text(json.dumps(_catalog(fp32, q4, fp16)), encoding="utf-8")
    client = CatalogClient(cache_dir=tmp_path / "cache", sources={"phonon": str(catalog_path)})
    selected = client.resolve("phonon:phonon-31m-en", quality="q4")
    default = client.resolve("phonon:phonon-31m-en")
    assert default.metadata["selected_quality"] == "fp32"
    assert item.metadata["source_revision"] == "a" * 40
    assert selected.metadata["selected_quality"] == "q4"
    assert selected.metadata["release_tag"] == "phonon-31m-en-onnx-q4-files-v1"
    assert {artifact.quality for artifact in selected.artifacts} == {"q4"}
    assert (
        selected.require_artifact("model", component="flow_lm_main").filename
        == "onnx/flow_lm_main.onnx"
    )
    with pytest.raises(AssetNotFoundError, match="available: fp16, fp32, q4"):
        client.resolve("phonon:phonon-31m-en", quality="q8")


def test_artifact_roles_paths_voice_details_and_default_voice_are_normalized():
    (item,) = _parse_phonon(_catalog(_bundle()))
    by_path = {artifact.filename: artifact for artifact in item.artifacts}
    assert by_path["config.json"].role == "config"
    assert by_path["tokenizer.json"].role == "tokenizer"
    assert by_path["onnx/text_conditioner.onnx"].role == "model"
    assert by_path["onnx/text_conditioner.onnx"].component == "text_conditioner"
    assert by_path["voices/easton.bin"].role == "voice_prompt"
    assert by_path["voices/easton.bin"].component == "easton"
    assert by_path["ATTRIBUTION.md"].role == "attribution"
    assert by_path["export-provenance.json"].role == "provenance"
    assert by_path["third_party/model/README.md"].role == "notice"
    assert item.voices == ("easton",)
    assert item.default_voice == "easton"
    assert item.metadata["voice_details"] == [
        {
            "id": "easton",
            "name": "Easton",
            "display_name": "Easton",
            "language": "en",
            "locale": "en-US",
            "language_label": "en-US",
            "gender": "male",
            "voice_id": "LMDIYE0Oa377Qh6Z",
            "upstream_voice_id": "LMDIYE0Oa377Qh6Z",
            "frames": 125,
        }
    ]


def test_descriptive_voice_locale_is_preserved_without_inventing_language():
    bundle = _bundle()
    bundle["voices"].append(
        {
            "name": "character",
            "display_name": "Character",
            "locale": "storybook voice",
            "gender": "other",
        }
    )
    bundle["artifacts"].append(
        {
            "path": "voices/character.bin",
            "bytes": 20,
            "sha256": "f" * 64,
            "download_url": "https://github.com/example/character.bin",
        }
    )
    (item,) = _parse_phonon(_catalog(bundle))
    detail = next(value for value in item.metadata["voice_details"] if value["id"] == "character")
    assert detail["source_locale"] == "storybook voice"
    assert detail["locale"] == ""
    assert detail["language"] == ""
    assert detail["gender"] == "unknown"


def test_future_catalog_fields_are_ignored_and_partial_availability_is_valid():
    bundle = _bundle()
    bundle["future_runtime_hint"] = {"unrecognized": True}
    data = _catalog(bundle)
    data["future_schema_note"] = "ignored"
    items = _parse_phonon(data)
    assert len(items) == 1
    assert items[0].metadata["quality_profiles"] == ["fp32"]


def test_missing_unpublished_qualities_and_models_do_not_fail_discovery():
    items = _parse_phonon(_catalog(_bundle("phonon-31m-en")))
    assert len(items) == 1
    assert items[0].metadata["quality_profiles"] == ["fp32"]


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda row: row.update(precision="q16"), "unsupported precision"),
        (lambda row: row.update(source_revision="main"), "immutable 40-character SHA"),
        (
            lambda row: row.update(
                source_uri="https://huggingface.co/gradium/phonon-31m-en/tree/main"
            ),
            "source_uri must pin source_revision",
        ),
        (lambda row: row["runtime"].update(kind="pocket"), "unsupported runtime.kind"),
        (
            lambda row: row["runtime"].update(sample_rate=0),
            "sample_rate must be a positive integer",
        ),
        (lambda row: row["generation"].pop("frames_after_eos"), "must be explicit"),
        (lambda row: row["license"].pop("spdx"), "unsupported license.spdx"),
        (lambda row: row.update(default_voice="missing"), "default_voice must name"),
        (lambda row: row["artifacts"].pop(2), "missing required artifacts"),
        (lambda row: row["artifacts"][0].update(sha256="bad"), "sha256 must be 64"),
        (lambda row: row["artifacts"][0].update(bytes=0), "bytes must be a positive integer"),
        (
            lambda row: row["artifacts"][0].update(path="../escape"),
            "Unsafe phonon-31m-en/fp32 artifact path",
        ),
        (
            lambda row: row["artifacts"][0].update(download_url="http://example.com/file"),
            "HTTPS URL",
        ),
        (
            lambda row: row["artifacts"].append(copy.deepcopy(row["artifacts"][0])),
            "duplicate artifact path",
        ),
    ],
)
def test_catalog_rejects_invalid_runtime_contract(change, message):
    bundle = _bundle()
    change(bundle)
    with pytest.raises(CatalogError, match=message):
        _parse_phonon(_catalog(bundle))


def test_duplicate_model_precision_pair_is_rejected():
    with pytest.raises(CatalogError, match="duplicate published Phonon variant"):
        _parse_phonon(_catalog(_bundle(), _bundle()))


def test_duplicate_quality_rows_with_inconsistent_voice_or_license_metadata_fail():
    q4 = _bundle("phonon-31m-en", "q4", tag="phonon-31m-en-q4", revision="b" * 40)
    q4["voices"][0]["display_name"] = "Different"
    with pytest.raises(CatalogError, match="metadata is inconsistent"):
        _parse_phonon(_catalog(_bundle(), q4))


def test_catalog_schema_version_and_bundle_shape_are_checked():
    with pytest.raises(CatalogError, match="schema_version must be 2"):
        _parse_phonon({"schema_version": 1, "bundles": []})
    with pytest.raises(CatalogError, match="bundles must be a sequence"):
        _parse_phonon({"schema_version": 2, "bundles": {}})


def test_noncommercial_license_provenance_and_required_upstream_notices_are_preserved():
    bundle = _bundle("phonon-40m-en")
    (item,) = _parse_phonon(_catalog(bundle))
    license_info = item.metadata["license"]
    assert license_info["spdx"] == "CC-BY-NC-4.0"
    assert license_info["commercial_use"] is False
    assert license_info["restrictions"] == ["non-commercial"]
    assert license_info["legalcode_uri"].endswith("legalcode.en")
    assert license_info["warranty_disclaimer_section"] == "Section 5"

    bundle["artifacts"] = [
        artifact
        for artifact in bundle["artifacts"]
        if artifact["path"] != "third_party/voices/README.md"
    ]
    with pytest.raises(CatalogError, match="third_party/voices/README.md"):
        _parse_phonon(_catalog(bundle))


def test_noncommercial_license_cannot_be_mislabeled_as_commercial():
    bundle = _bundle("phonon-40m-en")
    bundle["license"]["commercial_use"] = True
    with pytest.raises(CatalogError, match="commercial-use status conflicts"):
        _parse_phonon(_catalog(bundle))
