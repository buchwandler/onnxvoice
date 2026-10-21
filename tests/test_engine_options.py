from __future__ import annotations

import builtins
import json
from collections.abc import Sequence
from dataclasses import FrozenInstanceError, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from onnxvoice.engine_options import (
    EngineOptionsError,
    contract_coverage,
    get_engine_contract,
    list_engine_targets,
    resolve_engine_options,
)
from onnxvoice.errors import OfflineError
from onnxvoice.systems import registered_systems
from onnxvoice.types import CatalogItem


def _item(
    engine: str,
    target: str,
    metadata: dict[str, Any] | None = None,
    *,
    aliases: tuple[str, ...] = (),
    sample_rate: int | None = 24_000,
) -> CatalogItem:
    return CatalogItem(
        system=engine,
        id=target,
        kind="bundle",
        artifacts=(),
        aliases=aliases,
        sample_rate=sample_rate,
        metadata=metadata or {},
    )


class FakeCatalog:
    def __init__(self, items: builtins.list[CatalogItem]) -> None:
        self.items = items
        self.resolve_calls: list[tuple[str, str | None, str | None]] = []

    def list(self, engine: str, refresh: bool = False) -> Sequence[CatalogItem]:
        del refresh
        return [item for item in self.items if item.system == engine]

    def resolve(
        self,
        ref: str,
        *,
        refresh: bool = False,
        quality: str | None = None,
        distribution: str | None = None,
    ) -> CatalogItem:
        del refresh
        self.resolve_calls.append((ref, quality, distribution))
        engine, target = ref.split(":", 1)
        item = next(item for item in self.list(engine) if item.id == target)
        metadata = dict(item.metadata)
        if quality is not None:
            profiles = metadata.get("profiles", {})
            if isinstance(profiles, dict) and profiles and quality not in profiles:
                raise ValueError(
                    f"Unknown quality {quality!r}; valid choices: {', '.join(sorted(profiles))}"
                )
            metadata["selected_quality"] = quality
        if distribution is not None:
            metadata["distribution_id"] = distribution
            runtime = dict(metadata.get("runtime", {}))
            if engine == "kokoro":
                runtime["layout"] = {
                    "clone": "cloning-onnx-v1",
                    "split": "split-onnx-v1",
                    "single": "single-onnx-v1",
                }[distribution]
            metadata["runtime"] = runtime
        return replace(item, metadata=metadata)


class FakeManager:
    def __init__(
        self,
        items: list[CatalogItem],
        installations: list[Any] | None = None,
    ) -> None:
        self.catalog = FakeCatalog(items)
        self.installations = installations or []

    def installed(self, system: str | None = None) -> list[Any]:
        return [item for item in self.installations if system is None or item.system == system]


def _fixture_manager() -> FakeManager:
    return FakeManager(
        [
            _item(
                "kokoro",
                "v1.0",
                {
                    "distribution_id": "split",
                    "distribution_choices": ["split", "clone"],
                    "runtime": {"layout": "split-onnx-v1", "max_tokens": 510},
                },
                aliases=("kokoro-latest",),
            ),
            _item(
                "qwen3",
                "base-0.6b",
                {
                    "model_type": "base",
                    "profiles": {"fp32": {}, "int4": {}},
                    "runtime": {"model_type": "base", "layout": "qwen3-wavekat-v1"},
                },
            ),
            _item(
                "qwen3",
                "voice-design-1.7b",
                {
                    "model_type": "voice_design",
                    "profiles": {"fp32": {}, "int4": {}},
                    "runtime": {
                        "model_type": "voice_design",
                        "layout": "qwen3-wavekat-v1",
                    },
                },
            ),
            _item("pocket", "clone-yes", {"supports_voice_cloning": True, "max_frames": 128}),
            _item("pocket", "clone-no", {"supports_voice_cloning": False}),
            _item("pocket", "clone-unknown", {}),
        ]
    )


def test_contract_registry_covers_every_builtin_and_third_party_is_explicit() -> None:
    coverage = contract_coverage()
    assert set(coverage["registered"]) == set(registered_systems())
    assert coverage["missing"] == []
    assert coverage["extra"] == []
    assert get_engine_contract("third-party-engine").contract_status == "unavailable"
    assert "No engine-options contract is registered" in (
        get_engine_contract("third-party-engine").contract_reason or ""
    )


def test_static_descriptors_are_immutable_and_json_serializable() -> None:
    contract = get_engine_contract("piper")
    payload = contract.to_dict()
    assert json.dumps(payload, sort_keys=True) == json.dumps(contract.to_dict(), sort_keys=True)
    attribute_name = "default"
    with pytest.raises(FrozenInstanceError):
        setattr(contract.parameters[0], attribute_name, 99)

    piper_parameters = {entry.name: entry for entry in contract.parameters}
    assert piper_parameters["speaker_id"].type == "integer"
    assert "not reference-audio cloning" in piper_parameters["speaker_id"].description
    unknown_text_limit = next(
        limit for limit in contract.limits if limit.name == "max_input_tokens"
    )
    assert unknown_text_limit.value is None
    assert unknown_text_limit.status == "not_declared"


def test_kokoro_distribution_variants_keep_cloning_boundary_semantics() -> None:
    manager = _fixture_manager()
    data = resolve_engine_options(manager, "kokoro", target="kokoro-latest")
    assert data["target"] == "v1.0"
    variants = {variant["distribution"]: variant for variant in data["variants"]}
    assert set(variants) == {"clone", "split"}

    cloning = variants["clone"]
    assert cloning["layout"] == "cloning-onnx-v1"
    assert cloning["capabilities"]["voice_cloning"]["supported"] is True
    assert cloning["capabilities"]["voice_cloning"]["requires_reference_transcript"] is True
    assert cloning["capabilities"]["voice_cloning"]["requires"] == [
        "reference_audio",
        "reference_transcript_token_ids",
        "reference",
    ]
    parameters = {entry["name"]: entry for entry in cloning["parameters"]}
    assert parameters["speed"]["supported"] is False
    assert parameters["speed"]["status"] == "unsupported_except_value"
    assert parameters["speed"]["choices"] == [1.0]
    limits = {entry["name"]: entry for entry in cloning["limits"]}
    assert limits["max_input_tokens"]["value"] == 510
    assert limits["max_content_tokens"]["value"] == 508

    split = variants["split"]
    assert split["capabilities"]["voice_cloning"]["supported"] is False
    assert (
        next(limit for limit in split["limits"] if limit["name"] == "max_content_tokens")["status"]
        == "not_applicable"
    )

    rows = list_engine_targets(manager, engine="kokoro", capability="voice-cloning")["rows"]
    assert [(row["target"], row["distribution"]) for row in rows] == [("v1.0", "clone")]
    selected = resolve_engine_options(manager, "kokoro", target="v1.0", distribution="clone")
    assert selected["distribution"] == "clone"
    with pytest.raises(EngineOptionsError, match="valid choices: clone, split"):
        resolve_engine_options(manager, "kokoro", target="v1.0", distribution="invalid")


def test_qwen3_base_voice_design_and_profile_selection_are_distinct() -> None:
    manager = _fixture_manager()
    base = resolve_engine_options(manager, "qwen3", target="base-0.6b", quality="int4")
    design = resolve_engine_options(manager, "qwen3", target="voice-design-1.7b")
    assert base["quality"] == "int4"
    assert base["capabilities"]["voice_cloning"]["supported"] is True
    assert base["capabilities"]["voice_cloning"]["requires_reference_transcript"] is True
    assert design["capabilities"]["voice_cloning"]["supported"] is False
    assert design["capabilities"]["voice_design"]["supported"] is True
    assert design["capabilities"]["voice_design"]["mode"] == "instruction_tokens"
    assert next(option for option in design["inputs"] if option["name"] == "instruction_token_ids")[
        "required"
    ]
    assert [
        row["target"]
        for row in list_engine_targets(manager, engine="qwen3", capability="voice-cloning")["rows"]
    ] == ["base-0.6b"]


def test_pocket_uses_declared_clone_capability_and_frame_units() -> None:
    manager = _fixture_manager()
    supported = resolve_engine_options(manager, "pocket", target="clone-yes")
    unsupported = resolve_engine_options(manager, "pocket", target="clone-no")
    unknown = resolve_engine_options(manager, "pocket", target="clone-unknown")
    limits = {entry["name"]: entry for entry in supported["limits"]}
    assert supported["capabilities"]["voice_cloning"]["supported"] is True
    assert unsupported["capabilities"]["voice_cloning"]["supported"] is False
    assert unknown["capabilities"]["voice_cloning"]["supported"] is None
    assert limits["max_frames"]["value"] == 128
    assert limits["max_frames"]["unit"] == "frames"
    assert limits["max_input_tokens"]["value"] is None
    assert limits["max_input_tokens"]["status"] == "not_declared"
    assert [
        row["target"]
        for row in list_engine_targets(manager, engine="pocket", capability="voice-cloning")["rows"]
    ] == ["clone-yes"]


def test_sopro_merges_catalog_facts_with_installed_manifest(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "model": {"max_text_len": 256},
                "generation": {
                    "max_seconds": 12,
                    "prompt_tokens": 40,
                    "ref_seconds": 8,
                    "top_k": 25,
                },
            }
        ),
        encoding="utf-8",
    )
    item = _item(
        "sopro",
        "v2-turbo",
        {
            "runtime": {"layout": "sopro-v1"},
            "model": {"max_text_len": 200},
            "generation": {"max_seconds": 15, "top_p": 0.7},
        },
    )
    installation = SimpleNamespace(
        id="v2-turbo",
        system="sopro",
        metadata={},
        sample_rate=24_000,
        artifacts=[SimpleNamespace(role="manifest", path=manifest_path)],
    )
    data = resolve_engine_options(FakeManager([item], [installation]), "sopro", target="v2-turbo")
    limits = {entry["name"]: entry for entry in data["limits"]}
    parameters = {entry["name"]: entry for entry in data["parameters"]}
    assert limits["max_text_len"]["value"] == 200
    assert limits["max_text_len"]["source"] == "catalog_metadata"
    assert limits["max_seconds"]["value"] == 15
    assert limits["max_seconds"]["source"] == "catalog_metadata"
    assert limits["max_prompt_tokens"]["value"] == 40
    assert parameters["top_p"]["default"] == 0.7
    assert parameters["top_p"]["source"] == "catalog_metadata"
    assert parameters["top_k"]["default"] == 25
    assert parameters["top_k"]["source"] == "installed_manifest"
    assert parameters["steps"]["default_status"] == "unknown"
    assert "wasm-uint8" in data["capabilities"]["voice_cloning"]["constraints"][0]
    assert "CPUExecutionProvider" in data["capabilities"]["voice_cloning"]["constraints"][0]


def test_chatterbox_generation_and_kitten2_named_selections_are_target_specific() -> None:
    manager = FakeManager(
        [
            _item(
                "chatterbox",
                "nano-v2",
                {
                    "runtime": {
                        "layout": "chatterbox-multi-v1",
                        "default_max_new_tokens": 1024,
                        "default_repetition_penalty": 1.2,
                    },
                    "generation": {"max_new_tokens": 512, "repetition_penalty": 1.15},
                },
            ),
            _item(
                "kitten2",
                "kitten-tts-v2",
                {
                    "runtime": {"layout": "kitten2-onnx-bundle-v1"},
                    "voices": {"Bella": {}, "Bruno": {}},
                    "default_voice": "Bella",
                    "decode_presets": {"stable": {}, "expressive": {}},
                    "default_preset": "stable",
                },
            ),
        ]
    )
    chatterbox = resolve_engine_options(manager, "chatterbox", target="nano-v2")
    controls = {entry["name"]: entry for entry in chatterbox["parameters"]}
    assert chatterbox["capabilities"]["voice_cloning"]["supported"] is True
    assert controls["max_new_tokens"]["default"] == 512
    assert controls["max_new_tokens"]["source"] == "catalog_metadata"
    assert controls["repetition_penalty"]["default"] == 1.15

    kitten2 = resolve_engine_options(manager, "kitten2", target="kitten-tts-v2")
    selection = {entry["name"]: entry for entry in kitten2["selection_metadata"]}
    controls = {entry["name"]: entry for entry in kitten2["parameters"]}
    assert selection["voices"]["value"] == ["Bella", "Bruno"]
    assert selection["default_voice"]["value"] == "Bella"
    assert selection["presets"]["value"] == ["expressive", "stable"]
    assert controls["voice"]["default"] == "Bella"
    assert controls["preset"]["default"] == "stable"


def test_offline_discovery_and_unknown_limits_are_explicit() -> None:
    class OfflineCatalog:
        def list(self, engine: str, refresh: bool = False) -> Sequence[CatalogItem]:
            del engine, refresh
            raise OfflineError("catalog unavailable offline")

    class OfflineManager:
        catalog = OfflineCatalog()

        def installed(self, system: str | None = None) -> list[Any]:
            del system
            return []

    data = resolve_engine_options(OfflineManager(), "piper")
    assert data["contract_status"] == "available"
    assert data["target_discovery"] == {
        "status": "unavailable",
        "reason": "catalog unavailable offline",
    }
    assert data["known_targets"] == []
    rows = list_engine_targets(OfflineManager(), engine="piper", capability="voice-cloning")
    assert rows["target_discovery"] == "partial"
    assert rows["rows"] == []
    limit = next(limit for limit in data["limits"] if limit["name"] == "max_input_tokens")
    assert limit["value"] is None
    assert limit["status"] == "not_declared"


def test_resolved_json_is_deterministic_and_copies_static_data() -> None:
    manager = _fixture_manager()
    first = resolve_engine_options(manager, "kokoro", target="v1.0", distribution="clone")
    second = resolve_engine_options(manager, "kokoro", target="v1.0", distribution="clone")
    assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)
    first["parameters"].clear()
    third = resolve_engine_options(manager, "kokoro", target="v1.0", distribution="clone")
    assert third["parameters"]
