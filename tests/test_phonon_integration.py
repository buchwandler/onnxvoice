from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import onnxvoice
from onnxvoice import cli
from onnxvoice.engine_options import contract_coverage, resolve_engine_options
from onnxvoice.inventory import query_inventory
from onnxvoice.manager import OnnxVoice
from onnxvoice.systems import get_adapter, registered_systems
from onnxvoice.systems.phonon import PhononAdapter, PhononStreamChunk, PhononVoiceState
from onnxvoice.types import CatalogItem

_GRAPH_PATHS = {
    "text_conditioner": "onnx/text_conditioner.onnx",
    "flow_lm_main": "onnx/flow_lm_main.onnx",
    "flow_lm_flow": "onnx/flow_lm_flow.onnx",
    "mimi_decoder": "onnx/mimi_decoder.onnx",
}
_RUNTIME = {
    "kind": "phonon-streaming-onnx-v1",
    "graphs": list(_GRAPH_PATHS.values()),
    "sample_rate": 24000,
    "frame_samples": 1920,
    "frame_rate": 12.5,
    "d_model": 4,
    "latent_dim": 3,
    "num_layers": 2,
    "num_heads": 2,
    "head_dim": 2,
    "mimi_state": [{"name": "cache", "shape": [1, 2], "dtype": "float32"}],
}
_GENERATION = {
    "duration_delta_default": 0.0,
    "temperature_default": 0.8,
    "eos_threshold_default": 4.0,
    "frames_after_eos": 5,
    "min_frames_before_eos": 2,
    "max_frames": 500,
}
_APACHE = {
    "spdx": "Apache-2.0",
    "uri": "https://www.apache.org/licenses/LICENSE-2.0",
    "commercial_use": True,
    "restrictions": [],
    "attribution_path": "ATTRIBUTION.md",
}
_CC_BY_NC = {
    "spdx": "CC-BY-NC-4.0",
    "uri": "https://creativecommons.org/licenses/by-nc/4.0/",
    "legalcode_uri": "https://creativecommons.org/licenses/by-nc/4.0/legalcode.en",
    "warranty_disclaimer_section": "Section 5",
    "commercial_use": False,
    "restrictions": ["non-commercial"],
    "attribution_path": "ATTRIBUTION.md",
}


class _Catalog:
    def __init__(self, items: Sequence[CatalogItem]) -> None:
        self.items = list(items)

    def list(self, system: str, refresh: bool = False) -> Sequence[CatalogItem]:
        del refresh
        return [item for item in self.items if item.system == system]

    def resolve(
        self,
        ref: str,
        *,
        refresh: bool = False,
        quality: str | None = None,
        distribution: str | None = None,
    ) -> CatalogItem:
        del refresh, distribution
        system, target = ref.split(":", 1)
        item = next(row for row in self.list(system) if row.id == target)
        profiles = item.metadata["profiles"]
        selected = quality or "fp32"
        profile = profiles[selected]
        return replace(
            item,
            metadata={
                **item.metadata,
                **profile,
                "selected_quality": selected,
                "quality": selected,
            },
        )


class _Manager:
    def __init__(self, items: Sequence[CatalogItem]) -> None:
        self.catalog = _Catalog(items)

    def installed(self, system: str | None = None) -> list[Any]:
        del system
        return []


def _catalog_item() -> CatalogItem:
    profile = {"generation": dict(_GENERATION), "license": dict(_APACHE)}
    metadata = {
        "runtime": dict(_RUNTIME),
        "generation": dict(_GENERATION),
        "profiles": {"fp32": profile},
        "quality_profiles": ["fp32"],
        "license": dict(_APACHE),
        "license_spdx": "Apache-2.0",
        "commercial_use": True,
        "restrictions": [],
        "languages": ["en"],
        "language_codes": ["en"],
        "voices": ["easton"],
        "voice_details": [{"id": "easton", "name": "Easton", "locale": "en-US", "language": "en"}],
        "default_voice": "easton",
        "voice_cloning": False,
        "streaming": True,
        "named_voices": True,
    }
    return CatalogItem(
        system="phonon",
        id="phonon-31m-en",
        kind="bundle",
        artifacts=(),
        sample_rate=24000,
        voices=("easton",),
        default_voice="easton",
        metadata=metadata,
    )


def test_adapter_and_public_types_are_registered_and_exported() -> None:
    assert get_adapter("phonon") is PhononAdapter
    assert "phonon" in registered_systems()
    assert contract_coverage()["missing"] == []
    for name, value in (
        ("PhononAdapter", PhononAdapter),
        ("PhononVoiceState", PhononVoiceState),
        ("PhononStreamChunk", PhononStreamChunk),
    ):
        assert name in onnxvoice.__all__
        assert getattr(onnxvoice, name) is value
        assert getattr(onnxvoice.systems, name) is value


def test_generic_local_open_reads_config_and_prompts_without_store_or_network(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "local-phonon"
    config_path = root / "config.json"
    config_path.parent.mkdir(parents=True)
    config_path.write_text(
        json.dumps(
            {
                "runtime": _RUNTIME,
                "generation": _GENERATION,
                "voices": [{"name": "easton", "locale": "en-US", "frames": 2}],
                "default_voice": "easton",
            }
        ),
        encoding="utf-8",
    )
    artifacts: dict[str, Path] = {"config": config_path}
    for component, relative in _GRAPH_PATHS.items():
        graph = root / relative
        graph.parent.mkdir(parents=True, exist_ok=True)
        graph.write_bytes(b"fake-onnx")
        artifacts[f"model:{component}"] = graph
    prompt = root / "voices" / "easton.bin"
    prompt.parent.mkdir(parents=True)
    np.arange(8, dtype="<f4").tofile(prompt)
    artifacts["voice_prompt:easton"] = prompt
    cache = tmp_path / "must-not-exist"
    monkeypatch.setenv("ONNXVOICE_CACHE_DIR", str(cache))

    adapter = OnnxVoice.open_local(
        system="phonon",
        artifacts=artifacts,
        metadata={"license": _CC_BY_NC},
    )
    assert isinstance(adapter, PhononAdapter)
    assert adapter.installation.metadata["managed"] is False
    assert adapter.installation.metadata["license"] == _CC_BY_NC
    assert adapter.installation.path == root / "onnx"
    assert (
        adapter.installation.artifact("model", component="flow_lm_flow").path
        == root / _GRAPH_PATHS["flow_lm_flow"]
    )
    voice = adapter.prepare_voice()
    assert voice.name == "easton"
    assert voice.embeddings.shape == (1, 2, 4)
    assert adapter.diagnostics().layout == "phonon-four-graph-streaming-v1"
    assert adapter.diagnostics().sessions == ()
    assert not cache.exists()

    class LocalManager:
        def __init__(self, installation: Any) -> None:
            self.installation = installation
            self.catalog = _Catalog([])

        def installed(self, system: str | None = None) -> list[Any]:
            return [self.installation] if system in {None, "phonon"} else []

    local_options = resolve_engine_options(
        LocalManager(adapter.installation), "phonon", target=adapter.installation.id
    )
    assert local_options["sample_rate"] == 24000
    assert local_options["license"] == _CC_BY_NC
    local_parameters = {parameter["name"]: parameter for parameter in local_options["parameters"]}
    assert local_parameters["temperature"]["default"] == 0.8
    assert local_parameters["voice"]["choices"] == ["easton"]


def test_options_are_target_sensitive_and_preserve_capability_and_license_facts() -> None:
    manager = _Manager([_catalog_item()])
    data = resolve_engine_options(manager, "phonon", target="phonon-31m-en")
    assert data["quality"] == "fp32"
    assert data["sample_rate"] == 24000
    assert data["layout"] == "phonon-four-graph-streaming-v1"
    assert data["capabilities"]["voice_cloning"]["supported"] is False
    assert data["capabilities"]["streaming"]["supported"] is True
    assert data["capabilities"]["named_voices"]["supported"] is True
    assert data["license"] == _APACHE
    parameters = {parameter["name"]: parameter for parameter in data["parameters"]}
    assert parameters["temperature"]["default"] == 0.8
    assert parameters["temperature"]["source"] == "catalog_metadata"
    assert parameters["max_frames"]["default"] == 500
    assert parameters["max_frames"]["required"] is False
    assert parameters["voice"]["choices"] == ["easton"]
    assert parameters["voice"]["default"] == "easton"
    limits = {limit["name"]: limit for limit in data["limits"]}
    assert limits["max_frames"]["value"] == 500
    assert limits["max_input_tokens"]["status"] == "not_declared"

    unavailable_default = _catalog_item()
    updated_generation = {**_GENERATION, "duration_delta_default": None}
    unavailable_metadata = {
        **unavailable_default.metadata,
        "generation": updated_generation,
        "profiles": {"fp32": {"generation": updated_generation, "license": _APACHE}},
    }
    unavailable = replace(unavailable_default, metadata=unavailable_metadata)
    required = resolve_engine_options(_Manager([unavailable]), "phonon", target="phonon-31m-en")
    duration = next(
        parameter for parameter in required["parameters"] if parameter["name"] == "duration_delta"
    )
    assert duration["required"] is True
    assert duration["default_status"] == "unknown"


def test_inventory_and_cli_discover_only_catalog_targets_and_show_license(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    item = _catalog_item()
    records = query_inventory([], [item])
    assert [(record.system, record.id) for record in records] == [("phonon", "phonon-31m-en")]
    assert records[0].language_codes == ("en",)
    catalog_item = records[0].catalog_item
    assert catalog_item is not None
    assert catalog_item.metadata["license"]["spdx"] == "Apache-2.0"

    fake_manager = _Manager([item])

    def manager_factory(args: Any) -> _Manager:
        del args
        return fake_manager

    monkeypatch.setattr(cli, "_manager", manager_factory)
    result = cli.main(["engines", "list", "--engine", "phonon", "--format", "json"])
    assert result == 0
    listing = json.loads(capsys.readouterr().out)
    assert [(row["target"], row["quality_choices"]) for row in listing["rows"]] == [
        ("phonon-31m-en", ["fp32"])
    ]
    assert listing["rows"][0]["license_spdx"] == "Apache-2.0"
    assert listing["rows"][0]["commercial_use"] is True

    result = cli.main(
        [
            "engines",
            "options",
            "--engine",
            "phonon",
            "--target",
            "phonon-31m-en",
            "--format",
            "json",
        ]
    )
    assert result == 0
    options = json.loads(capsys.readouterr().out)
    assert options["capabilities"]["voice_cloning"]["supported"] is False
    assert options["license"]["commercial_use"] is True
