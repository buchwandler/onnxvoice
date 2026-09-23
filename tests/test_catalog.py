from __future__ import annotations

import json

import pytest

from onnxvoice.catalog import CatalogClient
from onnxvoice.errors import CatalogError
from onnxvoice.types import CatalogItem


def test_piper_catalog_is_normalized(tmp_path):
    raw = {
        "source": {
            "repository": "buchwandler/piper-onnx-voices",
            "revision": "a" * 40,
            "requested_revision": "main",
        },
        "voices": {
            "en_US-test-medium": {
                "id": "en_US-test-medium",
                "aliases": ["en-us-test-medium"],
                "quality": "medium",
                "language": {"code": "en_US"},
                "artifacts": {
                    "config": {
                        "role": "config",
                        "filename": "voice.onnx.json",
                        "url": "https://example.invalid/voice.onnx.json",
                        "size": 10,
                        "md5": "a" * 32,
                    },
                    "model": {
                        "role": "model",
                        "filename": "voice.onnx",
                        "url": "https://example.invalid/voice.onnx",
                        "size": 20,
                        "md5": "b" * 32,
                    },
                },
            }
        },
    }
    catalog = tmp_path / "piper.json"
    catalog.write_text(json.dumps(raw), encoding="utf-8")
    client = CatalogClient(cache_dir=tmp_path / "cache", sources={"piper": str(catalog)})
    item = client.resolve("piper:en-us-test-medium")
    assert item.id == "en_US-test-medium"
    assert item.metadata["quality"] == "medium"
    assert {artifact.role for artifact in item.artifacts} == {"config", "model"}
    assert item.metadata["source_repository"] == "buchwandler/piper-onnx-voices"
    assert item.metadata["source_revision"] == "a" * 40
    assert item.metadata["requested_revision"] == "main"


def test_kokoro_default_selects_one_model_quality(tmp_path):
    raw = {
        "models": {
            "v1.0": {
                "sample_rate": 24000,
                "runtime": {"default_voice": "af_heart", "voices": ["af_heart"]},
                "distributions": [
                    {
                        "id": "dist",
                        "runtime_ready": True,
                        "artifacts": [
                            {
                                "id": "m1",
                                "role": "model",
                                "local_name": "m.onnx",
                                "url": "x",
                                "sha256": "1" * 64,
                                "quality": "fp32",
                            },
                            {
                                "id": "m2",
                                "role": "model",
                                "local_name": "m.fp16.onnx",
                                "url": "x",
                                "sha256": "2" * 64,
                                "quality": "fp16",
                            },
                            {
                                "id": "v",
                                "role": "voices",
                                "local_name": "voices.npz",
                                "url": "x",
                                "sha256": "3" * 64,
                            },
                        ],
                    }
                ],
            }
        }
    }
    catalog = tmp_path / "kokoro.json"
    catalog.write_text(json.dumps(raw), encoding="utf-8")
    client = CatalogClient(cache_dir=tmp_path / "cache", sources={"kokoro": str(catalog)})
    default = client.resolve("kokoro:v1.0")
    assert [a.quality for a in default.artifacts if a.role == "model"] == ["fp32"]
    fp16 = client.resolve("kokoro:v1.0", quality="fp16")
    assert [a.quality for a in fp16.artifacts if a.role == "model"] == ["fp16"]


def test_kokoro_distribution_selection_is_explicit(tmp_path):
    raw = {
        "models": {
            "v1.0": {
                "runtime": {"layout": "single"},
                "distributions": [
                    {"id": "cpu", "artifacts": [{"role": "model", "id": "cpu.onnx", "url": "x"}]},
                    {
                        "id": "mobile",
                        "artifacts": [{"role": "model", "id": "mobile.onnx", "url": "x"}],
                    },
                ],
            }
        }
    }
    source = tmp_path / "kokoro.json"
    source.write_text(json.dumps(raw), encoding="utf-8")
    client = CatalogClient(cache_dir=tmp_path / "cache", sources={"kokoro": str(source)})

    assert client.resolve("kokoro:v1.0").metadata["distribution_id"] == "cpu"
    selected = client.resolve("kokoro:v1.0", distribution="mobile")
    assert selected.metadata["distribution_id"] == "mobile"
    assert selected.artifacts[0].filename == "mobile.onnx"
    with pytest.raises(CatalogError, match="valid choices: cpu, mobile"):
        client.resolve("kokoro:v1.0", distribution="unknown")


def test_kokoro_distribution_selection_is_scoped_to_requested_model(tmp_path):
    """P0 regression: resolving v1.0 with its distribution must not fail on v1.1-zh."""
    raw = {
        "models": {
            "v1.0": {
                "distributions": [
                    {
                        "id": "dist-v1",
                        "artifacts": [{"role": "model", "id": "v1.onnx", "url": "x"}],
                    }
                ],
            },
            "v1.1-zh": {
                "distributions": [
                    {
                        "id": "dist-zh",
                        "artifacts": [{"role": "model", "id": "zh.onnx", "url": "x"}],
                    }
                ],
            },
        }
    }
    source = tmp_path / "kokoro.json"
    source.write_text(json.dumps(raw), encoding="utf-8")
    client = CatalogClient(cache_dir=tmp_path / "cache", sources={"kokoro": str(source)})

    # This used to fail because distribution validation applied to ALL models
    item = client.resolve("kokoro:v1.0", distribution="dist-v1")
    assert item.id == "v1.0"
    assert item.metadata["distribution_id"] == "dist-v1"

    # The other model should also work with its own distribution
    zh = client.resolve("kokoro:v1.1-zh", distribution="dist-zh")
    assert zh.id == "v1.1-zh"
    assert zh.metadata["distribution_id"] == "dist-zh"


def test_kokoro_wrong_distribution_error_names_requested_model(tmp_path):
    """Error must reference the requested model, not an unrelated one."""
    raw = {
        "models": {
            "v1.0": {
                "distributions": [
                    {"id": "dist-v1", "artifacts": [{"role": "model", "id": "a.onnx", "url": "x"}]}
                ],
            },
            "v1.1-zh": {
                "distributions": [
                    {"id": "dist-zh", "artifacts": [{"role": "model", "id": "b.onnx", "url": "x"}]}
                ],
            },
        }
    }
    source = tmp_path / "kokoro.json"
    source.write_text(json.dumps(raw), encoding="utf-8")
    client = CatalogClient(cache_dir=tmp_path / "cache", sources={"kokoro": str(source)})

    # Error must reference v1.0, not v1.1-zh
    with pytest.raises(CatalogError, match="for 'v1.0'"):
        client.resolve("kokoro:v1.0", distribution="dist-zh")

    # And vice versa
    with pytest.raises(CatalogError, match="for 'v1.1-zh'"):
        client.resolve("kokoro:v1.1-zh", distribution="dist-v1")


def test_kokoro_list_uses_each_models_own_default_distribution(tmp_path):
    """list() should parse all models without any caller-specific distribution constraint."""
    raw = {
        "models": {
            "v1.0": {
                "distributions": [
                    {"id": "dist-a", "artifacts": [{"role": "model", "id": "a.onnx", "url": "x"}]},
                    {"id": "dist-b", "artifacts": [{"role": "model", "id": "b.onnx", "url": "x"}]},
                ],
            },
            "v1.1-zh": {
                "distributions": [
                    {
                        "id": "dist-zh",
                        "artifacts": [{"role": "model", "id": "zh.onnx", "url": "x"}],
                    },
                ],
            },
        }
    }
    source = tmp_path / "kokoro.json"
    source.write_text(json.dumps(raw), encoding="utf-8")
    client = CatalogClient(cache_dir=tmp_path / "cache", sources={"kokoro": str(source)})

    items = client.list("kokoro")
    assert len(items) == 2
    by_id = {item.id: item for item in items}
    assert by_id["v1.0"].metadata["distribution_id"] == "dist-a"
    assert by_id["v1.1-zh"].metadata["distribution_id"] == "dist-zh"


def test_kokoro_list_skips_models_without_runtime_ready_distribution(tmp_path):
    raw = {
        "models": {
            "kokoro-en-us": {
                "language_codes": ["en-US"],
                "runtime": {"voices": ["af_heart"]},
                "distributions": [{"id": "cpu", "artifacts": [{"role": "model", "id": "en.onnx"}]}],
            },
            "he-hebrew-nc": {
                "language_codes": ["he"],
                "distributions": [{"id": "unsupported", "runtime_ready": False, "artifacts": []}],
            },
        }
    }
    source = tmp_path / "kokoro.json"
    source.write_text(json.dumps(raw), encoding="utf-8")
    client = CatalogClient(cache_dir=tmp_path / "cache", sources={"kokoro": str(source)})

    assert [item.id for item in client.list("kokoro")] == ["kokoro-en-us"]
    with pytest.raises(CatalogError, match="no runtime-ready distributions"):
        client.resolve("kokoro:he-hebrew-nc")


def test_kokoro_timing_contract_is_normalized(tmp_path):
    raw = {
        "models": {
            "v1.0": {
                "runtime": {"layout": "single-onnx-v1"},
                "onnx_contract": {
                    "outputs": {"waveform": "float32", "durations": "float32"},
                    "timing": {
                        "kind": "token-duration-v1",
                        "output": "durations",
                        "unit": "frame",
                        "samples_per_frame": 600,
                        "includes_boundary_tokens": True,
                    },
                },
                "distributions": [
                    {"id": "dist", "artifacts": [{"role": "model", "id": "model.onnx"}]}
                ],
            }
        }
    }
    source = tmp_path / "kokoro.json"
    source.write_text(json.dumps(raw), encoding="utf-8")
    item = CatalogClient(cache_dir=tmp_path / "cache", sources={"kokoro": str(source)}).resolve(
        "kokoro:v1.0"
    )

    assert item.metadata["runtime"]["timings_output"] == "durations"
    assert item.metadata["onnx_contract"] == raw["models"]["v1.0"]["onnx_contract"]
    assert item.metadata["onnx_contract"]["timing"]["output"] == "durations"
    assert item.timing_output == "durations"
    assert "timings_output" not in raw["models"]["v1.0"]["runtime"]


def test_kokoro_matching_timing_declarations_are_accepted(tmp_path):
    raw = {
        "models": {
            "v1.0": {
                "runtime": {"timings_output": "durations"},
                "onnx_contract": {"timing": {"output": "durations"}},
                "distributions": [
                    {"id": "dist", "artifacts": [{"role": "model", "id": "model.onnx"}]}
                ],
            }
        }
    }
    source = tmp_path / "kokoro.json"
    source.write_text(json.dumps(raw), encoding="utf-8")
    item = CatalogClient(cache_dir=tmp_path / "cache", sources={"kokoro": str(source)}).resolve(
        "kokoro:v1.0"
    )
    assert item.timing_output == "durations"


def test_kokoro_conflicting_timing_declarations_fail(tmp_path):
    raw = {
        "models": {
            "v1.0": {
                "runtime": {"timings_output": "pred_dur"},
                "onnx_contract": {"timing": {"output": "durations"}},
                "distributions": [
                    {"id": "dist", "artifacts": [{"role": "model", "id": "model.onnx"}]}
                ],
            }
        }
    }
    source = tmp_path / "kokoro.json"
    source.write_text(json.dumps(raw), encoding="utf-8")
    client = CatalogClient(cache_dir=tmp_path / "cache", sources={"kokoro": str(source)})

    with pytest.raises(
        CatalogError,
        match="runtime.timings_output='pred_dur'.*onnx_contract.timing.output='durations'",
    ):
        client.resolve("kokoro:v1.0")


@pytest.mark.parametrize(
    "metadata",
    [
        {},
        {"runtime": None},
        {"runtime": {"timings_output": ""}},
        {"runtime": {"timings_output": 1}},
    ],
)
def test_catalog_item_timing_output_tolerates_malformed_metadata(metadata):
    item = CatalogItem("kokoro", "v1.0", "model", (), metadata=metadata)
    assert item.timing_output is None
