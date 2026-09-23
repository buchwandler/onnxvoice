"""Contract and manifest tests for the Pocket v2 runtime boundary."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from onnxvoice.errors import RuntimeContractError
from onnxvoice.systems.pocket import (
    PocketAdapter,
    StateSpec,
    _adapt_state_tensor,
    _derive_step,
    _flow_state_from_model_state,
    initialize_state,
    parse_state_manifest,
    update_state_from_named_outputs,
)
from onnxvoice.types import Installation

MANIFEST = [
    {
        "index": 0,
        "input_name": "state_0",
        "output_name": "out_state_0",
        "shape": [2, 1, 4, 1, 2],
        "dtype": "float32",
        "fill": "nan",
        "module": "transformer.layers.0.self_attn",
        "key": "cache",
    },
    {
        "index": 1,
        "input_name": "state_1",
        "output_name": "out_state_1",
        "shape": [0],
        "dtype": "float32",
        "fill": "empty",
        "module": "transformer.layers.0.self_attn",
        "key": "current_end",
    },
    {
        "index": 2,
        "input_name": "state_2",
        "output_name": "out_state_2",
        "shape": [1],
        "dtype": "int64",
        "fill": "zeros",
        "module": "transformer.layers.0.self_attn",
        "key": "step",
    },
]


def _adapter() -> PocketAdapter:
    return PocketAdapter(
        Installation(
            system="pocket",
            id="fixture",
            kind="bundle",
            path=Path("."),
            artifacts=(),
            metadata={},
        )
    )


class _Session:
    def __init__(self, inputs: tuple[str, ...], outputs: tuple[str, ...]) -> None:
        self.input_names = inputs
        self.output_names = outputs


def test_manifest_parses_and_initializes_declared_fills() -> None:
    specs = parse_state_manifest(MANIFEST, name="flow_lm")
    assert all(isinstance(spec, StateSpec) for spec in specs)
    assert [spec.index for spec in specs] == [0, 1, 2]
    state = initialize_state(specs)
    assert state["state_0"].shape == (2, 1, 4, 1, 2)
    assert np.isnan(state["state_0"]).all()
    assert state["state_1"].shape == (0,)
    assert state["state_2"].dtype == np.int64
    assert np.array_equal(state["state_2"], [0])


def test_manifest_source_mappings_are_optional_but_paired() -> None:
    specs = parse_state_manifest(MANIFEST, name="flow_lm")
    assert specs[0].module == "transformer.layers.0.self_attn"
    assert specs[0].key == "cache"

    legacy_manifest = [
        {key: value for key, value in entry.items() if key not in {"module", "key"}}
        for entry in MANIFEST
    ]
    assert all(
        spec.module is None and spec.key is None
        for spec in parse_state_manifest(legacy_manifest, name="flow_lm")
    )


@pytest.mark.parametrize(
    "source",
    [
        {**MANIFEST[0], "key": None},
        {**MANIFEST[0], "module": None},
        {**MANIFEST[0], "module": "", "key": "cache"},
        {**MANIFEST[0], "module": "module", "key": ""},
        {**MANIFEST[0], "module": 1},
        {**MANIFEST[0], "key": 1},
    ],
)
def test_manifest_rejects_invalid_source_mappings(source: dict[str, object]) -> None:
    with pytest.raises(RuntimeContractError, match="module|key"):
        parse_state_manifest([source], name="flow_lm")


def test_predefined_flow_state_import_adapts_manifest_tensors() -> None:
    specs = parse_state_manifest(MANIFEST, name="flow_lm")
    cache = np.ones((2, 1, 3, 1, 2), dtype=np.float32)
    state = _flow_state_from_model_state(
        {
            "transformer.layers.0.self_attn": {
                "cache": cache,
                "current_end": np.zeros(3, dtype=np.float32),
            }
        },
        specs,
        name="alba",
    )

    assert state["state_0"].shape == (2, 1, 4, 1, 2)
    assert np.array_equal(state["state_0"][:, :, :3], cache)
    assert np.isnan(state["state_0"][:, :, 3:]).all()
    assert state["state_1"].shape == (0,)
    assert np.array_equal(state["state_2"], [3])
    assert np.array_equal(_derive_step({"step": np.asarray(7)}), [7])
    assert np.array_equal(_derive_step({}), [0])
    assert np.array_equal(_derive_step({"offset": np.asarray(4)}), [4])
    assert np.array_equal(
        _derive_step(
            {"offset": np.asarray(4), "end_offset": np.asarray(9), "current_end": np.zeros(6)}
        ),
        [6],
    )
    partial = _flow_state_from_model_state(
        {"transformer.layers.0.self_attn": {"current_end": np.zeros(3, dtype=np.float32)}},
        specs,
        name="alba",
    )
    assert np.isnan(partial["state_0"]).all()


def test_state_tensor_adaptation_keeps_initialized_state_on_rank_mismatch() -> None:
    spec = parse_state_manifest(MANIFEST[:1], name="flow_lm")[0]
    state = _adapt_state_tensor(np.ones((3, 2), dtype=np.float32), spec)
    assert state.shape == spec.shape
    assert np.isnan(state).all()


def test_manifest_updates_by_named_outputs() -> None:
    specs = parse_state_manifest(MANIFEST, name="flow_lm")
    state = initialize_state(specs)
    replacement = np.ones((2, 1, 4, 1, 2), dtype=np.float32)
    update_state_from_named_outputs(
        state,
        {
            "out_state_0": replacement,
            "out_state_1": np.empty((0,), np.float32),
            "out_state_2": np.array([3]),
        },
        specs,
    )
    assert np.array_equal(state["state_0"], replacement)
    assert state["state_2"][0] == 3


@pytest.mark.parametrize("raw", [{}, "not-a-manifest", {"state": {}}])
def test_manifest_rejects_non_list_schema(raw: object) -> None:
    with pytest.raises(RuntimeContractError, match="must be a sequence"):
        parse_state_manifest(raw, name="flow_lm")


def test_manifest_rejects_nan_for_integer_state() -> None:
    bad = [{**MANIFEST[2], "fill": "nan"}]
    with pytest.raises(RuntimeContractError, match="non-floating"):
        parse_state_manifest(bad, name="flow_lm")


def test_flow_contract_requires_named_inputs_and_state_wiring() -> None:
    adapter = _adapter()
    adapter._flow_specs = parse_state_manifest(MANIFEST[:1], name="flow_lm")
    adapter._validate_session_contract(
        "flow_lm_main",
        _Session(
            ("sequence", "text_embeddings", "state_0"), ("conditioning", "eos", "out_state_0")
        ),
    )

    with pytest.raises(RuntimeContractError, match="missing inputs: x"):
        adapter._validate_session_contract(
            "flow_lm_flow",
            _Session(("c", "s", "t"), ("velocity",)),
        )


def test_decoder_contract_requires_latent_and_audio() -> None:
    adapter = _adapter()
    with pytest.raises(RuntimeContractError, match="latent"):
        adapter._validate_session_contract("mimi_decoder", _Session(("audio",), ("audio",)))
    adapter._validate_session_contract("mimi_decoder", _Session(("latent",), ("audio",)))
