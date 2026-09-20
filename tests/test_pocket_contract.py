"""Contract and manifest tests for the Pocket v2 runtime boundary."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from onnxvoice.errors import RuntimeContractError
from onnxvoice.systems.pocket import (
    PocketAdapter,
    StateSpec,
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
    },
    {
        "index": 1,
        "input_name": "state_1",
        "output_name": "out_state_1",
        "shape": [0],
        "dtype": "float32",
        "fill": "empty",
    },
    {
        "index": 2,
        "input_name": "state_2",
        "output_name": "out_state_2",
        "shape": [1],
        "dtype": "int64",
        "fill": "zeros",
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
