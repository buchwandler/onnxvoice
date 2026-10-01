from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from onnxvoice.errors import RuntimeContractError
from onnxvoice.systems import KittenAdapter
from onnxvoice.types import Installation, InstalledArtifact, TensorSpec

_DEFAULT_SPECS = (
    TensorSpec("input_ids", "tensor(int64)", (1, "tokens")),
    TensorSpec("style", "tensor(float)", (1, 4)),
    TensorSpec("speed", "tensor(float)", (1,)),
)
_TOKEN_IDS = [0, 31, 10, 0]
_STYLE = np.asarray([[0.25, 0.5, 0.75, 1.0]], dtype=np.float32)


class FakeSession:
    def __init__(
        self,
        *,
        input_specs: tuple[TensorSpec, ...] = _DEFAULT_SPECS,
        input_names: tuple[str, ...] | None = None,
        outputs: list[np.ndarray] | None = None,
    ) -> None:
        self.input_specs = input_specs
        self.input_names = input_names or tuple(spec.name for spec in input_specs)
        self.outputs = (
            outputs if outputs is not None else [np.arange(6000, dtype=np.float32)[None, :]]
        )
        self.calls: list[dict[str, np.ndarray]] = []

    def run(self, inputs: dict[str, np.ndarray]) -> list[np.ndarray]:
        self.calls.append({name: np.array(value, copy=True) for name, value in inputs.items()})
        return self.outputs


@pytest.fixture
def adapter_factory(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    created: list[tuple[Path, dict[str, Any]]] = []
    sessions: list[FakeSession] = []

    def make(
        *,
        input_specs: tuple[TensorSpec, ...] = _DEFAULT_SPECS,
        input_names: tuple[str, ...] | None = None,
        outputs: list[np.ndarray] | None = None,
        sample_rate: int | None = 24000,
        providers: Any = None,
        provider_options: Any = None,
        session_options: Any = None,
    ) -> KittenAdapter:
        model = tmp_path / f"model-{len(sessions)}.onnx"
        model.write_bytes(b"fixture-model")
        installation = Installation(
            system="kitten",
            id="nano-0.8-int8",
            kind="external",
            path=tmp_path,
            artifacts=(
                InstalledArtifact(
                    role="model",
                    filename=model.name,
                    path=model,
                    sha256="a" * 64,
                    size=model.stat().st_size,
                    format="onnx",
                ),
            ),
            sample_rate=sample_rate,
            metadata={"voice_aliases": {"Bella": "expr-voice-2-f"}},
        )
        session = FakeSession(
            input_specs=input_specs,
            input_names=input_names,
            outputs=outputs,
        )
        sessions.append(session)

        def create_session(path: Path, **kwargs: Any) -> FakeSession:
            created.append((path, kwargs))
            return session

        monkeypatch.setattr("onnxvoice.systems.kitten.OnnxSession", create_session)
        return KittenAdapter(
            installation,
            providers=providers,
            provider_options=provider_options,
            session_options=session_options,
        )

    return make, sessions, created


def test_infer_preserves_model_ready_tokens_style_speed_and_trims_exact_tail(
    adapter_factory,
) -> None:
    make, sessions, created = adapter_factory
    adapter = make()
    result = adapter.infer(_TOKEN_IDS, style=_STYLE, speed=0.8)

    assert created[0][0].name == "model-0.onnx"
    assert len(sessions[0].calls) == 1
    inputs = sessions[0].calls[0]
    assert inputs["input_ids"].tolist() == [_TOKEN_IDS]
    assert inputs["input_ids"].dtype == np.int64
    np.testing.assert_array_equal(inputs["style"], _STYLE)
    np.testing.assert_array_equal(inputs["speed"], np.asarray([0.8], dtype=np.float32))
    assert result.audio.dtype == np.float32
    assert result.audio.ndim == 1
    assert result.audio.shape == (1000,)
    np.testing.assert_array_equal(result.audio, np.arange(1000, dtype=np.float32))
    assert result.sample_rate == 24000
    assert result.metadata == {"system": "kitten", "speed": 0.8, "tail_trim_samples": 5000}


def test_session_is_lazy_and_receives_provider_configuration(adapter_factory) -> None:
    make, sessions, created = adapter_factory
    options = [{"device_id": "3"}]
    session_options = object()
    adapter = make(
        providers=["CUDAExecutionProvider"],
        provider_options=options,
        session_options=session_options,
    )

    assert created == []
    adapter.infer(_TOKEN_IDS, style=_STYLE)
    assert len(created) == 1
    _, kwargs = created[0]
    assert kwargs == {
        "providers": ["CUDAExecutionProvider"],
        "provider_options": options,
        "session_options": session_options,
    }
    assert sessions[0].calls


def test_input_dtype_is_derived_from_graph_specs(adapter_factory) -> None:
    make, sessions, _ = adapter_factory
    specs = (
        TensorSpec("input_ids", "tensor(int32)", (1, "tokens")),
        TensorSpec("style", "tensor(double)", (1, 4)),
        TensorSpec("speed", "tensor(float16)", (1,)),
    )
    adapter = make(input_specs=specs)
    style = _STYLE.astype(np.float64)

    adapter.infer(_TOKEN_IDS, style=style)
    inputs = sessions[0].calls[0]
    assert inputs["input_ids"].dtype == np.int32
    assert inputs["style"].dtype == np.float64
    assert inputs["speed"].dtype == np.float16


def test_style_rank_follows_the_declared_graph_shape(adapter_factory) -> None:
    make, sessions, _ = adapter_factory
    specs = (
        _DEFAULT_SPECS[0],
        TensorSpec("style", "tensor(float)", (1, 1, 4)),
        _DEFAULT_SPECS[2],
    )
    style = _STYLE.reshape(1, 1, 4)
    adapter = make(input_specs=specs)

    adapter.infer(_TOKEN_IDS, style=style)
    np.testing.assert_array_equal(sessions[0].calls[0]["style"], style)


@pytest.mark.parametrize(
    ("token_ids", "message"),
    [
        ([], "non-empty integer sequence"),
        ([1.5, 2], "non-empty integer sequence"),
        ([True, 2], "non-empty integer sequence"),
        ([[1, 2]], "non-empty integer sequence"),
        ([2**63], "do not fit graph dtype"),
    ],
)
def test_rejects_invalid_token_ids(adapter_factory, token_ids, message: str) -> None:
    make, sessions, _ = adapter_factory
    adapter = make()

    with pytest.raises(RuntimeContractError, match=message):
        adapter.infer(token_ids, style=_STYLE)
    assert sessions[0].calls == []


@pytest.mark.parametrize(
    ("style", "spec", "message"),
    [
        (np.ones((4,), dtype=np.float32), _DEFAULT_SPECS[1], "batch dimension of 1"),
        (np.ones((2, 4), dtype=np.float32), _DEFAULT_SPECS[1], "batch dimension of 1"),
        (np.ones((1, 3), dtype=np.float32), _DEFAULT_SPECS[1], "axis 1 must have size 4"),
        (np.ones((1, 4), dtype=np.int64), _DEFAULT_SPECS[1], "finite floating array"),
        (np.full((1, 4), np.nan, dtype=np.float32), _DEFAULT_SPECS[1], "finite floating array"),
    ],
)
def test_rejects_invalid_style(adapter_factory, style, spec, message: str) -> None:
    make, sessions, _ = adapter_factory
    specs = (_DEFAULT_SPECS[0], spec, _DEFAULT_SPECS[2])
    adapter = make(input_specs=specs)

    with pytest.raises(RuntimeContractError, match=message):
        adapter.infer(_TOKEN_IDS, style=style)
    assert sessions[0].calls == []


@pytest.mark.parametrize("speed", [True, 0, -1, float("nan"), float("inf"), "fast"])
def test_rejects_invalid_speed(adapter_factory, speed) -> None:
    make, sessions, _ = adapter_factory
    adapter = make()

    with pytest.raises(RuntimeContractError, match="speed"):
        adapter.infer(_TOKEN_IDS, style=_STYLE, speed=speed)
    assert sessions[0].calls == []


def test_missing_style_and_unexpected_arguments_are_rejected(adapter_factory) -> None:
    make, sessions, _ = adapter_factory
    adapter = make()

    with pytest.raises(TypeError, match="style"):
        adapter.infer(_TOKEN_IDS)
    with pytest.raises(TypeError, match="unexpected"):
        adapter.infer(_TOKEN_IDS, style=_STYLE, unexpected=True)
    assert sessions[0].calls == []


@pytest.mark.parametrize(
    ("input_names", "message"),
    [
        (("input_ids", "style"), "missing: speed"),
        (("input_ids", "style", "speed", "extra"), "unexpected: extra"),
    ],
)
def test_rejects_wrong_graph_input_names(adapter_factory, input_names, message: str) -> None:
    make, sessions, _ = adapter_factory
    adapter = make(input_names=input_names)

    with pytest.raises(RuntimeContractError, match=message):
        adapter.infer(_TOKEN_IDS, style=_STYLE)
    assert sessions[0].calls == []


@pytest.mark.parametrize(
    ("index", "spec", "message"),
    [
        (0, TensorSpec("input_ids", "tensor(float)", (1, "tokens")), "input_ids.*integer"),
        (0, TensorSpec("input_ids", "tensor(int64)", (2, "tokens")), "axis 0 must have size 2"),
        (1, TensorSpec("style", "tensor(int64)", (1, 4)), "style.*floating"),
        (1, TensorSpec("style", "tensor(float)", (1, 8)), "style.*axis 1"),
        (2, TensorSpec("speed", "tensor(int64)", (1,)), "speed.*floating"),
        (2, TensorSpec("speed", "tensor(float)", (2,)), "axis 0 must have size 2"),
    ],
)
def test_rejects_incompatible_graph_tensor_specs(
    adapter_factory, index, spec, message: str
) -> None:
    make, sessions, _ = adapter_factory
    specs = list(_DEFAULT_SPECS)
    specs[index] = spec
    adapter = make(input_specs=tuple(specs))

    with pytest.raises(RuntimeContractError, match=message):
        adapter.infer(_TOKEN_IDS, style=_STYLE)
    assert sessions[0].calls == []


def test_rejects_output_shorter_than_or_equal_to_tail_trim(adapter_factory) -> None:
    make, _, _ = adapter_factory
    adapter = make(outputs=[np.zeros((1, 5000), dtype=np.float32)])

    with pytest.raises(RuntimeContractError, match="too short"):
        adapter.infer(_TOKEN_IDS, style=_STYLE)


def test_rejects_nonfinite_and_nonmono_outputs(adapter_factory) -> None:
    make, _, _ = adapter_factory
    nonfinite = make(outputs=[np.full((1, 6000), np.nan, dtype=np.float32)])
    with pytest.raises(RuntimeContractError, match="non-finite"):
        nonfinite.infer(_TOKEN_IDS, style=_STYLE)

    multichannel = make(outputs=[np.zeros((2, 6000), dtype=np.float32)])
    with pytest.raises(RuntimeContractError, match="mono"):
        multichannel.infer(_TOKEN_IDS, style=_STYLE)


def test_rejects_incompatible_sample_rate_and_defaults_missing_rate(adapter_factory) -> None:
    make, _, _ = adapter_factory
    incompatible = make(sample_rate=22050)
    with pytest.raises(RuntimeContractError, match="24000 Hz"):
        incompatible.infer(_TOKEN_IDS, style=_STYLE)

    missing = make(sample_rate=None)
    result = missing.infer(_TOKEN_IDS, style=_STYLE)
    assert result.sample_rate == 24000
