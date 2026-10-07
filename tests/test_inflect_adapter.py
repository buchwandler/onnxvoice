from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

import onnxvoice.systems.inflect as inflect_module
from onnxvoice.errors import CapabilityError, RuntimeContractError
from onnxvoice.systems.inflect import InflectAdapter
from onnxvoice.types import Installation, InstalledArtifact, SessionDiagnostic, TensorSpec

_DURATION_INPUTS = {
    "tokens": TensorSpec("tokens", "tensor(int64)", (1, None)),
    "lengths": TensorSpec("lengths", "tensor(int64)", (1,)),
    "length_scale": TensorSpec("length_scale", "tensor(float)", ()),
}
_DURATION_OUTPUTS = {
    "m_p_exp": TensorSpec("m_p_exp", "tensor(float)", (1, 1, 2)),
    "logs_p_exp": TensorSpec("logs_p_exp", "tensor(float)", (1, 1, 2)),
    "y_mask": TensorSpec("y_mask", "tensor(float)", (1, 1, 2)),
}
_DECODE_INPUTS = {
    name: TensorSpec(name, "tensor(float)", (1, 1, 2))
    for name in ("m_p_exp", "logs_p_exp", "y_mask", "zp_noise")
}
_DECODE_INPUTS["noise_scale"] = TensorSpec("noise_scale", "tensor(float)", ())
_DECODE_OUTPUTS = {"waveform": TensorSpec("waveform", "tensor(float)", (1, 1, 3))}


class FakeOnnxSession:
    def __init__(
        self,
        path: Path,
        *,
        component: str,
        providers: Any,
        provider_options: Any,
        session_options: Any,
        input_names: tuple[str, ...] | None = None,
        output_names: tuple[str, ...] | None = None,
        input_specs: dict[str, TensorSpec] | None = None,
        output_specs: dict[str, TensorSpec] | None = None,
        waveform: Any | None = None,
    ) -> None:
        self.model = path
        self.component = component
        self.providers = providers
        self.provider_options = provider_options
        self.session_options = session_options
        default_inputs = _DURATION_INPUTS if component == "duration" else _DECODE_INPUTS
        default_outputs = _DURATION_OUTPUTS if component == "duration" else _DECODE_OUTPUTS
        self.input_specs = tuple((input_specs or default_inputs).values())
        self.output_specs = tuple((output_specs or default_outputs).values())
        self.input_names = input_names or tuple(spec.name for spec in self.input_specs)
        self.output_names = output_names or tuple(spec.name for spec in self.output_specs)
        self.waveform = (
            np.asarray([[[0.1, -0.2, 0.3]]], dtype=np.float32)
            if waveform is None
            else waveform
        )
        self.inputs_seen: dict[str, Any] | None = None
        self.closed = False

    def run(self, inputs: dict[str, Any]) -> list[Any]:
        self.inputs_seen = inputs
        if self.component == "duration":
            outputs = {
                "m_p_exp": np.asarray([[[0.1, 0.2]]], dtype=np.float32),
                "logs_p_exp": np.asarray([[[0.3, 0.4]]], dtype=np.float32),
                "y_mask": np.asarray([[[1.0, 1.0]]], dtype=np.float32),
            }
        else:
            outputs = {"waveform": self.waveform}
        return [outputs[name] for name in self.output_names]

    def diagnostics(self) -> SessionDiagnostic:
        return SessionDiagnostic(
            component=self.component,
            model_path=self.model,
            providers_requested=("CPUExecutionProvider",),
            providers_active=("CPUExecutionProvider",),
            inputs=tuple(self.input_names),
            outputs=tuple(self.output_names),
        )

    def close(self) -> None:
        self.closed = True


def _installation(tmp_path: Path, roles: tuple[str, ...] = ("duration", "decode"), *, sample_rate: int | None = 24000) -> Installation:
    artifacts = tuple(
        InstalledArtifact(
            role=role,
            filename=f"{role}.onnx",
            path=tmp_path / f"{role}.onnx",
            sha256="0" * 64,
            size=1,
            format="onnx",
        )
        for role in roles
    )
    return Installation(
        system="inflect",
        id="nano-v2",
        kind="model",
        path=tmp_path,
        artifacts=artifacts,
        sample_rate=sample_rate,
        voices=("default",),
        default_voice="default",
        metadata={"runtime": {"profile": "inflect-v2-split-v1", "layout": "split"}},
    )


def _adapter(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    roles: tuple[str, ...] = ("duration", "decode"),
    sample_rate: int | None = 24000,
    duration_input_names: tuple[str, ...] | None = None,
    duration_output_names: tuple[str, ...] | None = None,
    decode_input_names: tuple[str, ...] | None = None,
    decode_output_names: tuple[str, ...] | None = None,
    duration_input_specs: dict[str, TensorSpec] | None = None,
    duration_output_specs: dict[str, TensorSpec] | None = None,
    decode_input_specs: dict[str, TensorSpec] | None = None,
    decode_output_specs: dict[str, TensorSpec] | None = None,
    waveform: Any | None = None,
    providers: Any = "cpu",
    provider_options: Any = None,
    session_options: Any = None,
) -> tuple[InflectAdapter, list[FakeOnnxSession]]:
    created: list[FakeOnnxSession] = []

    def factory(
        path: Path,
        *,
        component: str,
        providers: Any,
        provider_options: Any,
        session_options: Any,
    ) -> FakeOnnxSession:
        session = FakeOnnxSession(
            path,
            component=component,
            providers=providers,
            provider_options=provider_options,
            session_options=session_options,
            input_names=duration_input_names if component == "duration" else decode_input_names,
            output_names=duration_output_names if component == "duration" else decode_output_names,
            input_specs=duration_input_specs if component == "duration" else decode_input_specs,
            output_specs=duration_output_specs if component == "duration" else decode_output_specs,
            waveform=waveform,
        )
        created.append(session)
        return session

    monkeypatch.setattr(inflect_module, "OnnxSession", factory)
    adapter = InflectAdapter(
        _installation(tmp_path, roles, sample_rate=sample_rate),
        providers=providers,
        provider_options=provider_options,
        session_options=session_options,
    )
    return adapter, created


def test_inflect_adapter_requires_duration_artifact(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    adapter, _ = _adapter(tmp_path, monkeypatch, roles=("decode",))
    with pytest.raises(CapabilityError, match="missing its 'duration' artifact"):
        _ = adapter.duration_session


def test_inflect_adapter_requires_decode_artifact(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    adapter, _ = _adapter(tmp_path, monkeypatch, roles=("duration",))
    with pytest.raises(CapabilityError, match="missing its 'decode' artifact"):
        _ = adapter.decode_session


def test_inflect_adapter_sessions_are_lazy(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    adapter, created = _adapter(tmp_path, monkeypatch)
    assert created == []
    assert adapter.duration_session.component == "duration"
    assert [session.component for session in created] == ["duration"]
    assert adapter.decode_session.component == "decode"
    assert [session.component for session in created] == ["duration", "decode"]


def test_inflect_adapter_forwards_provider_options_to_both_sessions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    options = [{"device_id": "7"}]
    session_options = object()
    adapter, created = _adapter(
        tmp_path,
        monkeypatch,
        providers=["CPUExecutionProvider"],
        provider_options=options,
        session_options=session_options,
    )
    adapter.infer([4, 5])
    assert [session.component for session in created] == ["duration", "decode"]
    for session in created:
        assert session.providers == ["CPUExecutionProvider"]
        assert session.provider_options == options
        assert session.session_options is session_options


@pytest.mark.parametrize(
    ("component", "names", "message"),
    [
        ("duration_input", ("tokens", "lengths", "scale"), "duration input contract mismatch"),
        (
            "duration_output",
            ("m_p_exp", "logs_p_exp", "mask"),
            "duration output contract mismatch",
        ),
        (
            "decode_input",
            ("m_p_exp", "logs_p_exp", "y_mask", "zp_noise", "scale"),
            "decode input contract mismatch",
        ),
        ("decode_output", ("audio",), "decode output contract mismatch"),
    ],
)
def test_inflect_adapter_validates_graph_names(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    component: str,
    names: tuple[str, ...],
    message: str,
) -> None:
    options = {
        "duration_input": {"duration_input_names": names},
        "duration_output": {"duration_output_names": names},
        "decode_input": {"decode_input_names": names},
        "decode_output": {"decode_output_names": names},
    }
    adapter, _ = _adapter(tmp_path, monkeypatch, **options[component])
    with pytest.raises(RuntimeContractError, match=message):
        adapter.infer([1, 2])


def test_inflect_adapter_rejects_empty_token_ids(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    adapter, created = _adapter(tmp_path, monkeypatch)
    with pytest.raises(RuntimeContractError, match="non-empty integer sequence"):
        adapter.infer([])
    assert created == []


def test_inflect_adapter_rejects_bool_token_ids(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    adapter, _ = _adapter(tmp_path, monkeypatch)
    with pytest.raises(RuntimeContractError, match="integer sequence"):
        adapter.infer([True])


@pytest.mark.parametrize("speed", [True, 0.49, 2.01, float("inf"), float("nan")])
def test_inflect_adapter_rejects_invalid_speed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, speed: Any
) -> None:
    adapter, created = _adapter(tmp_path, monkeypatch)
    with pytest.raises(RuntimeContractError, match="speed"):
        adapter.infer([1], speed=speed)
    assert created == []


@pytest.mark.parametrize("variation", [True, -0.01, 1.01, float("inf"), float("nan")])
def test_inflect_adapter_rejects_invalid_variation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, variation: Any
) -> None:
    adapter, created = _adapter(tmp_path, monkeypatch)
    with pytest.raises(RuntimeContractError, match="variation"):
        adapter.infer([1], variation=variation)
    assert created == []


def test_inflect_adapter_rejects_bool_seed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    adapter, created = _adapter(tmp_path, monkeypatch)
    with pytest.raises(RuntimeContractError, match="seed must be a Python integer"):
        adapter.infer([1], seed=True)
    assert created == []


def test_inflect_adapter_rejects_unexpected_kwargs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    adapter, created = _adapter(tmp_path, monkeypatch)
    with pytest.raises(TypeError, match="unexpected arguments: voice"):
        adapter.infer([1], voice="default")
    assert created == []


def test_inflect_adapter_builds_expected_duration_feeds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    adapter, created = _adapter(tmp_path, monkeypatch)
    adapter.infer([7, 8, 9], speed=2.0)
    feeds = created[0].inputs_seen
    assert feeds is not None
    np.testing.assert_array_equal(feeds["tokens"], np.asarray([[7, 8, 9]], dtype=np.int64))
    np.testing.assert_array_equal(feeds["lengths"], np.asarray([3], dtype=np.int64))
    assert feeds["length_scale"].shape == ()
    assert feeds["length_scale"].dtype == np.float32
    assert feeds["length_scale"].item() == pytest.approx(0.5)


def test_inflect_adapter_builds_expected_decode_feeds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    adapter, created = _adapter(tmp_path, monkeypatch)
    adapter.infer([7, 8], variation=0.25)
    feeds = created[1].inputs_seen
    assert feeds is not None
    assert set(feeds) == {"m_p_exp", "logs_p_exp", "y_mask", "zp_noise", "noise_scale"}
    assert feeds["zp_noise"].shape == feeds["m_p_exp"].shape == (1, 1, 2)
    assert feeds["noise_scale"].shape == ()
    assert feeds["noise_scale"].dtype == np.float32
    assert feeds["noise_scale"].item() == pytest.approx(0.25)


def test_inflect_adapter_seed_is_deterministic(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    adapter, created = _adapter(tmp_path, monkeypatch)
    adapter.infer([3, 4], seed=123)
    first = created[1].inputs_seen["zp_noise"].copy()
    adapter.infer([3, 4], seed=123)
    second = created[1].inputs_seen["zp_noise"].copy()
    np.testing.assert_array_equal(first, second)


def test_inflect_adapter_different_seed_changes_noise(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    adapter, created = _adapter(tmp_path, monkeypatch)
    adapter.infer([3, 4], seed=123)
    first = created[1].inputs_seen["zp_noise"].copy()
    adapter.infer([3, 4], seed=124)
    second = created[1].inputs_seen["zp_noise"].copy()
    assert not np.array_equal(first, second)


def test_inflect_adapter_returns_mono_float32_audio(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    adapter, _ = _adapter(tmp_path, monkeypatch)
    result = adapter.infer([1, 2], speed=1.25, variation=0.4, seed=9)
    assert result.audio.dtype == np.float32
    assert result.audio.ndim == 1
    np.testing.assert_allclose(result.audio, [0.1, -0.2, 0.3])
    assert result.sample_rate == 24000
    assert result.outputs == {}
    assert result.metadata == {
        "system": "inflect",
        "speed": 1.25,
        "variation": 0.4,
        "seed": 9,
    }


def test_inflect_adapter_rejects_nonfinite_audio(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    adapter, _ = _adapter(tmp_path, monkeypatch, waveform=np.asarray([[[0.0, np.nan]]]))
    with pytest.raises(RuntimeContractError, match="waveform must be a finite real"):
        adapter.infer([1])


def test_inflect_adapter_rejects_ambiguous_multichannel_audio(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    adapter, _ = _adapter(
        tmp_path,
        monkeypatch,
        waveform=np.zeros((2, 3), dtype=np.float32),
        decode_output_specs={"waveform": TensorSpec("waveform", "tensor(float)", (2, 3))},
    )
    with pytest.raises(RuntimeContractError, match="unambiguous mono"):
        adapter.infer([1])


def test_inflect_adapter_close_releases_both_sessions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    adapter, created = _adapter(tmp_path, monkeypatch)
    adapter.infer([1, 2])
    adapter.close()
    assert [session.closed for session in created] == [True, True]
    assert adapter._sessions == {}


def test_inflect_adapter_diagnostics_reports_duration_and_decode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    adapter, _ = _adapter(tmp_path, monkeypatch)
    adapter.infer([1, 2])
    diagnostic = adapter.diagnostics()
    assert diagnostic.system == "inflect"
    assert diagnostic.layout == "split"
    assert [session.component for session in diagnostic.sessions] == ["duration", "decode"]


def test_inflect_adapter_rejects_non_24000_sample_rate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    adapter, created = _adapter(tmp_path, monkeypatch, sample_rate=22050)
    with pytest.raises(RuntimeContractError, match="requires a 24000 Hz"):
        adapter.infer([1])
    assert created == []



def test_inflect_adapter_maps_duration_outputs_by_name(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    adapter, created = _adapter(
        tmp_path,
        monkeypatch,
        duration_output_names=("y_mask", "logs_p_exp", "m_p_exp"),
    )
    adapter.infer([1])
    decode_inputs = created[1].inputs_seen
    assert decode_inputs is not None
    np.testing.assert_allclose(decode_inputs["m_p_exp"], [[[0.1, 0.2]]])
    np.testing.assert_allclose(decode_inputs["logs_p_exp"], [[[0.3, 0.4]]])
    np.testing.assert_allclose(decode_inputs["y_mask"], [[[1.0, 1.0]]])


def test_inflect_adapter_rejects_non_fp32_graph_tensors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    inputs = dict(_DURATION_INPUTS)
    inputs["length_scale"] = TensorSpec("length_scale", "tensor(float16)", ())
    adapter, created = _adapter(tmp_path, monkeypatch, duration_input_specs=inputs)
    with pytest.raises(RuntimeContractError, match="must use fp32 precision"):
        adapter.infer([1])
    assert created[0].inputs_seen is None



def test_inflect_adapter_accepts_dynamic_one_element_scale_shape(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    inputs = dict(_DURATION_INPUTS)
    inputs["length_scale"] = TensorSpec("length_scale", "tensor(float)", ("scale",))
    adapter, created = _adapter(tmp_path, monkeypatch, duration_input_specs=inputs)
    adapter.infer([1], speed=2.0)
    scale = created[0].inputs_seen["length_scale"]
    assert scale.shape == (1,)
    assert scale.item() == pytest.approx(0.5)


def test_inflect_adapter_checks_runtime_duration_output_shape(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    outputs = {
        name: TensorSpec(name, "tensor(float)", (1, 1, 3)) for name in _DURATION_OUTPUTS
    }
    adapter, created = _adapter(tmp_path, monkeypatch, duration_output_specs=outputs)
    with pytest.raises(RuntimeContractError, match="axis 2 must have size 3"):
        adapter.infer([1])
    assert created[0].inputs_seen is not None
    assert created[1].inputs_seen is None


def test_inflect_adapter_checks_runtime_waveform_shape(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    outputs = {"waveform": TensorSpec("waveform", "tensor(float)", (1, 1, 2))}
    adapter, created = _adapter(tmp_path, monkeypatch, decode_output_specs=outputs)
    with pytest.raises(RuntimeContractError, match="waveform.*axis 2 must have size 2"):
        adapter.infer([1])
    assert created[1].inputs_seen is not None
