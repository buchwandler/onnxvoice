from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from onnxvoice.errors import CapabilityError, RuntimeContractError
from onnxvoice.systems.kokoro_language_router import KokoroLanguageRouterRuntime
from onnxvoice.types import Installation, InstalledArtifact, SessionDiagnostic, TensorSpec


class FakeSession:
    input_names = ("input_ids", "style", "speed")
    output_names = ("audio", "pred_dur")

    def __init__(self, component: str, *, token_limit: int = 8) -> None:
        self.component = component
        self.input_specs = (
            TensorSpec("input_ids", "tensor(int32)", (1, token_limit)),
            TensorSpec("style", "tensor(float16)", (1, 3)),
            TensorSpec("speed", "tensor(float)", (1,)),
        )
        self.inputs: dict[str, np.ndarray] | None = None
        self.close_count = 0

    def run(self, inputs: dict[str, np.ndarray]) -> list[np.ndarray]:
        self.inputs = inputs
        return [np.asarray([[0.1, 0.2]], dtype=np.float16), np.asarray([2, 3], dtype=np.int32)]

    def diagnostics(self) -> SessionDiagnostic:
        return SessionDiagnostic(
            component=self.component,
            model_path=Path(self.component),
            providers_requested=("CPUExecutionProvider",),
            providers_active=("CPUExecutionProvider",),
            inputs=self.input_names,
            outputs=self.output_names,
        )

    def close(self) -> None:
        self.close_count += 1


def _installation(
    tmp_path: Path,
    *,
    runtime: dict[str, Any] | None = None,
    qualities: dict[str, str | None] | None = None,
    selected_quality: str | None = None,
) -> Installation:
    artifacts = []
    for component in ("model_en", "model_de"):
        for quality in (qualities or {}).get(component, "selected").split(","):
            if quality == "selected":
                quality = selected_quality
            path = tmp_path / f"{component}-{quality or 'default'}.onnx"
            path.write_bytes(component.encode())
            artifacts.append(
                InstalledArtifact(
                    "model",
                    path.name,
                    path,
                    "0" * 64,
                    path.stat().st_size,
                    quality=quality,
                    component=component,
                )
            )
    metadata: dict[str, Any] = {
        "runtime": {
            "layout": "language-router-onnx-v1",
            "language_models": {"en": "model_en", "de": "model_de"},
            "language_codes": ["en", "de"],
        }
    }
    if runtime is not None:
        metadata["runtime"].update(runtime)
    if selected_quality is not None:
        metadata["selected_quality"] = selected_quality
    return Installation(
        "kokoro",
        "akinvox-multispeaker-v1",
        "model",
        tmp_path,
        tuple(artifacts),
        sample_rate=24000,
        metadata=metadata,
    )


def _runtime(
    tmp_path: Path,
    *,
    runtime: dict[str, Any] | None = None,
    qualities: dict[str, str | None] | None = None,
    selected_quality: str | None = None,
) -> tuple[KokoroLanguageRouterRuntime, dict[str, FakeSession], list[tuple[Path, str]]]:
    created: dict[str, FakeSession] = {}
    calls: list[tuple[Path, str]] = []

    def factory(path: Path, component: str) -> FakeSession:
        calls.append((path, component))
        limit = 4 if component == "en:model_en" else 8
        created[component] = FakeSession(component, token_limit=limit)
        return created[component]

    runtime_instance = KokoroLanguageRouterRuntime(
        _installation(
            tmp_path,
            runtime=runtime,
            qualities=qualities,
            selected_quality=selected_quality,
        ),
        session_factory=factory,
    )
    return runtime_instance, created, calls


@pytest.mark.parametrize(
    ("language", "component"),
    (
        ("en", "en:model_en"),
        ("en-US", "en:model_en"),
        ("de", "de:model_de"),
        ("de-DE", "de:model_de"),
    ),
)
def test_language_routes_only_to_matching_model(
    tmp_path: Path, language: str, component: str
) -> None:
    runtime, sessions, calls = _runtime(tmp_path)

    result = runtime.infer([5, 6], style=np.asarray([1, 2, 3]), language=language, speed=0.8)

    assert [label for _, label in calls] == [component]
    session = sessions[component]
    assert session.inputs is not None
    np.testing.assert_array_equal(session.inputs["input_ids"], [[0, 5, 6, 0]])
    assert session.inputs["input_ids"].dtype == np.int32
    assert session.inputs["style"].shape == (1, 3)
    assert session.inputs["style"].dtype == np.float16
    np.testing.assert_allclose(session.inputs["speed"], [0.8])
    assert result.sample_rate == 24000
    assert result.audio.dtype == np.float32
    np.testing.assert_array_equal(result.timings, [2, 3])
    assert set(result.outputs) == {"pred_dur"}


@pytest.mark.parametrize("language", (None, "", "fr"))
def test_language_is_required_and_must_be_supported(tmp_path: Path, language: str | None) -> None:
    runtime, _, calls = _runtime(tmp_path)

    with pytest.raises(RuntimeContractError, match="language"):
        runtime.infer([1], style=np.ones(3), language=language)

    assert calls == []


def test_static_style_is_required_and_reference_is_rejected(tmp_path: Path) -> None:
    runtime, _, calls = _runtime(tmp_path)

    with pytest.raises(RuntimeContractError, match="style"):
        runtime.infer([1], language="en")
    with pytest.raises(RuntimeContractError, match="Reference conditioning"):
        runtime.infer([1], style=np.ones(3), reference=object(), language="en")

    assert calls == []


def test_token_limit_comes_from_selected_language_graph(tmp_path: Path) -> None:
    runtime, _, calls = _runtime(tmp_path)

    with pytest.raises(RuntimeContractError, match="maximum is 2"):
        runtime.infer([1, 2, 3], style=np.ones(3), language="en")

    assert [label for _, label in calls] == ["en:model_en"]


def test_quality_selects_component_artifact_and_sessions_are_lazy(tmp_path: Path) -> None:
    runtime, sessions, calls = _runtime(
        tmp_path,
        qualities={"model_en": "q1,q2", "model_de": "q1,q2"},
        selected_quality="q2",
    )

    runtime.infer([1], style=np.ones(3), language="en")

    assert calls == [(tmp_path / "model_en-q2.onnx", "en:model_en")]
    assert set(sessions) == {"en:model_en"}


def test_diagnostics_label_language_and_component_and_close_all_sessions(tmp_path: Path) -> None:
    runtime, sessions, _ = _runtime(tmp_path)
    runtime.infer([1], style=np.ones(3), language="en")
    runtime.infer([1], style=np.ones(3), language="de")

    diagnostics = runtime.diagnostics()
    assert diagnostics.properties["language_models"] == {"en": "model_en", "de": "model_de"}
    assert {session.component for session in diagnostics.sessions} == {
        "en:model_en",
        "de:model_de",
    }

    runtime.close()

    assert {session.close_count for session in sessions.values()} == {1}


@pytest.mark.parametrize(
    ("runtime", "message"),
    (
        ({"language_models": {"en": "model_de", "de": "model_en"}}, "language_models"),
        ({"language_codes": ["en"]}, "language_codes"),
    ),
)
def test_invalid_language_router_metadata_is_rejected(
    tmp_path: Path, runtime: dict[str, Any], message: str
) -> None:
    with pytest.raises(RuntimeContractError, match=message):
        _runtime(tmp_path, runtime=runtime)


def test_missing_component_artifact_is_reported(tmp_path: Path) -> None:
    runtime = KokoroLanguageRouterRuntime(
        Installation(
            "kokoro",
            "akinvox-multispeaker-v1",
            "model",
            tmp_path,
            (),
            metadata={
                "runtime": {
                    "layout": "language-router-onnx-v1",
                    "language_models": {"en": "model_en", "de": "model_de"},
                    "language_codes": ["en", "de"],
                }
            },
        ),
        session_factory=lambda *_: FakeSession("unused"),
    )

    with pytest.raises(CapabilityError, match="model_en"):
        runtime.infer([1], style=np.ones(3), language="en")
