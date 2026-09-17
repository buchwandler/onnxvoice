from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from .errors import OptionalDependencyError, RuntimeContractError


def _ort() -> Any:
    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise OptionalDependencyError(
            "ONNX Runtime is not installed. Install onnxvoice[cpu], onnxvoice[gpu], "
            "onnxvoice[directml], or onnxvoice[openvino]."
        ) from exc
    return ort


def available_providers() -> tuple[str, ...]:
    return tuple(_ort().get_available_providers())


class OnnxSession:
    def __init__(
        self,
        model: str | Path,
        *,
        providers: Sequence[str] | None = None,
        provider_options: Sequence[dict[str, Any]] | None = None,
        session_options: Any | None = None,
    ) -> None:
        self.model = Path(model)
        self.providers = tuple(providers or ("CPUExecutionProvider",))
        self.provider_options = provider_options
        self.session_options = session_options
        self._session: Any | None = None

    @property
    def session(self) -> Any:
        if self._session is None:
            self._session = self._create()
        return self._session

    @property
    def input_names(self) -> tuple[str, ...]:
        return tuple(node.name for node in self.session.get_inputs())

    @property
    def output_names(self) -> tuple[str, ...]:
        return tuple(node.name for node in self.session.get_outputs())

    def _create(self) -> Any:
        if not self.model.is_file():
            raise FileNotFoundError(self.model)
        ort = _ort()
        available = set(ort.get_available_providers())
        missing = [provider for provider in self.providers if provider not in available]
        if missing:
            raise RuntimeContractError(
                f"Requested ONNX Runtime provider(s) unavailable: {', '.join(missing)}. "
                f"Available: {', '.join(sorted(available))}"
            )
        kwargs: dict[str, Any] = {"providers": list(self.providers)}
        if self.provider_options is not None:
            kwargs["provider_options"] = list(self.provider_options)
        if self.session_options is not None:
            kwargs["sess_options"] = self.session_options
        return ort.InferenceSession(str(self.model), **kwargs)

    def run(self, inputs: dict[str, Any]) -> list[Any]:
        return self.session.run(None, inputs)

    def close(self) -> None:
        self._session = None
