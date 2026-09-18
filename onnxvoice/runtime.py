from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .errors import OptionalDependencyError, RuntimeContractError
from .types import TensorSpec

_PROVIDER_ALIASES = {
    "cpu": "CPUExecutionProvider",
    "cuda": "CUDAExecutionProvider",
    "gpu": "CUDAExecutionProvider",
    "tensorrt": "TensorrtExecutionProvider",
    "directml": "DmlExecutionProvider",
    "dml": "DmlExecutionProvider",
    "openvino": "OpenVINOExecutionProvider",
    "rocm": "ROCMExecutionProvider",
    "coreml": "CoreMLExecutionProvider",
}
_AUTO_PRIORITY = (
    "TensorrtExecutionProvider",
    "CUDAExecutionProvider",
    "ROCMExecutionProvider",
    "DmlExecutionProvider",
    "OpenVINOExecutionProvider",
    "CoreMLExecutionProvider",
    "CPUExecutionProvider",
)


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


def normalize_provider_name(provider: str) -> str:
    if not isinstance(provider, str) or not provider.strip():
        raise RuntimeContractError("Provider names must be non-empty strings")
    name = provider.strip()
    return _PROVIDER_ALIASES.get(name.lower(), name)


def resolve_providers(
    providers: str | Sequence[str] | None = None,
    *,
    available: Sequence[str] | None = None,
    environment: Mapping[str, str] | None = None,
) -> tuple[str, ...]:
    """Resolve aliases and automatic provider policy without importing ORT."""
    env = os.environ if environment is None else environment
    requested: str | Sequence[str] | None = providers
    if requested is None:
        requested = env.get("ONNXVOICE_PROVIDERS") or env.get("ONNXVOICE_PROVIDER") or "cpu"
    if isinstance(requested, str):
        requested_names = tuple(part.strip() for part in requested.split(",") if part.strip())
    else:
        requested_names = tuple(requested)
    if not requested_names:
        raise RuntimeContractError("At least one ONNX Runtime provider is required")

    available_names = tuple(available if available is not None else available_providers())
    available_set = set(available_names)
    if any(name.lower() == "auto" for name in requested_names):
        selected = next(
            (provider for provider in _AUTO_PRIORITY if provider in available_set), None
        )
        if selected is None:
            raise RuntimeContractError(
                "Automatic provider selection found no usable provider, "
                f"available: {', '.join(available_names)}"
            )
        return (selected,)

    resolved = tuple(normalize_provider_name(name) for name in requested_names)
    missing = [provider for provider in resolved if provider not in available_set]
    if missing:
        raise RuntimeContractError(
            f"Requested ONNX Runtime provider(s) unavailable: {', '.join(missing)}. "
            f"Available: {', '.join(sorted(available_set))}"
        )
    return resolved


def provider_options_for(
    providers: Sequence[str],
    options: Sequence[dict[str, Any]] | Mapping[str, dict[str, Any]] | None,
) -> list[dict[str, Any]] | None:
    if options is None:
        return None
    if isinstance(options, Mapping):
        normalized = {normalize_provider_name(name): value for name, value in options.items()}
        return [dict(normalized.get(provider, {})) for provider in providers]
    values = [dict(value) for value in options]
    if len(values) != len(providers):
        raise RuntimeContractError(
            "provider_options must contain one mapping for each selected provider"
        )
    return values


class OnnxSession:
    def __init__(
        self,
        model: str | Path,
        *,
        providers: str | Sequence[str] | None = None,
        provider_options: Sequence[dict[str, Any]] | Mapping[str, dict[str, Any]] | None = None,
        session_options: Any | None = None,
    ) -> None:
        self.model = Path(model)
        self.provider_request = providers
        self.providers = (
            tuple(providers) if not isinstance(providers, str) and providers else providers
        )
        self.provider_options = provider_options
        self.session_options = session_options
        self._session: Any | None = None
        self._resolved_providers: tuple[str, ...] | None = None

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

    @property
    def input_specs(self) -> tuple[TensorSpec, ...]:
        return self._tensor_specs(self.session.get_inputs())

    @property
    def output_specs(self) -> tuple[TensorSpec, ...]:
        return self._tensor_specs(self.session.get_outputs())

    @property
    def resolved_providers(self) -> tuple[str, ...]:
        _ = self.session
        assert self._resolved_providers is not None
        return self._resolved_providers

    def _create(self) -> Any:
        if not self.model.is_file():
            raise FileNotFoundError(self.model)
        ort = _ort()
        selected = resolve_providers(self.provider_request, available=ort.get_available_providers())
        kwargs: dict[str, Any] = {"providers": list(selected)}
        options = provider_options_for(selected, self.provider_options)
        if options is not None:
            kwargs["provider_options"] = options
        if self.session_options is not None:
            kwargs["sess_options"] = self.session_options
        try:
            session = ort.InferenceSession(str(self.model), **kwargs)
        except Exception as exc:
            raise RuntimeContractError(f"Could not create ONNX Runtime session: {exc}") from exc
        self._resolved_providers = selected
        return session

    @staticmethod
    def _tensor_specs(nodes: Sequence[Any]) -> tuple[TensorSpec, ...]:
        return tuple(
            TensorSpec(
                name=node.name,
                ort_type=str(node.type),
                shape=tuple(node.shape) if node.shape is not None else (),
            )
            for node in nodes
        )

    def run(self, inputs: dict[str, Any]) -> list[Any]:
        return self.session.run(None, inputs)

    def close(self) -> None:
        self._session = None
