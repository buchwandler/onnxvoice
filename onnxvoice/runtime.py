from __future__ import annotations

import importlib
import os
from collections.abc import Mapping, Sequence
from importlib import metadata
from pathlib import Path
from typing import Any, Literal

from .errors import OptionalDependencyError, RuntimeContractError
from .runtime_config import (
    clear_runtime_config,
    load_runtime_config,
    save_runtime_config,
)
from .types import (
    OnnxRuntimeStatus,
    ProviderPreference,
    RuntimeDistributionInfo,
    SessionDiagnostic,
    TensorSpec,
)

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
    "nnapi": "NnapiExecutionProvider",
    "xnnpack": "XnnpackExecutionProvider",
}
_AUTO_PRIORITY = (
    "TensorrtExecutionProvider",
    "CUDAExecutionProvider",
    "ROCMExecutionProvider",
    "DmlExecutionProvider",
    "OpenVINOExecutionProvider",
    "CoreMLExecutionProvider",
    "XnnpackExecutionProvider",
    "NnapiExecutionProvider",
    "CPUExecutionProvider",
)


def _ort() -> Any:
    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise OptionalDependencyError(
            "ONNX Runtime is not installed. Install one of onnxvoice[cpu], "
            "onnxvoice[gpu], onnxvoice[directml], or onnxvoice[openvino]. "
            "CoreML, NNAPI, and XNNPACK are provided by compatible ONNX Runtime "
            "builds on supported platforms."
        ) from exc
    return ort


def available_providers() -> tuple[str, ...]:
    """Return providers reported by the imported ONNX Runtime build."""
    return tuple(_ort().get_available_providers())


def normalize_provider_name(provider: str) -> str:
    """Normalize a user-friendly provider alias to its ONNX Runtime name."""
    if not isinstance(provider, str) or not provider.strip():
        raise RuntimeContractError("Provider names must be non-empty strings")
    name = provider.strip()
    return _PROVIDER_ALIASES.get(name.lower(), name)


def _requested_provider_names(providers: str | Sequence[str]) -> tuple[str, ...]:
    if isinstance(providers, str):
        names = tuple(part.strip() for part in providers.split(",") if part.strip())
    else:
        try:
            names = tuple(
                provider.strip() if isinstance(provider, str) else provider
                for provider in providers
            )
        except TypeError as exc:
            raise RuntimeContractError(
                "Providers must be a string or a sequence of strings"
            ) from exc
    if not names:
        raise RuntimeContractError("At least one ONNX Runtime provider is required")
    if any(not isinstance(name, str) or not name.strip() for name in names):
        raise RuntimeContractError("Provider names must be non-empty strings")
    return tuple(name.strip() for name in names)


def provider_preference(
    providers: str | Sequence[str] | None = None,
    *,
    environment: Mapping[str, str] | None = None,
) -> ProviderPreference:
    """Capture the configured provider policy without importing ONNX Runtime."""
    env = os.environ if environment is None else environment
    source: Literal[
        "explicit", "ONNXVOICE_PROVIDERS", "ONNXVOICE_PROVIDER", "user_config", "default"
    ]
    if providers is not None:
        requested: str | Sequence[str] = providers
        source = "explicit"
    elif env.get("ONNXVOICE_PROVIDERS"):
        requested = env["ONNXVOICE_PROVIDERS"]
        source = "ONNXVOICE_PROVIDERS"
    elif env.get("ONNXVOICE_PROVIDER"):
        requested = env["ONNXVOICE_PROVIDER"]
        source = "ONNXVOICE_PROVIDER"
    else:
        saved = load_runtime_config()
        if saved is not None:
            requested = saved
            source = "user_config"
        else:
            requested = "cpu"
            source = "default"
    return ProviderPreference(_requested_provider_names(requested), source)


def get_default_providers(*, environment: Mapping[str, str] | None = None) -> ProviderPreference:
    """Return the effective default provider policy and its source."""
    return provider_preference(environment=environment)


def resolve_provider_preference(
    preference: ProviderPreference, *, available: Sequence[str]
) -> tuple[str, ...]:
    """Resolve a captured provider policy against providers reported by ONNX Runtime."""
    requested_names = _requested_provider_names(preference.requested)
    available_names = tuple(available)
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


def resolve_providers(
    providers: str | Sequence[str] | None = None,
    *,
    available: Sequence[str] | None = None,
    environment: Mapping[str, str] | None = None,
) -> tuple[str, ...]:
    """Resolve an explicit, environmental, saved, or default provider policy."""
    preference = provider_preference(providers, environment=environment)
    available_names = available if available is not None else available_providers()
    return resolve_provider_preference(preference, available=available_names)


def _safe_exception_text(exc: BaseException) -> str:
    message = " ".join(str(exc).split())
    if len(message) > 300:
        message = message[:297] + "..."
    return f"{type(exc).__name__}: {message}" if message else type(exc).__name__


def _module_path(module: Any | None) -> Path | None:
    value = getattr(module, "__file__", None) if module is not None else None
    if isinstance(value, os.PathLike):
        value = os.fspath(value)
    if not isinstance(value, str):
        return None
    try:
        return Path(value).resolve()
    except (OSError, RuntimeError, TypeError, ValueError):
        return Path(value)


def _distribution_candidates() -> tuple[str, ...]:
    names: list[str] = []
    try:
        mapping = metadata.packages_distributions()
        mapped = mapping.get("onnxruntime", ()) if isinstance(mapping, Mapping) else ()
        names.extend(name for name in mapped if isinstance(name, str) and name)
    except Exception:
        pass
    names.extend(
        (
            "onnxruntime",
            "onnxruntime-gpu",
            "onnxruntime-directml",
            "onnxruntime-openvino",
        )
    )
    unique: dict[str, str] = {}
    for name in names:
        unique.setdefault(name.casefold().replace("_", "-"), name)
    return tuple(unique.values())


def _runtime_distributions(
    module_path: Path | None,
) -> tuple[RuntimeDistributionInfo, ...]:
    found: dict[str, RuntimeDistributionInfo] = {}
    for candidate in _distribution_candidates():
        try:
            distribution = metadata.distribution(candidate)
        except Exception:
            continue
        try:
            raw_name = distribution.metadata.get("Name")
        except Exception:
            raw_name = None
        name = raw_name if isinstance(raw_name, str) and raw_name else candidate
        key = name.casefold().replace("_", "-")
        try:
            version = distribution.version
            version = str(version) if version is not None else None
        except Exception:
            version = None
        try:
            location = Path(str(distribution.locate_file(""))).resolve()
        except Exception:
            location = None

        try:
            files = distribution.files
        except Exception:
            files = None
        owns_import: bool | None = None
        if module_path is not None and files is not None:
            owns_import = False
            for file in files:
                parts = Path(str(file)).parts
                if len(parts) < 2 or parts[-2:] != ("onnxruntime", "__init__.py"):
                    continue
                try:
                    recorded_path = Path(str(distribution.locate_file(file))).resolve()
                except Exception:
                    continue
                if recorded_path == module_path:
                    owns_import = True
                    break

        found.setdefault(
            key,
            RuntimeDistributionInfo(
                name=name,
                version=version,
                location=location,
                owns_import=owns_import,
            ),
        )
    return tuple(sorted(found.values(), key=lambda item: item.name.casefold()))


def runtime_distributions() -> tuple[RuntimeDistributionInfo, ...]:
    """Discover distributions that may supply the imported ``onnxruntime`` module."""
    try:
        module = importlib.import_module("onnxruntime")
    except Exception:
        module = None
    return _runtime_distributions(_module_path(module))


def _safe_import_ort() -> tuple[Any | None, str | None]:
    try:
        return importlib.import_module("onnxruntime"), None
    except Exception as exc:
        return None, _safe_exception_text(exc)


def _optional_runtime_value(module: Any, name: str) -> Any | None:
    try:
        value = getattr(module, name, None)
        return value() if callable(value) else value
    except Exception:
        return None


def _optional_text(module: Any, name: str) -> str | None:
    value = _optional_runtime_value(module, name)
    return str(value) if value is not None else None


def runtime_status(*, environment: Mapping[str, str] | None = None) -> OnnxRuntimeStatus:
    """Inspect ONNX Runtime and provider policy without making inference or network calls."""
    preference_error: str | None = None
    try:
        preference = get_default_providers(environment=environment)
    except RuntimeContractError as exc:
        preference = ProviderPreference((), "user_config")
        preference_error = str(exc)

    module, import_error = _safe_import_ort()
    module_path = _module_path(module)
    distributions = _runtime_distributions(module_path)
    state: Literal["resolved", "ambiguous", "vendor", "not_installed", "import_failed"]
    if module is None:
        state = "import_failed" if distributions else "not_installed"
        return OnnxRuntimeStatus(
            importable=False,
            module_version=None,
            module_path=None,
            build_device=None,
            build_info=None,
            distributions=distributions,
            imported_distribution=None,
            distribution_state=state,
            available_providers=(),
            configured_providers=preference.requested,
            provider_source=preference.source,
            resolved_providers=(),
            import_error=import_error,
            resolution_error=preference_error,
        )

    owners = [item for item in distributions if item.owns_import is True]
    if len(owners) == 1:
        imported_distribution = owners[0].name
        state = "resolved"
    elif len(owners) > 1:
        imported_distribution = None
        state = "ambiguous"
    else:
        imported_distribution = None
        state = "vendor"

    module_version = _optional_text(module, "__version__")
    if module_version is None:
        module_version = _optional_text(module, "get_version_string")
    build_device = _optional_text(module, "get_device")
    build_info = _optional_text(module, "get_build_info")

    resolution_error = preference_error
    available: tuple[str, ...] = ()
    provider_getter = getattr(module, "get_available_providers", None)
    if not callable(provider_getter):
        provider_error = "Imported ONNX Runtime does not expose get_available_providers()."
    else:
        try:
            available = tuple(str(name) for name in provider_getter())
            provider_error = None
        except Exception as exc:
            provider_error = f"Could not read ONNX Runtime providers: {_safe_exception_text(exc)}"
    if resolution_error is None and provider_error is not None:
        resolution_error = provider_error

    resolved: tuple[str, ...] = ()
    if resolution_error is None:
        try:
            resolved = resolve_provider_preference(preference, available=available)
        except RuntimeContractError as exc:
            resolution_error = str(exc)

    return OnnxRuntimeStatus(
        importable=True,
        module_version=module_version,
        module_path=module_path,
        build_device=build_device,
        build_info=build_info,
        distributions=distributions,
        imported_distribution=imported_distribution,
        distribution_state=state,
        available_providers=available,
        configured_providers=preference.requested,
        provider_source=preference.source,
        resolved_providers=resolved,
        import_error=None,
        resolution_error=resolution_error,
    )


def set_default_providers(providers: str | Sequence[str]) -> ProviderPreference:
    """Persist a validated policy for future ONNXVoice sessions."""
    preference = provider_preference(providers, environment={})
    resolve_provider_preference(preference, available=available_providers())
    save_runtime_config(preference.requested)
    return ProviderPreference(preference.requested, "user_config")


def clear_default_providers() -> None:
    """Remove the persisted ONNXVoice provider policy."""
    clear_runtime_config()


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
    """Create ONNX Runtime sessions lazily and expose graph metadata.


    Provider selection is resolved when a session is first needed. Importing this
    module does not require the optional ``onnxruntime`` package.
    """

    def __init__(
        self,
        model: str | Path,
        *,
        component: str | None = None,
        providers: str | Sequence[str] | None = None,
        provider_options: Sequence[dict[str, Any]] | Mapping[str, dict[str, Any]] | None = None,
        session_options: Any | None = None,
        strict_providers: bool = False,
    ) -> None:
        """Configure a model path and optional provider policy."""
        self.model = Path(model)
        self.component = component
        self.provider_request = providers
        self._provider_preference = provider_preference(providers)
        self.providers = (
            tuple(providers) if not isinstance(providers, str) and providers else providers
        )
        self.provider_options = provider_options
        self.session_options = session_options
        self.strict_providers = strict_providers
        self._session: Any | None = None
        self._resolved_providers: tuple[str, ...] | None = None
        self._active_providers: tuple[str, ...] | None = None

    @property
    def session(self) -> Any:
        """Create and return the ONNX Runtime session on first access."""
        if self._session is None:
            self._session = self._create()
        return self._session

    @property
    def input_names(self) -> tuple[str, ...]:
        """Return input tensor names, creating the session if necessary."""
        return tuple(node.name for node in self.session.get_inputs())

    @property
    def output_names(self) -> tuple[str, ...]:
        """Return output tensor names, creating the session if necessary."""
        return tuple(node.name for node in self.session.get_outputs())

    @property
    def input_specs(self) -> tuple[TensorSpec, ...]:
        """Return input names, runtime types, and shapes."""
        return self._tensor_specs(self.session.get_inputs())

    @property
    def output_specs(self) -> tuple[TensorSpec, ...]:
        """Return output names, runtime types, and shapes."""
        return self._tensor_specs(self.session.get_outputs())

    @property
    def resolved_providers(self) -> tuple[str, ...]:
        """Return the selected provider after session initialization."""
        _ = self.session
        assert self._resolved_providers is not None
        return self._resolved_providers

    @property
    def active_providers(self) -> tuple[str, ...]:
        """Return the providers actually active in the initialized session."""
        _ = self.session
        assert self._active_providers is not None
        return self._active_providers

    def _create(self) -> Any:
        if not self.model.is_file():
            raise FileNotFoundError(self.model)
        ort = _ort()
        selected = resolve_provider_preference(
            self._provider_preference, available=ort.get_available_providers()
        )
        kwargs: dict[str, Any] = {"providers": list(selected)}
        options = provider_options_for(selected, self.provider_options)
        if options is not None:
            kwargs["provider_options"] = options
        session_options = self.session_options
        if self.strict_providers and "CPUExecutionProvider" not in selected:
            try:
                if session_options is None:
                    session_options = ort.SessionOptions()
                add_config_entry = getattr(session_options, "add_session_config_entry", None)
                if not callable(add_config_entry):
                    raise RuntimeContractError(
                        "Strict provider policy requires session_options.add_session_config_entry"
                    )
                add_config_entry("session.disable_cpu_ep_fallback", "1")
            except RuntimeContractError:
                raise
            except Exception as exc:
                raise RuntimeContractError(
                    f"Could not enforce strict ONNX provider policy: {exc}"
                ) from exc
        if session_options is not None:
            kwargs["sess_options"] = session_options
        try:
            session = ort.InferenceSession(str(self.model), **kwargs)
        except Exception as exc:
            raise RuntimeContractError(f"Could not create ONNX Runtime session: {exc}") from exc
        if self.strict_providers:
            disable_fallback = getattr(session, "disable_fallback", None)
            if not callable(disable_fallback):
                raise RuntimeContractError(
                    "ONNX Runtime session does not support strict provider fallback control"
                )
            try:
                disable_fallback()
            except Exception as exc:
                raise RuntimeContractError(
                    f"Could not disable ONNX provider fallback: {exc}"
                ) from exc
        self._resolved_providers = selected
        get_providers = getattr(session, "get_providers", None)
        active = tuple(get_providers()) if callable(get_providers) else selected
        self._active_providers = active or selected
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
        """Run inference with a mapping of graph input names to values."""
        return self.session.run(None, inputs)

    def diagnostics(self) -> SessionDiagnostic:
        """Return provider and input/output diagnostics for this session."""
        return SessionDiagnostic(
            component=self.component,
            model_path=self.model,
            providers_requested=tuple(
                normalize_provider_name(name) for name in self._provider_preference.requested
            ),
            providers_active=self.active_providers,
            inputs=self.input_names,
            outputs=self.output_names,
            providers_resolved=self.resolved_providers,
            provider_source=self._provider_preference.source,
        )

    def close(self) -> None:
        """Release the current session so it can be recreated on next access."""
        self._session = None
        self._resolved_providers = None
        self._active_providers = None
