from __future__ import annotations

try:
    from ._version import __version__
except ImportError:  # source checkout before setuptools_scm has generated _version.py
    try:
        from importlib.metadata import version

        __version__ = version("onnxvoice")
    except Exception:
        __version__ = "0.0.0"


from typing import Any

from .catalog import CatalogClient
from .errors import NotInstalledError
from .manager import OnnxVoice
from .runtime import OnnxSession, available_providers
from .store import AssetStore
from .systems import SplitKokoroRuntime, register_adapter, registered_systems
from .types import (
    Artifact,
    AssetProgress,
    CatalogItem,
    InferenceResult,
    Installation,
    InstalledArtifact,
    RuntimeDiagnostic,
    SessionDiagnostic,
    TensorSpec,
)

_default: OnnxVoice | None = None


def _manager() -> OnnxVoice:
    global _default
    if _default is None:
        _default = OnnxVoice()
    return _default


def install(ref: str, **kwargs) -> Installation:
    return _manager().install(ref, **kwargs)


def open(ref: str | Installation, **kwargs: Any):
    return _manager().open(ref, **kwargs)


def open_local(**kwargs: Any):
    return OnnxVoice.open_local(**kwargs)


def resolve(ref: str, **kwargs) -> Installation:
    return _manager().resolve(ref, **kwargs)


def load(ref: str | Installation, **kwargs: Any):
    if kwargs.get("download") is True:
        raise ValueError("load(download=True) is obsolete; call install() before open()")
    return _manager().load(ref, **kwargs)


def load_local(**kwargs: Any):
    return OnnxVoice.load_local(**kwargs)


def installed(system: str | None = None) -> list[Installation]:
    return _manager().installed(system)


def where(ref: str, **kwargs):
    return _manager().where(ref, **kwargs)


def remove(ref: str, **kwargs):
    return _manager().remove(ref, **kwargs)


__all__ = [
    "__version__",
    "OnnxVoice",
    "CatalogClient",
    "AssetStore",
    "OnnxSession",
    "Artifact",
    "AssetProgress",
    "InstalledArtifact",
    "NotInstalledError",
    "InferenceResult",
    "TensorSpec",
    "RuntimeDiagnostic",
    "SessionDiagnostic",
    "CatalogItem",
    "Installation",
    "available_providers",
    "register_adapter",
    "registered_systems",
    "SplitKokoroRuntime",
    "verify_installation",
    "validate_onnx",
    "validate_audio",
    "install",
    "open",
    "open_local",
    "resolve",
    "load",
    "load_local",
    "installed",
    "where",
    "remove",
]
