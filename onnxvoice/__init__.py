from __future__ import annotations

try:
    from ._version import __version__
except ImportError:  # source checkout before setuptools_scm has generated _version.py
    try:
        from importlib.metadata import version

        __version__ = version("onnxvoice")
    except Exception:
        __version__ = "0.0.0"

from .catalog import CatalogClient
from .manager import OnnxVoice
from .runtime import OnnxSession, available_providers
from .store import AssetStore
from .systems import register_adapter, registered_systems
from .types import (
    AssetProgress,
    AudioResult,
    CatalogItem,
    InferenceResult,
    Installation,
    TensorSpec,
)
from .validation import validate_audio, validate_onnx, verify_installation

_default: OnnxVoice | None = None


def _manager() -> OnnxVoice:
    global _default
    if _default is None:
        _default = OnnxVoice()
    return _default


def install(ref: str, **kwargs) -> Installation:
    return _manager().install(ref, **kwargs)


def resolve(ref: str, **kwargs) -> Installation:
    return _manager().resolve(ref, **kwargs)


def load(ref: str | Installation, **kwargs):
    return _manager().load(ref, **kwargs)


def load_local(**kwargs):
    return OnnxVoice.load_local(**kwargs)


def installed(system: str | None = None) -> list[Installation]:
    return _manager().installed(system)


def where(ref: str):
    return _manager().where(ref)


__all__ = [
    "__version__",
    "OnnxVoice",
    "CatalogClient",
    "AssetStore",
    "OnnxSession",
    "AssetProgress",
    "AudioResult",
    "InferenceResult",
    "TensorSpec",
    "CatalogItem",
    "Installation",
    "available_providers",
    "register_adapter",
    "registered_systems",
    "verify_installation",
    "validate_onnx",
    "validate_audio",
    "install",
    "resolve",
    "load",
    "load_local",
    "installed",
    "where",
]
