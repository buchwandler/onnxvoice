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
from .types import AudioResult, CatalogItem, Installation
from .validation import validate_audio, validate_onnx, verify_installation

_default = OnnxVoice()


def install(ref: str, **kwargs) -> Installation:
    return _default.install(ref, **kwargs)


def resolve(ref: str, **kwargs) -> Installation:
    return _default.resolve(ref, **kwargs)


def load(ref: str | Installation, **kwargs):
    return _default.load(ref, **kwargs)


def installed(system: str | None = None) -> list[Installation]:
    return _default.installed(system)


def where(ref: str):
    return _default.where(ref)


__all__ = [
    "__version__",
    "OnnxVoice",
    "CatalogClient",
    "AssetStore",
    "OnnxSession",
    "AudioResult",
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
    "installed",
    "where",
]
