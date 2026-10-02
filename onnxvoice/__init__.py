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
from .errors import NotInstalledError, VoiceNotFoundError
from .inventory import (
    language_base,
    language_tags_match,
    normalize_language_code,
    normalize_language_tag,
)
from .manager import OnnxVoice
from .runtime import OnnxSession, available_providers
from .store import AssetStore
from .systems import (
    KokoroCloningRuntime,
    KokoroReferenceState,
    SplitKokoroRuntime,
    register_adapter,
    registered_systems,
)
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
    VoiceMetadata,
    VoiceRecord,
)
from .validation import validate_audio, validate_onnx, verify_installation

_default: OnnxVoice | None = None


def _manager() -> OnnxVoice:
    global _default
    if _default is None:
        _default = OnnxVoice()
    return _default


def install(ref: str, **kwargs) -> Installation:
    """Install a catalog asset into the shared store.


    Catalog resolution and asset acquisition may perform network I/O.
    """
    return _manager().install(ref, **kwargs)


def open(ref: str | Installation, **kwargs: Any):
    """Open an existing managed installation without acquiring assets.


    Install the reference first with :func:`install` when it is not already stored.
    """
    return _manager().open(ref, **kwargs)


def open_local(**kwargs: Any):
    """Open explicit local model files without registering or copying them.


    See :meth:`OnnxVoice.open_local` for supported file mappings.
    """
    return OnnxVoice.open_local(**kwargs)


def resolve_voice(ref: str, **kwargs: Any) -> VoiceRecord:
    """Resolve a semantic voice reference from the normalized catalog."""
    return _manager().resolve_voice(ref, **kwargs)


def resolve(ref: str, **kwargs) -> Installation:
    """Resolve and verify an existing local installation without catalog access."""
    return _manager().resolve(ref, **kwargs)


def load(ref: str | Installation, **kwargs: Any):
    """Deprecated alias for :func:`open`; this function does not install assets."""
    if kwargs.get("download") is True:
        raise ValueError("load(download=True) is obsolete; call install() before open()")
    return _manager().load(ref, **kwargs)


def load_local(**kwargs: Any):
    """Deprecated alias for :func:`open_local`."""
    return OnnxVoice.load_local(**kwargs)


def installed(system: str | None = None) -> list[Installation]:
    """List locally installed assets, optionally restricted to one system."""
    return _manager().installed(system)


def where(ref: str, **kwargs):
    """Return the path of a verified local installation."""
    return _manager().where(ref, **kwargs)


def remove(ref: str, **kwargs):
    """Remove an installation; shared blobs may remain until garbage collection."""
    return _manager().remove(ref, **kwargs)


__all__ = [
    "__version__",
    "OnnxVoice",
    "CatalogClient",
    "language_base",
    "language_tags_match",
    "normalize_language_code",
    "normalize_language_tag",
    "AssetStore",
    "OnnxSession",
    "Artifact",
    "AssetProgress",
    "InstalledArtifact",
    "NotInstalledError",
    "VoiceNotFoundError",
    "InferenceResult",
    "TensorSpec",
    "RuntimeDiagnostic",
    "SessionDiagnostic",
    "CatalogItem",
    "Installation",
    "available_providers",
    "VoiceRecord",
    "VoiceMetadata",
    "register_adapter",
    "registered_systems",
    "SplitKokoroRuntime",
    "KokoroReferenceState",
    "KokoroCloningRuntime",
    "verify_installation",
    "validate_onnx",
    "validate_audio",
    "install",
    "open",
    "open_local",
    "resolve",
    "resolve_voice",
    "load",
    "load_local",
    "installed",
    "where",
    "remove",
]
