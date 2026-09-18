class OnnxVoiceError(Exception):
    """Base exception for onnxvoice."""


class CatalogError(OnnxVoiceError):
    """Catalog loading or resolution failed."""


class AssetNotFoundError(OnnxVoiceError):
    """Requested model or voice does not exist."""


class NotInstalledError(AssetNotFoundError):
    """Requested model or voice is not installed locally."""


class IntegrityError(OnnxVoiceError):
    """Downloaded or cached content failed integrity validation."""


class ManifestError(OnnxVoiceError):
    """An installation manifest is malformed or unsupported."""


class UnsafePathError(ManifestError):
    """A user-controlled path component is unsafe."""


class LockError(OnnxVoiceError):
    """A process lock could not be acquired or released."""


class OfflineError(OnnxVoiceError):
    """An operation requires network access while offline mode is active."""


class OptionalDependencyError(OnnxVoiceError):
    """An optional runtime dependency is missing."""


class UnsupportedSystemError(OnnxVoiceError):
    """No adapter exists for the requested TTS system."""


class RuntimeContractError(OnnxVoiceError):
    """An ONNX model does not match the expected system contract."""


class CapabilityError(RuntimeContractError):
    """The requested model layout or runtime capability is unsupported."""
