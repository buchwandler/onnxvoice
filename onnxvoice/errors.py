class OnnxVoiceError(Exception):
    """Base exception for onnxvoice."""


class CatalogError(OnnxVoiceError):
    """Catalog loading or resolution failed."""


class AssetNotFoundError(OnnxVoiceError):
    """Requested model or voice does not exist."""


class IntegrityError(OnnxVoiceError):
    """Downloaded or cached content failed integrity validation."""


class OfflineError(OnnxVoiceError):
    """An operation requires network access while offline mode is active."""


class OptionalDependencyError(OnnxVoiceError):
    """An optional runtime dependency is missing."""


class UnsupportedSystemError(OnnxVoiceError):
    """No adapter exists for the requested TTS system."""


class RuntimeContractError(OnnxVoiceError):
    """An ONNX model does not match the expected system contract."""
