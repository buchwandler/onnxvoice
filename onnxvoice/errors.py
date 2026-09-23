class OnnxVoiceError(Exception):
    """Base exception for onnxvoice."""


class VoiceSelectorError(OnnxVoiceError):
    """Base error for short voice selector parsing and resolution."""


class VoiceSelectorNotFoundError(VoiceSelectorError):
    """A syntactically valid selector or identity is not registered."""


class VoiceSelectorRetiredError(VoiceSelectorError):
    """A selector is known but its assigned identity is retired."""


class VoiceSelectorRegistryError(VoiceSelectorError):
    """The authoritative voice selector registry is invalid."""


class CatalogError(OnnxVoiceError):
    """Catalog loading or resolution failed."""


class AssetNotFoundError(OnnxVoiceError):
    """Requested model or voice does not exist."""


class NotInstalledError(AssetNotFoundError):
    """Requested model or voice is not installed locally."""


class IntegrityError(OnnxVoiceError):
    """Downloaded or cached content failed integrity validation."""


class AssetDownloadError(OnnxVoiceError):
    """A remote asset could not be downloaded."""


class PredefinedVoiceError(OnnxVoiceError):
    """A predefined voice state could not be resolved or loaded."""


class PredefinedVoiceAccessError(PredefinedVoiceError):
    """A gated predefined voice asset requires authorization."""


class PredefinedVoiceNotFoundError(PredefinedVoiceError):
    """A pinned predefined voice asset does not exist."""


class PredefinedVoiceIntegrityError(PredefinedVoiceError):
    """A predefined voice asset is corrupt or incompatible with its bundle."""


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
