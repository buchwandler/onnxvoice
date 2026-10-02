class OnnxVoiceError(Exception):
    """Base exception for onnxvoice."""


class CatalogError(OnnxVoiceError):
    """Catalog loading or resolution failed."""


class AssetNotFoundError(OnnxVoiceError):
    """Requested model or voice does not exist."""


class NotInstalledError(AssetNotFoundError):
    """Requested model or voice is not installed locally."""


class VoiceNotFoundError(AssetNotFoundError):
    """Requested voice is not declared by its resolved catalog item."""


class IntegrityError(OnnxVoiceError):
    """Downloaded or cached content failed integrity validation."""


class AssetDownloadError(OnnxVoiceError):
    """A remote asset could not be downloaded."""


class AssetAccessError(AssetDownloadError):
    """A remote asset requires credentials or repository access."""


class AssetAuthenticationError(AssetAccessError):
    """No usable authentication is configured for a remote asset."""


class AssetPermissionError(AssetAccessError):
    """Configured credentials lack permission to access a remote asset."""


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


class VoicePromptError(OnnxVoiceError):
    """A managed Pocket voice prompt could not be parsed, resolved, or fetched."""


class InvalidVoicePromptRefError(VoicePromptError):
    """A voice prompt reference or source URL is malformed or unsafe."""


class VoicePromptFormatError(VoicePromptError):
    """A voice prompt uses an unsupported audio format."""


class VoicePromptCatalogError(VoicePromptError, CatalogError):
    """Voice prompt catalog data is invalid, unsafe, or incomplete."""


class UnknownVoicePromptError(VoicePromptError, AssetNotFoundError):
    """The requested voice prompt is not declared by the prompt catalog."""


class VoicePromptNotFoundError(VoicePromptError, AssetNotFoundError):
    """A cataloged voice prompt asset is missing at its pinned upstream revision."""


class VoicePromptOfflineError(VoicePromptError, OfflineError):
    """A voice prompt is unavailable while offline and not cached."""


class VoicePromptAccessError(VoicePromptError, AssetAccessError):
    """A voice prompt asset requires authentication or repository access."""


class VoicePromptDownloadError(VoicePromptError, AssetDownloadError):
    """A voice prompt asset could not be downloaded."""


class VoicePromptIntegrityError(VoicePromptError, IntegrityError):
    """A voice prompt asset failed size or SHA-256 verification."""
