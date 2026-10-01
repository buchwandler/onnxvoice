from onnxvoice import OnnxVoice, VoiceNotFoundError, VoiceRecord, resolve_voice


def resolve_with_manager(manager: OnnxVoice, ref: str) -> VoiceRecord:
    try:
        return manager.resolve_voice(ref)
    except VoiceNotFoundError:
        raise


def resolve_from_package(ref: str) -> VoiceRecord:
    return resolve_voice(ref)
