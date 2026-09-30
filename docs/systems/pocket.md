# Pocket

Pocket is a bundle runtime with explicit graph contracts and recurrent state. Catalog assets and predefined voice-state downloads are separate concerns. A public ONNX bundle does not necessarily make every predefined voice state public.

For direct Hugging Face downloads, optional dependencies, credentials, and offline caching, see [Pocket downloads](../pocket-downloads.md). The `doctor` command reports local, non-secret configuration and does not test access to a gated repository over the network.

## Managed bundle and voice preparation

```python
from onnxvoice import OnnxVoice

voice = OnnxVoice()
installation = voice.install("pocket:english_2026-04", quality="int8")
runtime = voice.open(installation, provider="cpu")
try:
    state = runtime.prepare_voice(reference_audio, sample_rate=24000)
    result = runtime.infer(
        token_ids,
        voice_state=state,
        temperature=0.7,
        lsd_steps=1,
        max_frames=200,
    )
finally:
    runtime.close()
```

`prepare_voice()` encodes reference audio into reusable voice state. Each inference call creates fresh Flow-LM and Mimi recurrent state. `max_frames` limits generation frames, not tokens or audio samples. The adapter validates the bundle's graph inputs, outputs, state manifests, and runtime metadata.

Predefined state downloads are a separate optional asset operation. Stable Pocket selectors require explicit catalog `voice_states` identity records; a predefined voice name alone does not allocate a selector.

## Open a local bundle

For local assets, supply the bundle files and metadata explicitly with `OnnxVoice.open_local(system="pocket", files=..., metadata=...)`. See [Local models](../local-models.md) and [Pocket downloads](../pocket-downloads.md) for the managed and offline boundaries.
