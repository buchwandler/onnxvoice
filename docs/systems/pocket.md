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

## Generation, EOS, and temperature

OnnxVoice applies Pocket's minimum-EOS guard: EOS observations before generation step 6 (zero-based) are ignored for stopping. The first accepted boundary is reported as `eos_step`; `eos_observed` indicates whether any EOS signal was seen, including ignored early signals.

An explicit non-negative integer `frames_after_eos` is measured from that first accepted step. The stop check happens before appending the boundary frame: `0` omits the accepted-EOS frame, `1` includes it, and larger values extend the generated tail by the same step distance. For direct low-level callers, `None` retains the backward-compatible behavior of appending the accepted-EOS frame and stopping. OnnxVoice does not infer a text-length policy from token IDs; text-aware layers such as PocketSynth should resolve automatic tail selection to a concrete integer before calling OnnxVoice for normal text synthesis.

Pocket latent noise uses a standard deviation of `sqrt(temperature)`, matching Pocket TTS semantics. The `temperature=0.7` in the example above is the current OnnxVoice API default, not a recommendation or an assertion that it matches a model-specific upstream default. Inference metadata includes `frames_generated`, `eos_step`, `frames_after_eos`, and `min_frames_before_eos` for diagnostics.

Predefined state downloads are a separate optional asset operation. Pocket voice discovery reflects the IDs already present in normalized `CatalogItem.voices`; it does not infer voices from other descriptive or predefined names.

## Open a local bundle

For local assets, supply the bundle files and metadata explicitly with `OnnxVoice.open_local(system="pocket", files=..., metadata=...)`. See [Local models](../local-models.md) and [Pocket downloads](../pocket-downloads.md) for the managed and offline boundaries.
