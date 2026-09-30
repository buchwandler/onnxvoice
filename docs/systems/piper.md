# Piper

Piper runs an installed Piper ONNX model and optional JSON config. `onnxvoice` does not phonemize text or choose a speaker by name. The caller supplies model-ready token IDs and, when the graph has a `sid` input, the numeric speaker ID.

## Managed installation and inference

```python
from onnxvoice import OnnxVoice

voice = OnnxVoice()
installation = voice.install("piper:en_US-lessac-medium")
runtime = voice.open(installation, provider="cpu")
try:
    result = runtime.infer(
        [1, 20, 14, 5, 2],
        length_scale=1.0,
        noise_scale=0.667,
        noise_w=0.8,
    )
    print(result.audio.shape, result.sample_rate)
finally:
    runtime.close()
```

The runtime reads the sample rate from installation metadata or the Piper config, with a fallback when neither supplies it. A single-speaker model does not accept `speaker_id`; a multi-speaker model with a `sid` input requires it.

## Open local files

```python
from onnxvoice import OnnxVoice


runtime = OnnxVoice.open_local(
    system="piper",
    model="voice.onnx",
    config="voice.onnx.json",
    provider="cpu",
)
```

The local files are opened in place, without catalog access or store registration. See [Local models](../local-models.md) for the distinction between opening and importing.
