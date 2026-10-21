# Piper

Piper runs an installed Piper ONNX model and optional JSON config. `onnxvoice` does not phonemize text or choose a speaker by name. The caller supplies model-ready token IDs and, when the graph has a `sid` input, the numeric speaker ID.

`onnxvoice engines options --engine piper --target <voice-id>` reports the target's numeric speaker controls when known. `speaker_id` selects an existing graph speaker; it is not reference-audio cloning. Unknown model limits remain unasserted. See [CLI engine discovery](../cli.md#engine-option-and-capability-discovery).

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

## Sequence-axis contract

`runtime.phoneme_sequence_contract` reports the evidence for the active ONNX graph's Piper `input` sequence axis (`axis == 1`):

- `variable`: the graph explicitly declares a non-empty symbolic dimension;
- `fixed`: the graph declares an exact positive integer dimension;
- `unknown`: the input metadata is missing, malformed, or otherwise not authoritative.

The contract's `dimension` preserves the graph value when available, and `source` is `onnx.input.shape`. Inspecting it reads runtime graph metadata but never runs acoustic inference. `runtime.max_phonemes` remains only as a compatibility property and returns `None`: a static ONNX dimension is not treated as a maximum unless the runtime implements and documents padding for shorter sequences. OnnxVoice does not invent a practical text or phoneme limit.

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
