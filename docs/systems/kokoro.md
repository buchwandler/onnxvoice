# Kokoro

Kokoro runs either a single ONNX model or the `split-onnx-v1` multi-component layout. The caller supplies token IDs and a model-ready style tensor. Resolving a stable voice selector does not construct or choose that tensor.

## Quality and distribution

A quality selection chooses a model variant within a distribution. It does not install every quality variant. Non-model runtime artifacts required by the selected distribution are installed with it. Explicit quality and distribution selections can have separate installation identities.

## Single-model runtime

```python
from onnxvoice import OnnxVoice

voice = OnnxVoice()

runtime = voice.open("kokoro:v1.0", quality="fp16", provider="cpu")
try:
    result = runtime.infer(
        token_ids,
        style=style_tensor,
        speed=1.0,
        seed=1234,
    )
finally:
    runtime.close()
```

The style tensor must already be in the format expected by the selected model. The frontend owns voice policy and style preparation.

## Split local layout

For `split-onnx-v1`, pass the model components and runtime metadata explicitly:

```python
from onnxvoice import OnnxVoice

runtime = OnnxVoice.open_local(
    system="kokoro",
    artifacts={
        "prosody": "prosody.onnx",
        "curves": "curves.onnx",
        "decoder": "decoder.onnx",
        "voices": "voices.npz",
        "config": "manifest.json",
        "source_params": "source-params.npz",
    },
    runtime={"layout": "split-onnx-v1"},
    sample_rate=24000,
    provider="cpu",
)
```

Both layouts expose the same model-ready inference boundary. Timing outputs are available in `InferenceResult` when the selected model provides them.
