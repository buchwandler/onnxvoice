# Supertonic

Supertonic has a built-in adapter and catalog support. The runtime is a multi-component bundle rather than a single ONNX file. `onnxvoice` executes the model contract; the caller supplies the prepared token IDs, text mask, and style tensors. It does not add text normalization or phonemization.

## Runtime components

The model components are:

- `duration_predictor`
- `text_encoder`
- `vector_estimator`
- `vocoder`

The runtime also uses `config`, `unicode_indexer`, and one or more `voice_style:<name>` files.

## Open local files

```python
from onnxvoice import OnnxVoice

runtime = OnnxVoice.open_local(
    system="supertonic",
    files={
        "config": "onnx/tts.json",
        "unicode_indexer": "onnx/unicode_indexer.json",
        "duration_predictor": "onnx/duration_predictor.onnx",
        "text_encoder": "onnx/text_encoder.onnx",
        "vector_estimator": "onnx/vector_estimator.onnx",
        "vocoder": "onnx/vocoder.onnx",
        "voice_style:F1": "voice_styles/F1.json",
    },
    sample_rate=44100,
    provider="cpu",
)
```

Inference requires model-ready values for all style and mask inputs:

```python
result = runtime.infer(
    token_ids,
    text_mask=text_mask,
    style_ttl=style_ttl,
    style_dp=style_dp,
    steps=5,
    speed=1.05,
    seed=1234,
)
```

`steps` must be from 1 through 100, `speed` from 0.7 through 2.0, and `seed` may be an integer or `None`. The ONNX Runtime adapter does not construct tokens, text masks, or style tensors from user-facing text or voice names.

Catalog installable bundles are available where configured by the Supertonic catalog source. The same component roles are used by managed installations; see [Local models](../local-models.md) for the unmanaged local-file boundary.
