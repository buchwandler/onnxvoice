# Kokoro

Kokoro supports a single ONNX model, the `split-onnx-v1` multi-component layout, and the reference-conditioned `cloning-onnx-v1` layout. Single and split inference accept caller-prepared model-ready style tensors. Cloning has a separate enrollment boundary and uses a reusable `KokoroReferenceState` instead of stock voices.

## Quality and distribution

A quality selection chooses a model variant within a distribution. It does not install every quality variant. Non-model runtime artifacts required by the selected distribution are installed with it. Explicit quality and distribution selections can have separate installation identities. A cloning distribution includes its enrollment graphs, synthesis graphs, source parameters, and config together for each supported quality.

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

## Reference cloning layout

`cloning-onnx-v1` enrolls a reference waveform once, then reuses its mapped style and reference memory for synthesis. All learned inference runs through ONNX Runtime. NumPy handles deterministic mel preprocessing and the existing split runtime's duration expansion, harmonic source, and STFT operations.

A local installation requires all six model components, `metadata:source_params`, and a JSON `config`, `bundle`, or `manifest` artifact. It does not require a `voices` archive:

```python
from onnxvoice import OnnxVoice

runtime = OnnxVoice.open_local(
    system="kokoro",
    artifacts={
        "model:reference_wavlm": "reference_wavlm.onnx",
        "model:reference_encoders": "reference_encoders.onnx",
        "model:reference_mapper": "reference_mapper.onnx",
        "model:prosody": "prosody.onnx",
        "model:curves": "curves.onnx",
        "model:decoder": "decoder.onnx",
        "metadata:source_params": "source-params.npz",
        "config": "config.json",
    },
    runtime={
        "layout": "cloning-onnx-v1",
        "voice_mode": "reference",
        "max_tokens": 510,
        "sample_rate": 24000,
        "speed_supported": False,
        "style_dimensions": {"acoustic": 128, "duration": 128},
        "reference": {
            "format": "akinvox-cloning-reference-v1",
            "sample_rate": 24000,
            "identity_sample_rate": 16000,
            "min_seconds": 3.0,
            "max_seconds": 30.0,
            "style_width": 256,
            "memory_width": 192,
        },
    },
    sample_rate=24000,
    provider="cpu",
)
```

The high-level caller owns file I/O and audio resampling. Enrollment takes finite mono floating-point arrays at 24 kHz and 16 kHz with matching duration. The 24 kHz reference must last 3 to 30 seconds and must not be silent or severely clipped. `reference_token_ids` are the reference transcript's phone IDs without boundary tokens; the runtime adds boundaries and enforces the 510-token limit.

```python
reference = runtime.prepare_reference(
    reference_token_ids,
    audio_24k=wave_24k,
    audio_16k=wave_16k,
)

result = runtime.infer(
    token_ids,
    reference=reference,
    speed=1.0,
    seed=1234,
)
```

`KokoroReferenceState` is publicly available from `onnxvoice`. Its style, memory, and mask arrays are owned read-only copies and its fingerprint binds it to the exact installed component artifacts. Reuse it only with that same model build. Cross-build use is rejected. Clone synthesis rejects any speed other than `1.0`; stock style input is not accepted by the cloning layout. Reference-state persistence and WAV loading are outside this package.

`runtime.close()` closes only sessions that were opened. `runtime.diagnostics()` reports the providers and components for those sessions. With a precomputed reference state, synthesis does not initialize the enrollment graphs.

Timing outputs are available in `InferenceResult` when the selected model provides them.
