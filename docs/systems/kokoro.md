# Kokoro

Kokoro supports a single ONNX model, the `split-onnx-v1` multi-component layout, and the reference-conditioned `cloning-onnx-v1` layout. Single and split inference accept caller-prepared model-ready style tensors. Ordinary single and split installations may also advertise optional Inno voice enrollment through `runtime.voice_enrollers`; Inno adds components without changing the Kokoro layout. Cloning remains separate and uses a reusable `KokoroReferenceState` instead of stock voices.

## Quality and distribution

A quality selection chooses a base Kokoro model variant within a distribution. It does not install every quality variant. Required runtime artifacts are installed with the selected model. An Inno distribution keeps its shared `model:inno_voicepack` and `metadata:inno_tuner` artifacts when selecting a base-model quality. Explicit quality and distribution selections can have separate installation identities. A cloning distribution includes its enrollment graphs, synthesis graphs, source parameters, and config together for each supported quality.

## Optional Inno voice enrollment

Inno v0.2 adds reference-audio enrollment as an optional capability on ordinary single or split Kokoro installations. The base Kokoro inference graph remains separate. Enrollment uses an ONNX Runtime graph plus host-side preprocessing, and does not import PyTorch, Torchaudio, Transformers, or upstream Inno. Install the optional preprocessing dependencies with `pip install 'onnxvoice[inno]'`. The ONNX Runtime provider remains an independent installation choice.

Catalog runtime metadata advertises the capability and names its artifacts as `model:inno_voicepack` and `metadata:inno_tuner`. Catalog parsing verifies that both are present and shared across base-model qualities. A local installation can use the same artifact roles and components:

```python
from onnxvoice import OnnxVoice

# Copy the inno-v0.2 entry from catalog runtime.voice_enrollers.
runtime = OnnxVoice.open_local(
    system="kokoro",
    artifacts={
        "model": "model.onnx",
        "voices": "voices.npz",
        "model:inno_voicepack": "inno-voicepack.onnx",
        "metadata:inno_tuner": "inno-tuner.npz",
    },
    runtime={
        "layout": "single-onnx-v1",
        "voice_enrollers": [inno_capability_metadata],
    },
    sample_rate=24000,
)
```

`inno_capability_metadata` must declare id `inno-v0.2`, kind `kokoro-voicepack-tuner`, reference-audio input, no transcript requirement, duration limits of 3/5/30 seconds, output format `kokoro-voicepack-v1` with shape `[510, 1, 256]` and dtype `float32`, and the two component names above. Split installations use their usual `model:prosody`, `model:curves`, and `model:decoder` artifacts alongside the same Inno artifacts.

Enrollment accepts a floating-point waveform and its sample rate. SciPy and Praat/Parselmouth are optional dependencies supplied by the `inno` extra. Inno's ONNX session is loaded only when enrollment is requested; the returned pack can then be used with the ordinary Kokoro inference API without loading the tuner again:

```python
pack = runtime.enroll_voice(reference_audio, sample_rate=24000)
result = runtime.infer(token_ids, style=pack.style_for(len(token_ids)))
```

The `KokoroVoicePack` contains the stock-compatible `[510, 1, 256]` style table and artifact-bound provenance. `style_for` selects the row matching the token count and returns a model-ready style tensor. Enrollment does not replace or alter stock voice styles.

The offline exporter targets `remsky/inno-kokoro` v0.2.0 at commit `892ef184bc932aa3ff9d72c1509d5b81ff6941e6`. Install its exporter-only dependencies with `pip install 'onnxvoice[inno-export]'`, then supply a local checkpoint: `python tools/export_kokoro_inno.py --weights model.safetensors --output inno_voicepack.onnx --metadata-output inno_tuner.npz`. It checks the checkpoint version and records the pinned source revision and checkpoint SHA-256 in the generated metadata and ONNX graph. It does not download weights, and model files are not bundled with onnxvoice.

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
