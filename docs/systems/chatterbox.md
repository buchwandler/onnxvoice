# Chatterbox

OnnxVoice provides catalog-backed installation and a cleanroom ONNX runtime adapter for the supported Chatterbox bundles. It is the graph/runtime layer, not a complete user-facing text-to-speech package: ChatterboxSynth owns the tokenizer frontend, reference-audio workflow, and public synthesis result/API.

## Supported bundles and profiles

| Bundle reference     | Aliases             | Profiles        |
| -------------------- | ------------------- | --------------- |
| `chatterbox:nano-v2` | `nano`, `nano-110m` | `cpu`, `fp32`   |
| `chatterbox:turbo`   | `turbo-350m`        | `fp32`, `q4f16` |

Both bundles generate audio at the native sample rate of 24 kHz. The profile selects the matching graph artifacts; install a compatible ONNX Runtime provider for the target device. For example, the Nano `cpu` profile uses CPU-oriented graph variants, while Turbo also offers the `q4f16` profile.

## Ownership boundary

OnnxVoice owns the `chatterbox-onnx-bundles` catalog integration, revision-pinned artifact metadata, managed cache and installation, ONNX sessions, graph-contract validation, and generation. Its adapter runs four graph components:

- `speech_encoder` prepares reference conditioning.
- `embed_tokens` embeds frontend-provided token IDs.
- `language_model` generates speech tokens.
- `conditional_decoder` turns generated speech tokens into audio.

The adapter does not tokenize text, decode or resample WAV files, or provide a high-level synthesis request/result API. A frontend such as ChatterboxSynth owns those user-facing responsibilities. OnnxVoice accepts model-ready text token IDs and mono finite reference audio already converted to `float32` at 24 kHz.

## Install and open a managed bundle

Install the CPU runtime extra, then install and open a bundle. The frontend must supply model-ready token IDs and prepared 24 kHz mono reference samples.

```bash
python -m pip install "onnxvoice[cpu]"
onnxvoice install chatterbox:nano-v2 --quality cpu
```

```python
from onnxvoice import OnnxVoice

manager = OnnxVoice()
installation = manager.install("chatterbox:nano-v2", quality="cpu")
runtime = manager.open(installation, providers="cpu")
try:
    # The frontend decodes/resamples the reference and tokenizes the text.
    voice_state = runtime.prepare_voice(reference_audio_24khz_mono_float32)
    result = runtime.infer(text_token_ids, voice_state=voice_state)
    assert result.sample_rate == 24_000
finally:
    runtime.close()
```

Use `chatterbox:turbo` with `quality="fp32"` or `quality="q4f16"` for the Turbo profiles. `OnnxVoice.install()` acquires the catalog artifacts; `open()` uses an existing managed installation and does not download assets.

## Runtime contract

`prepare_voice(audio_values)` runs the speech encoder once and returns an immutable, reusable `ChatterboxVoiceState`. Reuse the state for multiple utterances with the same prepared reference. The adapter requires non-empty, finite mono audio at the bundle's native sample rate; WAV decoding and resampling are frontend responsibilities.

`infer(token_ids, *, voice_state=..., max_new_tokens=None, repetition_penalty=None)` accepts a non-empty sequence of non-negative model-ready integer token IDs and a prepared `ChatterboxVoiceState`. The adapter owns the `max_new_tokens` and `repetition_penalty` generation controls. It returns the common `InferenceResult` with mono `float32` audio and the 24 kHz sample rate. It does not accept raw text or promise frontend-level synthesis behavior.

## Open a local multi-file bundle

`open_local()` can open the four graph files directly, without catalog access, copying, or registering them in managed storage. The local graph set and runtime metadata must match the supported Chatterbox ABI:

```python
from pathlib import Path

from onnxvoice import OnnxVoice

root = Path("/models/chatterbox")
components = ("speech_encoder", "embed_tokens", "language_model", "conditional_decoder")
files = {f"graph:{name}": root / f"{name}.onnx" for name in components}
runtime_metadata = {
    "start_speech_token": 6561,
    "stop_speech_token": 6562,
    "silence_token": 4299,
    "trailing_silence_tokens": 3,
    "default_max_new_tokens": 1024,
    "default_repetition_penalty": 1.2,
    "profile": "fp32",
}

runtime = OnnxVoice.open_local(
    system="chatterbox",
    files=files,
    runtime=runtime_metadata,
    sample_rate=24_000,
    providers="cpu",
)
try:
    voice_state = runtime.prepare_voice(reference_audio_24khz_mono_float32)
    result = runtime.infer(text_token_ids, voice_state=voice_state)
finally:
    runtime.close()
```

The local example assumes the four files are compatible with one another and that `text_token_ids` and reference samples have already been prepared by a compatible frontend.

## Catalog source and maintenance

The default catalog is published by [chatterbox-onnx-bundles](https://github.com/buchwandler/chatterbox-onnx-bundles). Its seed requests named revisions; catalog building resolves those revisions to immutable commit IDs and records artifact integrity. Verification is offline and checks the catalog against its source snapshot.

```bash
onnxvoice catalog chatterbox verify \
  --catalog ../chatterbox-onnx-bundles/catalog/bundles.json \
  --source ../chatterbox-onnx-bundles/catalog/source.json
```

The maintainer build command takes an explicit seed and writes both catalog and source snapshot:

```bash
onnxvoice catalog chatterbox build \
  --seed onnxvoice/catalog_tools/chatterbox_seed.json \
  --output /tmp/chatterbox-bundles.json \
  --source-output /tmp/chatterbox-source.json
```
