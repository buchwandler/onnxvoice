# Qwen3-TTS WaveKat

OnnxVoice supports two WaveKat Qwen3-TTS ONNX bundles through the explicit `qwen3-wavekat-v1` runtime ABI:

| Catalog reference         | Capability                                                           | Qualities      |
| ------------------------- | -------------------------------------------------------------------- | -------------- |
| `qwen3:voice-design-1.7b` | Generate speech from frontend-tokenized text and a voice description | `fp32`, `int4` |
| `qwen3:base-0.6b`         | Clone from a prepared reference voice and transcript                 | `fp32`, `int4` |

The catalog also accepts the short aliases `voice-design-1.7b` and `base-0.6b`. These entries refer to immutable source revisions in the [qwen3-onnx-bundles catalog](https://github.com/buchwandler/qwen3-onnx-bundles). Other Qwen3 exporters or graph layouts are not compatible by implication; they require a separately versioned ABI and their own validation/parity evidence.

## Ownership boundary

OnnxVoice owns Qwen3 catalog parsing and selection, asset acquisition and integrity checks, installation layout, config/embedding/graph validation, provider/session lifecycle, generation, codec prediction, vocoding, and non-sensitive runtime diagnostics.

A producer such as [Qwen3Synth](https://github.com/buchwandler/qwen3synth) owns text tokenization, Qwen chat templates, language normalization, WAV I/O and resampling, generation-facing request policy, and saved voice profiles. OnnxVoice accepts **already-tokenized, model-ready IDs**; it does not accept raw text or construct chat templates. For normal text-to-speech workflows, use the producer API rather than assembling token IDs yourself.

## Install and open a managed bundle

Install a provider-specific ONNX Runtime extra, then explicitly install a bundle. Installation/catalog access may use the network; opening an existing installation does not acquire assets.

```bash
python -m pip install "onnxvoice[cpu]"
onnxvoice install qwen3:voice-design-1.7b --quality int4
```

```python
from onnxvoice import OnnxVoice

manager = OnnxVoice()
installation = manager.install("qwen3:base-0.6b", quality="int4")
runtime = manager.open(installation, providers="cpu")
try:
    print(runtime.diagnostics())
finally:
    runtime.close()
```

`quality` must be `fp32` or `int4`. Available providers depend on the installed ONNX Runtime build and platform; INT4 does not guarantee low peak memory or support on every provider. See [ONNX Runtime providers](../providers.md).

The default catalog source is the companion bundle repository. Override it with `ONNXVOICE_QWEN3_CATALOG` or `OnnxVoice(catalog_sources={"qwen3": "https://…/bundles.json"})`. Local/offline applications can construct `OnnxVoice(offline=True)` and open an installation already present in that cache.

## Runtime API

`OnnxVoice.open()` returns the Qwen3 adapter for the selected installation. `infer()` takes frontend-tokenized IDs and returns the common `InferenceResult` (`audio` is mono `float32`; the WaveKat sample rate is 24 kHz). The examples below intentionally use names such as `text_ids` as placeholders: obtain these arrays from a compatible tokenizer/chat-template frontend such as Qwen3Synth.

### VoiceDesign

The text IDs must use the supported assistant chat wrapping, and the description must be supplied separately as user/instruction token IDs. Both sequences must be within the installed text embedding vocabulary.

```python
import numpy as np

# Assume runtime is an open Qwen3 adapter; the ID sequences come from the frontend.
result = runtime.infer(
    text_ids,
    instruction_token_ids=instruction_ids,
    language="english",
    temperature=0.9,
    top_k=50,
    repetition_penalty=1.05,
    max_new_tokens=512,
    seed=42,
)
assert result.sample_rate == 24_000
assert result.audio.dtype == np.float32
```

Sampling defaults are temperature `0.9`, top-k `50`, repetition penalty `1.05`, and at most `2048` new codec frames. Temperature must be in `(0, 5]`, top-k is non-negative (`0` disables top-k), repetition penalty is positive, `max_new_tokens` is `1..8192`, and the optional seed is an unsigned 32-bit integer. A seed makes the local sampling RNG repeatable; it is not a promise of cross-provider waveform identity.

### Base reference cloning

Prepare reference state once and reuse it for target utterances. The low-level API requires mono finite audio already converted to 24 kHz; it does not decode WAV files or resample. Supply the exact transcript as wrapped reference token IDs from the frontend.

```python
state = runtime.prepare_voice(
    reference_samples,  # one-dimensional, finite audio at 24 kHz
    sample_rate=24_000,
    reference_token_ids=reference_ids,
)
result = runtime.infer(
    target_text_ids,
    voice_state=state,
    language="english",
    temperature=0.9,
    top_k=50,
    max_new_tokens=512,
    seed=42,
)
```

The Base audio contract is mono one-dimensional numeric samples, at least 1,024 samples and at most 240,000 samples (10 seconds), finite and at 24 kHz. `Qwen3VoiceState` is an immutable, reusable value exported from `onnxvoice`; it retains the speaker embedding, 16-group reference codec codes, transcript token IDs, sample count, model identity, and layout. A state for another model identity/layout is rejected. Qwen3Synth adds its own WAV, duration, and profile checks at the frontend boundary.

Both model types require the supported WaveKat chat-wrapped token structure. `language` selects a codec language token when present in the installed model config; the frontend is responsible for normalizing user-facing language names.

## Local, unmanaged bundles

`OnnxVoice.open_local(system="qwen3", files=..., runtime=..., sample_rate=24000, providers="cpu")` opens explicit paths without catalog access, network requests, copying, or store registration. Supply the complete inventory. Each model graph needs a matching adjacent `.onnx.data` file; Base additionally needs its shared `speaker_encoder` and `tokenizer_encoder` graph/data pairs.

```python
from pathlib import Path
from onnxvoice import OnnxVoice

root = Path("/models/qwen3")
quality = "int4"
model_type = "base"  # or "voice_design"
files = {"config": root / "config.json"}

tokenizers = {
    "tokenizer_json": "tokenizer.json",
    "vocab": "vocab.json",
    "merges": "merges.txt",
    "tokenizer_config": "tokenizer_config.json",
    "special_tokens": "special_tokens_map.json",
    "added_tokens": "added_tokens.json",
}
for component, filename in tokenizers.items():
    files[f"tokenizer:{component}"] = root / "tokenizer" / filename

embeddings = [
    "text_embedding",
    "text_projection_fc1_weight",
    "text_projection_fc1_bias",
    "text_projection_fc2_weight",
    "text_projection_fc2_bias",
    "talker_codec_embedding",
    *(f"cp_codec_embedding_{index}" for index in range(15)),
]
for component in embeddings:
    files[f"embedding:{component}"] = root / "embeddings" / f"{component}.npy"

models = ["talker_prefill", "talker_decode", "code_predictor", "vocoder"]
if model_type == "base":
    models += ["speaker_encoder", "tokenizer_encoder"]
for component in models:
    graph = (
        root / f"{component}.onnx"
        if component in {"speaker_encoder", "tokenizer_encoder"}
        else root / quality / f"{component}.onnx"
    )
    files[f"model:{component}"] = graph
    files[f"model_data:{component}"] = Path(str(graph) + ".data")

runtime = OnnxVoice.open_local(
    system="qwen3",
    files=files,
    runtime={"layout": "qwen3-wavekat-v1", "model_type": model_type, "quality": quality},
    sample_rate=24_000,
    providers="cpu",
)
try:
    # Use the same prepare_voice()/infer() API described above.
    ...
finally:
    runtime.close()
```

The complete local inventory also requires the six listed tokenizer files and the 21 listed embeddings. Model, config, embedding, and external-data formats/paths are checked when opening; sessions are created lazily and their graph signatures are validated before inference. Do not split an ONNX graph from its adjacent external-data file.

## Providers and diagnostics

Sessions are lazy. The first inference or Base preparation creates the needed ONNX sessions and validates the WaveKat graph ABI. A Qwen3 adapter rejects inconsistent provider resolutions across graph sessions instead of silently mixing providers. `runtime.diagnostics()` reports each loaded session's requested/active provider and tensor signatures, plus non-sensitive model type, quality, layout, and config dimensions. Diagnostics do not prove speech quality or parity.

## Catalog maintenance

Catalog verification is offline; catalog building queries the pinned upstream repositories and computes or checks artifact integrity. The source of truth and schema are maintained in the companion `qwen3-onnx-bundles` repository.

```bash
onnxvoice catalog qwen3 verify --catalog ../qwen3-onnx-bundles/catalog/bundles.json \
  --source ../qwen3-onnx-bundles/catalog/source.json
onnxvoice catalog qwen3 build --output /tmp/qwen3-bundles.json \
  --source-output /tmp/qwen3-source.json
```

## Tests and parity limits

Offline unit tests cover catalog contracts, local inventories, graph signatures, deterministic sampling, fake-session generation, immutable Base conditioning, and the committed NumPy log-mel reference fixture. The CPU integration test is marked `integration` and `resource_heavy`; it exercises actual VoiceDesign and Base inference at both qualities when those bundles are already installed and never downloads assets. Without `ONNXVOICE_REQUIRE_QWEN3_REAL=1`, missing runtime or installations may be skipped for developer convenience. Set that variable for release verification; missing ONNX Runtime or any required profile then fails. Pytest's default discovery does not exclude integration markers, so use `-m "not integration"` for an offline-only suite.

```bash
export ONNXVOICE_TEST_CACHE=/path/to/onnxvoice-cache
for bundle in qwen3:voice-design-1.7b qwen3:base-0.6b; do
  for quality in fp32 int4; do
    onnxvoice --cache-dir "$ONNXVOICE_TEST_CACHE" install "$bundle" --quality "$quality"
  done
done
ONNXVOICE_REQUIRE_QWEN3_REAL=1 \
pytest -vv -m "integration and resource_heavy" tests/test_qwen3_integration.py
```

Real CPU WaveKat execution has been verified for all four advertised profiles. See [Qwen3 release validation](qwen3-release-validation.md) for the measured environment, source revisions, command, and results. The release gate checks non-empty finite non-silent mono float32 audio at 24 kHz and active `CPUExecutionProvider`. This confirms runtime graph execution, not numerical parity against another implementation. Audited output comparisons and listening/perceptual assessment remain unverified. Do not describe unit tests or the log-mel fixture as end-to-end speech parity.
