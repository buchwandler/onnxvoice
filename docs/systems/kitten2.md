# KittenTTS 2

ONNXVoice provides a native `kitten2` system for the `kitten2-onnx-bundle-v1` layout. It is separate from the existing Kitten v0.8 `kitten` system and does not import or depend on the sibling `kitten2-onnx` runtime package.

`onnxvoice engines options --engine kitten2 --target kitten-tts-v2` reports the installed bundle's named voices, presets, and declared generation controls without executing text synthesis. The bundle manifest provides target-specific details after installation; metadata not available to discovery remains unavailable. See [CLI engine discovery](../cli.md#engine-option-and-capability-discovery).

The adapter owns bundle validation, local tokenizer loading, text and voice preparation, seeded generation, ONNX session lifecycle, and waveform decoding. It does not own application-specific text policy or voice selection.

## Install

The KittenTTS 2 text frontend is optional. Install it together with an ONNX Runtime provider:

```bash
python -m pip install "onnxvoice[kitten2,cpu]"
```

The `kitten2` extra installs only `tokenizers>=0.22,<0.23`, the lightweight runtime used to execute the bundle's `tokenizer/tokenizer.json`. The `cpu` and `gpu` extras select ONNX Runtime; use `onnxvoice[kitten2,gpu]` with a compatible GPU runtime. The KittenTTS 2 extra alone does not install an inference provider.

Text inference executes the bundled `tokenizer/tokenizer.json` locally and never downloads tokenizer or model files. The `tokenizers` dependency is loaded only when Kitten2 text synthesis is requested, so importing `onnxvoice` and using other systems does not require the Kitten2 extra.

## Managed and local use

Install and open the published bundle through the normal ONNXVoice lifecycle:

```python
from onnxvoice import OnnxVoice

manager = OnnxVoice()
installation = manager.install("kitten2:kitten-tts-v2")

with manager.open(installation, providers="cpu") as runtime:
    result = runtime.infer_text(
        "Hello from KittenTTS 2.",
        voice="Bruno",
        preset="stable",
        seed=7,
        max_new_tokens=64,
    )
```

`infer_text()` accepts caller-prepared text, an optional manifest voice, a decode preset, a seed, a generation limit, `use_emotion`, and `use_reference_prompt`. It returns the normal `InferenceResult` with finite mono `float32` audio at 24 kHz. Its metadata reports resolved expression conditioning and chunk count. The adapter exposes read-only `voices`, `default_voice`, `default_preset`, `decode_presets`, and `sample_rate` properties. `infer(token_ids)` is not supported for this text-level multi-stage system and raises `CapabilityError` directing callers to `infer_text()`.

## Text ownership and expression conditioning

ONNXVoice does not normalize KittenTTS 2 text. It passes the caller's non-empty string to the bundled tokenizer unchanged, including punctuation, numbers, whitespace, and Unicode. Callers that need number, date, currency, abbreviation, unit, or other written-to-spoken conversion must do that before `infer_text()`.

Kitten2 only inspects its supported model-specific expression markup: a known emotion tag such as `[joyful]`, a supported vocal event such as `<laugh>`, or a supported `(((emphasis)))` span. The markup remains in the tokenizer input. Recognized markup automatically enables the bundle's `generation.emotion_control` prompt IDs, inserted after the reference prefix and before `text_start`. Unknown markup is passed through unchanged and does not enable expression conditioning.

`use_emotion=None` selects automatic detection, `True` forces the control prompt, and `False` disables it without changing text. `use_reference_prompt=None` follows `generation.use_reference_prompt` from the bundle, while `True` and `False` override that setting.

## Long-form generation

The runtime uses the bundle's `generation.chunk_chars` and `generation.chunk_min_chars` settings for Kitten2-specific chunking. Boundaries are selected from existing whitespace or punctuation where possible. Chunks preserve the original character sequence and expression markup. A single request-level random generator is shared across the generated chunks, so repeating a request with the same seed is deterministic without restarting the stream for each chunk.

A complete local bundle can be opened without catalog or store access:

```python
from onnxvoice import OnnxVoice

runtime = OnnxVoice.open_local(
    system="kitten2",
    files={"bundle": "/models/kitten2/bundle.json"},
    providers="cpu",
)
try:
    result = runtime.infer_text("Hello from a local bundle.", voice="Bruno", seed=7)
finally:
    runtime.close()
```

The bundle directory is derived from `bundle.json`. Every declared relative path and SHA-256 is checked, nested paths are preserved, and undeclared files are rejected. ONNX external data must remain adjacent to its graph. Managed installations carry ONNXVoice's own install record in addition to the bundle inventory; local bundle validation remains strict and does not treat that record as a bundle asset.

## Bundle contract

`bundle.json` is authoritative for the runtime layout, sample rate, component paths and metadata, embedding file, tokenizer directory and assets, voice reference files, token IDs, decode presets, and generation defaults. The runtime requires layout `kitten2-onnx-bundle-v1` and sample rate `24000`.

The bundle includes the language model, speaker projection, flow decoder, and mel-to-wave decoder, with their metadata. It also includes the LM embedding matrix, `tokenizer/tokenizer.json`, any other declared tokenizer assets, voice reference arrays and metadata, and any graph external-data files. The `files` object must enumerate every runtime asset other than `bundle.json` with a lowercase SHA-256 digest. The validator rejects unsafe paths, missing or extra files, hash mismatches, malformed metadata, invalid voice or preset defaults, malformed token maps, empty `generation.emotion_control`, and non-adjacent `.onnx.data` files before creating ONNX sessions.

The four ONNX sessions (`lm`, `speaker`, `flow`, and `mel2wav`) are created lazily. `runtime.diagnostics().sessions` reports the component sessions after they have been opened. `close()` releases created sessions and is safe to call repeatedly.

## Provider behavior

Provider aliases and provider availability follow the shared [ONNX Runtime provider policy](../providers.md). Explicit `providers` and `provider_options` are passed to each component session. ONNXVoice does not append an implicit CPU fallback. The bundle's LM metadata may require a particular execution provider; the adapter checks that requirement rather than silently substituting another provider.

Some bundles declare the `low-memory-exact-ternary` LM profile. That profile requires CPU-only execution and component-specific session options. When ONNXVoice creates those options, they apply only to the LM session. Caller-provided LM session options are accepted only when compatible; conflicting values fail explicitly. Other component sessions keep the caller's normal session options.

## Catalog maintenance

The native catalog parser requires schema version 1, kind `kitten2-onnx-bundle-catalog`, layout `kitten2-onnx-bundle-v1`, sample rate 24000, and a lowercase 40-character Hugging Face commit SHA. Mutable revisions such as `main` are rejected in a catalog. Artifact paths and SHA-256 values are retained, and download URLs are constructed from the pinned commit.

Build resolves a branch or tag to a commit, reads the bundle manifest, and writes both the catalog and a deterministic provenance snapshot. It requires network access. Verification validates both files offline:

```bash
onnxvoice catalog kitten2 build \
  --repository holgern/kitten-tts-v2-onnx \
  --revision main \
  --output catalog/bundles.json \
  --source-output catalog/source.json

onnxvoice catalog kitten2 verify \
  --catalog catalog/bundles.json \
  --source catalog/source.json
```

`build` may accept `main`, but generated catalog records contain only the resolved immutable commit. Review the upstream model license and terms before redistributing or publishing a catalog entry.

## Smoke and parity gates

A real-bundle CPU smoke test uses voice `Bruno`, preset `stable`, seed `7`, and `max_new_tokens=64`. It must produce valid finite mono speech at 24 kHz. Passing unit tests or successfully constructing ONNX sessions is not evidence that a real bundle synthesizes valid speech.

The expression smoke uses `[joyful] I can (((hardly))) believe it! <laugh>` with the same CPU, voice, preset, seed, and token limit. It verifies finite mono float32 audio at 24 kHz and confirms `expression_conditioning` is enabled. The runtime rejects a request that produces no speech tokens before decoding.

The migration parity check compares this native implementation with the validated `kitten2-onnx` implementation using the same bundle, raw text, voice, preset, seed, and generation limit. Supply the out-of-band reference as a float32 NumPy array through `ONNXVOICE_KITTEN2_REFERENCE_AUDIO`; the resource-heavy test asserts equal shape and `numpy.testing.assert_allclose(rtol=1e-5, atol=1e-5)`. The reference is generated with text normalization disabled. The ONNXVoice runtime and test do not import or depend on `kitten2_onnx`. The current CPU bundle/text/seed run matched all 52,320 samples exactly (maximum absolute error 0). The legacy `kitten` v0.8 adapter and its model/style/speed contract remain independent.
