# Python usage

The high-level API is `OnnxVoice`. It combines catalog access, managed asset storage, inventory, and local runtime opening without taking ownership of text or voice policy.

## Construct a manager

```python
from onnxvoice import OnnxVoice

voice = OnnxVoice()
```

Pass `cache_dir` to select a cache root, `catalog_sources` to override catalog sources, or `offline=True` to disable network access.

## Discover catalog entries and voices

```python
for item in voice.list("piper", language="en-US"):
    print(item.ref)

for record in voice.list_voices(language="en-US"):
    print(record.ref, record.system, record.asset_id, record.voice_id, record.gender)
```

`list()` returns catalog items by system, or local installations when called with `installed=True`. `list_voices()` returns voices explicitly exposed by normalized catalog items. Each `VoiceRecord` includes its semantic `ref`, `system`, `asset_id`, `voice_id`, and descriptive metadata. Use `voice.resolve_voice(ref)` to validate and resolve a voice against current catalog data. Bundle children use refs such as `supertonic:supertonic-3/F1`; the `VoiceRecord.languages` tuple describes supported synthesis languages and does not change that ref.
Kitten child refs include `kitten:nano-0.8-int8/Bella`, and the backing asset ref is `kitten:nano-0.8-int8`. OnnxVoice does not select a model-ready style tensor; see [KittenTTS](systems/kitten.md) for the runtime interface and ownership boundary.

For merged installed and available inventory, use `inventory()`:

```python
records = voice.inventory(
    system="piper",
    language="en-US",
    gender="male",
)
```

Inventory can load catalog data. Use `installed_only=True` for local-only data.

## Install, open, and resolve

Acquisition and runtime opening are separate operations:

```python
installation = voice.install("piper:en_US-lessac-medium")
runtime = voice.open(installation, provider="cpu")
try:
    result = runtime.infer(token_ids)
finally:
    runtime.close()
```

`install()` resolves catalog data and downloads verified assets as needed. `open()` verifies and opens an existing managed installation only. It does not acquire missing assets. `resolve()` retrieves and verifies the existing local installation without opening a runtime.

Kitten uses the same managed lifecycle. Its runtime receives model-ready token IDs, a selected `style` tensor, and `speed`; the text frontend and style selection belong to the producer. See [KittenTTS](systems/kitten.md).

KittenTTS 2 is a separate native system with a text-level API. Install `onnxvoice[kitten2,cpu]`, then use `runtime.infer_text(text, voice="Bruno", preset="stable", seed=7)`. ONNXVoice passes the caller's text to the bundled tokenizer unchanged and does not perform number, date, currency, abbreviation, unit, language, or other semantic normalization. Use `use_emotion=None` for automatic Kitten2 markup detection, or `True`/`False` to override expression conditioning. `use_reference_prompt=None` follows the bundle setting, while `True`/`False` overrides it. The Kitten v0.8 `infer(token_ids, style, speed)` contract remains separate. See [KittenTTS 2](systems/kitten2.md) for bundle, catalog, provider, text-ownership, and chunking details.

An installation can also be found with `voice.installed()` or `voice.find_installed(ref)`. `voice.update(ref)` refreshes catalog data and atomically replaces the selected installation while preserving its quality and distribution by default.

## Reference WAVs and Pocket runtime

Voicefolio owns cataloging, discovery, and download of Kyutai reference WAVs. OnnxVoice does not manage generic reference-WAV assets. OnnxVoice's Pocket support manages Pocket ONNX bundles and predefined runtime voice states. See [Pocket downloads](pocket-downloads.md) for the bundle and runtime assets.

## Open explicit local files

Use `open_local()` when files should remain outside the managed store:

```python
runtime = OnnxVoice.open_local(
    system="piper",
    model="voice.onnx",
    config="voice.onnx.json",
    provider="cpu",
)
```

No catalog access, store registration, or file copy occurs. For multi-component layouts, provide named `artifacts=` or system-specific `files=` mappings. See [Local models](local-models.md).

## Chatterbox

OnnxVoice manages Chatterbox bundles and runs the four-graph adapter. The tokenizer frontend, reference WAV handling, and user-facing synthesis API belong to ChatterboxSynth; see the [Chatterbox system guide](systems/chatterbox.md) for managed/local opening and the model-ready token and reference-state contract.

## Qwen3 WaveKat

Qwen3 requires its WaveKat tokenizer/chat frontend to provide model-ready token IDs. OnnxVoice owns Qwen3 catalog/install/runtime and exposes VoiceDesign inference, Base reference preparation, `Qwen3VoiceState`, graph/provider diagnostics, and local bundle opening. See [Qwen3-TTS WaveKat](systems/qwen3.md) for the exact ID, audio, bundle, and provider contracts; use [Qwen3Synth](https://github.com/buchwandler/qwen3synth) for the text/WAV/profile workflow.

## Runtime inspection and provider policy

ONNX Runtime distributions/builds and execution providers are distinct. The top-level API exposes the same structured runtime status used by the CLI:

```python
import onnxvoice

status = onnxvoice.runtime_status()
print(status.module_version)
print(status.imported_distribution)
print(status.available_providers)
print(status.configured_providers)
print(status.provider_source)
print(status.resolved_providers)
```

`OnnxRuntimeStatus` includes the imported module path/build, discovered distribution metadata and ownership state, providers reported by ORT, and any import or provider-resolution error. Reported providers are not a hardware/native-library self-test. Runtime status does not initialize a model session.

The effective provider policy precedence is:

1. Explicit `provider=` / `providers=` Python argument.
2. `ONNXVOICE_PROVIDERS`.
3. `ONNXVOICE_PROVIDER`.
4. Saved user-level provider preference.
5. Built-in `cpu` default.

Persist or clear the default with `onnxvoice.set_default_providers("cuda")` and `onnxvoice.clear_default_providers()`. Saved preferences apply only when explicit arguments and environment overrides are absent. They affect future sessions; they do not mutate an already-open runtime. Environment values remain process-owned and are not changed by clearing the saved preference.

## Providers and session diagnostics

Pass `provider=` or `providers=` to `open()` or `open_local()` for an explicit per-runtime override. `OnnxSession` resolves aliases against the providers reported by the imported build. `runtime.diagnostics()` returns provider and graph details for every session, including multi-session adapters:

```python
diagnostics = runtime.diagnostics()
for session in diagnostics.sessions:
    print(session.providers_requested)
    print(session.provider_source)
    print(session.providers_resolved)
    print(session.providers_active)
    print(session.inputs, session.outputs)
```

`providers_active` is specific to an initialized inference session and is obtained from that session; it is not a global runtime setting. See [ONNX Runtime providers](providers.md) for CLI commands, distribution guidance, persistence, automatic selection, and environment configuration.

## Runtime result

Inference returns an `InferenceResult` with one-dimensional float32 `audio`, a positive `sample_rate`, and optional `timings`, named `outputs`, and `metadata`. Model-specific inputs remain the caller's responsibility. For example, Piper needs model-ready token IDs, while Kokoro also requires a model-ready style tensor.

## Lower-level modules

The [API reference](api/index.md) covers the catalog client, asset store, runtime session, inventory helpers, semantic voice APIs, public data types, validation helpers, and errors.
