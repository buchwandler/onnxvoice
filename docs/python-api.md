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

`list()` returns catalog items by system, or local installations when called with `installed=True`. `list_voices()` returns voices explicitly exposed by normalized catalog items. Each `VoiceRecord` includes its semantic `ref`, `system`, `asset_id`, `voice_id`, and descriptive metadata. Use `voice.resolve_voice(ref)` to validate and resolve a voice against current catalog data.

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

An installation can also be found with `voice.installed()` or `voice.find_installed(ref)`. `voice.update(ref)` refreshes catalog data and atomically replaces the selected installation while preserving its quality and distribution by default.

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

## Providers and diagnostics

Pass `provider=` or `providers=` to `open()` or `open_local()`. `OnnxSession` resolves aliases against the installed ONNX Runtime provider set. `runtime.diagnostics()` returns provider and graph details for every session, including multi-session adapters:

```python
diagnostics = runtime.diagnostics()
for session in diagnostics.sessions:
    print(session.providers_requested)
    print(session.providers_active)
    print(session.inputs, session.outputs)
```

See [ONNX Runtime providers](providers.md) for installation extras, automatic selection, and environment configuration.

## Runtime result

Inference returns an `InferenceResult` with one-dimensional float32 `audio`, a positive `sample_rate`, and optional `timings`, named `outputs`, and `metadata`. Model-specific inputs remain the caller's responsibility. For example, Piper needs model-ready token IDs, while Kokoro also requires a model-ready style tensor.

## Lower-level modules

The [API reference](api/index.md) covers the catalog client, asset store, runtime session, inventory helpers, semantic voice APIs, public data types, validation helpers, and errors.
