# Local models

There are two different local workflows: open explicit files without managing them, or import a supported model into the shared store.

## Open local files without registration

```python
from onnxvoice import OnnxVoice

runtime = OnnxVoice.open_local(
    system="piper",
    model="voice.onnx",
    config="voice.onnx.json",
    provider="cpu",
)
try:
    result = runtime.infer(token_ids)
finally:
    runtime.close()
```

`open_local()` does not access a catalog, register files in the shared store, copy the files, or take ownership of their paths. The runtime uses the explicit local files.

Use `artifacts=` for named component mappings, or `files=` for system-specific file roles. These arguments are mutually exclusive with `model=`. See the system pages for multi-component mappings.

## Import a model into managed storage

```python
from onnxvoice import OnnxVoice

voice = OnnxVoice()
installation = voice.import_model(
    system="piper",
    item_id="my-voice",
    model="voice.onnx",
    config="voice.onnx.json",
)
runtime = voice.open(installation, provider="cpu")
try:
    result = runtime.infer(token_ids)
finally:
    runtime.close()
```

Importing creates a managed installation and manifest using the store's normal verification and content-addressed storage semantics. The high-level import helper accepts a model and optional config and voices files. Multi-component runtimes should be opened with their complete explicit layout rather than represented as a single model file.

## Multi-component runtimes

Kokoro supports single-model and `split-onnx-v1` layouts. For local split layout, provide the prosody, curves, decoder, voices, config, and source parameters artifacts and the matching runtime layout metadata. See [Kokoro](systems/kokoro.md).

Supertonic uses named model components, a config, a Unicode indexer, and one or more voice-style files. Pass these roles through `files=`. See [Supertonic](systems/supertonic.md).

Pocket local opening requires a complete bundle and its contract metadata. See [Pocket](systems/pocket.md).
Kitten uses a model ONNX file and `voices.npz`. Supply both through the `model=` and `voices=` parameters to `open_local()`; style extraction and selection stay with the caller. See [KittenTTS](systems/kitten.md).

The simple CLI `onnxvoice import --model ...` is not a generic multi-component bundle importer.
