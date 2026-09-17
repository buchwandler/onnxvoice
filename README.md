# onnxvoice

`onnxvoice` manages, validates and runs ONNX voice models from one shared local store.

The project deliberately sits below text processing. PyKokoro, PiperSynth, UtterRender or another frontend can turn text into tokens. `onnxvoice` resolves the requested model/voice, manages the shared cache, creates the ONNX Runtime session, maps system-specific inputs and returns NumPy audio.

> Status: MVP / API exploration. Expect breaking changes before 0.1.

## Why

Without a shared layer, each TTS package tends to implement its own catalog client, download logic, cache directory, checksums, ONNX Runtime provider handling and model-specific inference glue. `onnxvoice` centralizes that middle layer.

The cache is content-addressed:

```text
~/.cache/onnxvoice/
├── blobs/sha256/ab/abcdef...
├── catalogs/
│   ├── kokoro.json
│   └── piper.json
└── installs/
    ├── kokoro/v1.0/manifest.json
    └── piper/en_US-lessac-medium/manifest.json
```

Installations hard-link to immutable blobs when the filesystem supports it. The same bytes therefore do not need to be stored twice by different model installations.

## Install

Base package, without an ONNX runtime:

```bash
pip install -e .
```

CPU runtime:

```bash
pip install -e '.[cpu]'
```

Development:

```bash
pip install -e '.[dev,cpu]'
```

GPU/runtime variants are optional extras:

```bash
pip install -e '.[gpu]'
pip install -e '.[directml]'
pip install -e '.[openvino]'
```

## Catalogs

The MVP directly understands the existing catalogs from:

- `buchwandler/piper-onnx-voices` (`catalog/voices.json`)
- `buchwandler/kokoro-onnx-models` (`catalog/models.json`)

Override them without changing code:

```bash
export ONNXVOICE_PIPER_CATALOG=/path/to/voices.json
export ONNXVOICE_KOKORO_CATALOG=/path/to/models.json
```

A local path or HTTP(S) URL is accepted.

## CLI

```bash
onnxvoice list --system piper --language de
onnxvoice list --system kokoro --quality fp16

onnxvoice install piper:en_US-lessac-medium
onnxvoice install kokoro:v1.0 --quality fp16

onnxvoice list --system piper --installed
onnxvoice path piper:en_US-lessac-medium
onnxvoice verify piper:en_US-lessac-medium
onnxvoice show piper:en_US-lessac-medium

onnxvoice cache info
onnxvoice cache gc
```

Kokoro has multiple ONNX model qualities in one distribution. `onnxvoice install kokoro:v1.0` selects `fp32` by default rather than downloading all model variants. Non-model runtime artifacts from the selected distribution are installed with it.

## Python API

### Discover and install

```python
from onnxvoice import OnnxVoice

ov = OnnxVoice()

for voice in ov.list("piper", language="en_US"):
    print(voice.ref)

piper = ov.install("piper:en_US-lessac-medium")
kokoro = ov.install("kokoro:v1.0", quality="fp16")

print(ov.where("piper:en_US-lessac-medium"))
print([item.ref for item in ov.installed()])
```

### Piper inference

`onnxvoice` expects already-tokenized Piper IDs. It does not phonemize text.

```python
from onnxvoice import load

runtime = load("piper:en_US-lessac-medium")
result = runtime.infer(
    [1, 20, 14, 5, 2],
    length_scale=1.0,
    noise_scale=0.667,
    noise_w=0.8,
)

print(result.audio.dtype)
print(result.sample_rate)
runtime.close()
```

### Kokoro inference

Kokoro voice styles are loaded from the installed voice archive. The style row is selected from the effective token length.

```python
runtime = load("kokoro:v1.0", quality="fp16")
result = runtime.infer(
    [50, 31, 12, 99],
    voice="af_heart",
    speed=1.0,
)
print(result.audio.shape, result.sample_rate)
runtime.close()
```

### External/local models

External files can be imported into the same store:

```python
ov.import_model(
    system="piper",
    item_id="my-voice",
    model="voice.onnx",
    config="voice.onnx.json",
)

runtime = ov.load("piper:my-voice", download=False)
```

For Kokoro:

```python
ov.import_model(
    system="kokoro",
    item_id="my-kokoro",
    model="kokoro.onnx",
    voices="voices.npz",
    sample_rate=24000,
)
```

## System adapters

A TTS system adapter owns only the model-specific ONNX contract. It does not own text normalization, G2P, sentence splitting or document planning.

```python
from onnxvoice.systems import SystemAdapter, register_adapter


class MyTTSAdapter(SystemAdapter):
    system = "mytts"
    ...


register_adapter("mytts", MyTTSAdapter)
```

The built-in MVP adapters are `piper` and `kokoro`.

## Validation levels

The MVP includes three inexpensive building blocks:

- installed asset verification: file presence, size and SHA-256
- ONNX load/contract smoke check via ONNX Runtime
- returned audio sanity: numeric, finite and non-silent

Release-grade waveform parity, spectral gates and reference comparisons belong in a later validation layer. They should not run on every inference.

## Architecture

```text
PyKokoro / PiperSynth / another frontend
             │
             │ tokens + voice/model choice
             ▼
        onnxvoice
        ├── CatalogClient
        ├── AssetStore
        ├── SystemAdapter
        │   ├── PiperAdapter
        │   └── KokoroAdapter
        ├── OnnxSession
        └── validation
             │
             ▼
        NumPy audio
```

UtterRender should normally consume PyKokoro/PiperSynth and let those packages use `onnxvoice` underneath, rather than becoming another downloader/cache owner.

## Versioning

The project uses `setuptools_scm`. There is no hard-coded project version and no `src/` layout. Tagged Git commits produce package versions dynamically. A source tree without SCM metadata falls back to `0.1.0`.

## MVP limitations

The first cut intentionally does not yet include cross-process install locks, resumable downloads, progress callbacks, model-license presentation, release-grade waveform parity gates, or a stable third-party catalog schema. The current Piper and Kokoro catalog readers are adapters around the repositories that already exist.

## License

The `onnxvoice` source code is Apache-2.0. Downloaded models, voice packs and model cards retain their own licenses and terms; installing them through `onnxvoice` does not relicense those artifacts.
