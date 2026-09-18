[![PyPI - Version](https://img.shields.io/pypi/v/onnxvoice)](https://pypi.org/project/onnxvoice/)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/onnxvoice)
![PyPI - Downloads](https://img.shields.io/pypi/dm/onnxvoice)
[![codecov](https://codecov.io/gh/buchwandler/onnxvoice/graph/badge.svg?token=qKZyL4Zidh)](https://codecov.io/gh/buchwandler/onnxvoice)

# onnxvoice

`onnxvoice` is the published Python infrastructure package for shared ONNX voice-model catalogs, asset installation, integrity verification, ONNX Runtime sessions, and model tensor-contract execution.

Install the package and an optional ONNX Runtime provider with:

```bash
pip install onnxvoice
pip install "onnxvoice[cpu]"
pip install "onnxvoice[gpu]"
pip install "onnxvoice[directml]"
pip install "onnxvoice[openvino]"
```

Model catalogs and model artifacts remain external data. `onnxvoice` does not bundle model files or speech-engine policy.

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

## Stable low-level contract

Milestone A defines the dependency boundary used by downstream frontends:

- Installation manifests use schema 2 and preserve artifact component, format, quality, and metadata fields. Schema 1 manifests remain readable.
- System, item, and artifact paths are validated before filesystem access.
- Installations, catalog writes, blob publication, and garbage collection use process locks. Interrupted staging is removed.
- Asset operations accept progress callbacks receiving `AssetProgress` events.
- The canonical inference result is float32, one-dimensional NumPy audio with a positive sample rate. Kokoro timing and named auxiliary outputs are available on the result.
- `open()` resolves and verifies an existing installation only. `open_local()` uses explicit local files without copying them into the shared cache. Call `install()` explicitly for catalog access and downloads.
- Provider names support aliases such as `cpu`, `cuda`, `gpu`, `directml`, and `openvino`. Use `auto` for deterministic priority selection, or set `ONNXVOICE_PROVIDER` / `ONNXVOICE_PROVIDERS` for an environment policy.
- `load_local()` constructs an unmanaged runtime from local files and does not register or copy them into the shared cache.

The shared cache is never required for importing the package. Offline mode reads existing catalog and blob data only and does not make network requests.

## Install

The base package does not install ONNX Runtime. Choose the extra for the deployment provider:

```bash
pip install onnxvoice
pip install "onnxvoice[cpu]"
pip install "onnxvoice[gpu]"
pip install "onnxvoice[directml]"
pip install "onnxvoice[openvino]"
```

For development, install `onnxvoice[dev,cpu]`.

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

onnxvoice catalog piper build --output catalog/voices.json --source-output catalog/source.json
onnxvoice catalog piper verify --catalog catalog/voices.json --source catalog/source.json
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

`onnxvoice` expects already-tokenized Piper IDs and a model-ready numeric speaker ID when the graph has a `sid` input. It does not phonemize text or resolve speaker names.

```python
from onnxvoice import open

runtime = open("piper:en_US-lessac-medium")
result = runtime.infer(
    [1, 20, 14, 5, 2],
    speaker_id=0,
    length_scale=1.0,
    noise_scale=0.667,
    noise_w=0.8,
)

print(result.audio.dtype)
print(result.sample_rate)
runtime.close()
```

### Kokoro inference

Kokoro receives an explicit model-ready style tensor. Logical voice selection and style archives belong to the higher-level engine.

```python
runtime = open("kokoro:v1.0", quality="fp16")
result = runtime.infer(
    [50, 31, 12, 99],
    style=style_tensor,
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

runtime = ov.open("piper:my-voice")
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

Local files can be opened without cache registration:

```python
from onnxvoice import open_local

runtime = open_local(
    system="piper",
    model="voice.onnx",
    config="voice.onnx.json",
    provider="cpu",
)
```

The unmanaged runtime keeps the original file paths. Use `import_model()` when a durable managed installation and manifest are required.

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

## Current limitations

The current release supports the built-in Piper and Kokoro catalog formats and deterministic rejection of split or multi-component Kokoro layouts that the adapter cannot execute. Resumable downloads, general third-party catalog schemas, release-grade waveform parity gates, and downstream package bridge migrations remain separate work.

## License

The `onnxvoice` source code is Apache-2.0. Downloaded models, voice packs and model cards retain their own licenses and terms; installing them through `onnxvoice` does not relicense those artifacts.
