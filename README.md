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

Installations hard-link to immutable blobs when the platform and filesystem support hard links. Otherwise `onnxvoice` falls back to copying the verified blob into the installation directory. The blob cache remains content-addressed; copy-mode installations may use additional disk space. The same bytes therefore do not need to be stored twice by different model installations.

## Stable low-level contract

Milestone A defines the dependency boundary used by downstream frontends:

- Installation manifests use schema 2 and preserve artifact component, format, quality, and metadata fields. Schema 1 manifests remain readable.
- System, item, and artifact paths are validated before filesystem access.
- Installations, catalog writes, blob publication, and garbage collection use process locks. Interrupted staging is removed.
- Asset operations accept progress callbacks receiving `AssetProgress` events.
- The canonical inference result is float32, one-dimensional NumPy audio with a positive sample rate. Kokoro timing and named auxiliary outputs are available on the result.
- `open()` resolves and verifies an existing installation only. `open_local()` uses explicit local files without copying them into the shared cache. Call `install()` explicitly for catalog access and downloads.
- Provider names support aliases such as `cpu`, `cuda`, `gpu`, `directml`, and `openvino`. Use `auto` for deterministic priority selection, or set `ONNXVOICE_PROVIDER` / `ONNXVOICE_PROVIDERS` for an environment policy.
- Provider names support aliases `cpu`, `cuda`, `gpu`, `directml`, `dml`, `openvino`, `coreml`, `nnapi`, and `xnnpack`, plus canonical ONNX Runtime names. Use `auto` for the documented deterministic priority policy, or set `ONNXVOICE_PROVIDER` / `ONNXVOICE_PROVIDERS` for an explicit environment policy. The `coreml`, `nnapi`, `xnnpack`, and `mobile` extras are markers because compatible platform ONNX Runtime builds supply those providers.

The shared cache is never required for importing the package. Offline mode reads existing catalog and blob data only and does not make network requests.

## Stable voice selectors

Short voice selectors are persisted identity aliases, not positions in the current catalog. The canonical form is `<language-key>-<engine-code>-<slot>`:

```text
de-ko-1       -> kokoro:de-anna, logical voice df_anna
de-pi-1       -> piper:de_DE-eva_k-x_low
en_us-ko-12   -> a future Kokoro registry identity
```

`ko` is the permanent Kokoro code and `pi` is the permanent Piper code. Slots are append-only and remain reserved when a voice is removed, so catalog insertion, sorting, filtering, installation state, and network availability cannot silently rename an existing selector. Use the selector API to resolve the complete identity:

```python
from onnxvoice import resolve_voice_selector

identity = resolve_voice_selector("de-ko-1")
assert identity.backing_ref == "kokoro:de-anna"
assert identity.voice_id == "df_anna"
```

Asset operations still use canonical `system:id` references such as `kokoro:de-anna` and `piper:de_DE-eva_k-x_low`. Resolving a Kokoro selector does not choose a style tensor; producer packages remain responsible for style/policy selection. Catalog voices without registry assignments are reported as unassigned rather than receiving a runtime-generated number.

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

### Inventory and discovery

```bash
# Everything installed locally
onnxvoice installed

# Installed US English voices
onnxvoice installed --kind voice --lang en-US

# Installed male US English voices
onnxvoice installed --kind voice --lang en-US --gender male

# Installed + available male US English voices (merged inventory)
onnxvoice list --kind voice --lang en-US --gender male

# Only available (not installed) entries
onnxvoice list --status available

# JSON output for scripting
onnxvoice installed --format json
```

### Updates

```bash
# Check what's outdated (refreshes catalogs by default)
onnxvoice updates

# Check against cached catalogs only
onnxvoice updates --cached

# Update one asset (preserves quality/distribution selection)
onnxvoice update piper:en_US-lessac-medium

# Update all outdated assets
onnxvoice update --all
```

### Storage

```bash
# Show cache usage with byte totals
onnxvoice cache info

# Show cache usage as JSON
onnxvoice cache info --format json

# Show what GC would remove
onnxvoice cache gc --dry-run

# Remove orphaned blobs
onnxvoice cache gc
```

### Removal

```bash
# Show what would be removed (dry run)
onnxvoice remove piper:en_US-lessac-medium --dry-run

# Remove one asset
onnxvoice remove piper:en_US-lessac-medium

# Remove and clean up orphaned blobs
onnxvoice remove piper:en_US-lessac-medium --gc

# Remove multiple assets
onnxvoice remove piper:en_US-lessac-medium kokoro:v1.0

# Remove all variants of a ref
onnxvoice remove kokoro:v1.0 --all-variants --yes
```

### Info

```bash
# Detailed info about an installed asset
onnxvoice info piper:en_US-lessac-medium

# Info with update check
onnxvoice info piper:en_US-lessac-medium --check-updates
```

### Install and verify

```bash
onnxvoice install piper:en_US-lessac-medium
onnxvoice install kokoro:v1.0 --quality fp16
onnxvoice path piper:en_US-lessac-medium
onnxvoice verify piper:en_US-lessac-medium
onnxvoice show piper:en_US-lessac-medium
```

### Catalog management

```bash
onnxvoice catalog piper build --output catalog/voices.json --source-output catalog/source.json
onnxvoice catalog piper verify --catalog catalog/voices.json --source catalog/source.json
```

### Filter flags

Common flags for `list`, `installed`, and `updates`:

| Flag                  | Description                             |
| --------------------- | --------------------------------------- | ------- | ---------------------------- | ---------------- |
| `--system piper       | kokoro                                  | pocket` | Filter by system             |
| `--kind voice         | model                                   | bundle` | Filter by kind               |
| `--lang / --language` | Filter by language (e.g. `en`, `en-US`) |
| `--gender male        | female                                  | neutral | unknown`                     | Filter by gender |
| `--quality`           | Filter by quality                       |
| `--distribution`      | Filter by distribution                  |
| `--status installed   | available                               | local`  | Filter by status (list only) |
| `--format table       | plain                                   | json    | tsv`                         | Output format    |

**Note on gender**: Gender metadata depends on authoritative catalog sources. If the catalog does not supply a gender field, entries default to `unknown`. The CLI never infers gender from voice names or IDs.

**Note on installed vs cached blobs**: After `onnxvoice remove REF`, the installation is gone but content-addressed blobs may remain until `onnxvoice cache gc` is run. Use `cache info` to see reclaimable space.

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

The same call works for the catalog's `split-onnx-v1` layout. The frontend still supplies token IDs and a complete style row; OnnxVoice does not select voices or phonemize text.

### Local split Kokoro

```python
runtime = open_local(
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
result = runtime.infer(token_ids, style=style, speed=1.0, seed=1234)
```

Runtime diagnostics are available through `runtime.diagnostics()` for both single and multi-session layouts.

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
        │   ├── KokoroAdapter
        │   └── SplitKokoroRuntime (prosody / curves / decoder)
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

The current release supports the built-in Piper and Kokoro catalog formats, single-file Kokoro, and the first-class `split-onnx-v1` multi-component Kokoro layout. Catalog distributions are selectable by identifier and cached with distinct identities. Resumable downloads, general third-party catalog schemas, and release-grade waveform parity gates remain separate work.

## License

The `onnxvoice` source code is Apache-2.0. Downloaded models, voice packs and model cards retain their own licenses and terms; installing them through `onnxvoice` does not relicense those artifacts.
