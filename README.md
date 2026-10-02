[![PyPI - Version](https://img.shields.io/pypi/v/onnxvoice)](https://pypi.org/project/onnxvoice/)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/onnxvoice)
[![codecov](https://codecov.io/gh/buchwandler/onnxvoice/graph/badge.svg?token=qKZyL4Zidh)](https://codecov.io/gh/buchwandler/onnxvoice)

# onnxvoice

`onnxvoice` manages, verifies, stores, and executes ONNX voice-model assets through shared catalogs, a content-addressed cache, and a common Python API.

It is a model asset and runtime layer, not a complete text-to-speech frontend. Text normalization, G2P/phonemization, sentence planning, and higher-level voice policy belong to producer packages.

## Install

The base package does not install ONNX Runtime. Install the provider extra for the target system:

```bash
python -m pip install onnxvoice
python -m pip install "onnxvoice[cpu]"
python -m pip install "onnxvoice[gpu]"
python -m pip install "onnxvoice[directml]"
python -m pip install "onnxvoice[openvino]"
```

For Pocket's optional Hugging Face voice-state downloads, add the `pocket` extra. See [Pocket download setup](docs/pocket-downloads.md).

Pocket reference-voice prompts are managed assets as well: `kyutai-tts-voices:<id>` references resolve to pinned, integrity-checked prompts. See [Python usage](docs/python-api.md).

## Quick start

### CLI

```bash
onnxvoice list --kind voice --lang en-US
onnxvoice install piper:en_US-lessac-medium
onnxvoice installed
onnxvoice cache info
```

### Python

```python
from onnxvoice import OnnxVoice

voice = OnnxVoice()
installation = voice.install("piper:en_US-lessac-medium")
runtime = voice.open(installation, provider="cpu")
try:
    # Supply model-ready token IDs from your frontend.
    result = runtime.infer([1, 20, 14, 5, 2])
    print(result.audio.shape, result.sample_rate)
finally:
    runtime.close()
```

`install()` acquires catalog assets. `open()` uses an existing managed installation and does not download assets. `open_local()` opens explicit files without copying or registering them.

## Supported systems

| System     | Built-in adapter | Catalog | Named voices      |
| ---------- | ---------------- | ------- | ----------------- |
| Piper      | Yes              | Yes     | Yes               |
| Kokoro     | Yes              | Yes     | Yes               |
| Pocket     | Yes              | Yes     | Catalog-dependent |
| Supertonic | Yes              | Yes     | Yes               |
| Kitten     | Yes              | Yes     | Yes               |

Catalog contents determine which named voices are available. Discovery does not depend on a separate capability list. Adapters execute system-specific model contracts and do not imply text normalization or phonemization support.

## Documentation

The [full documentation](docs/index.md) includes task-oriented CLI and Python guides, storage and update behavior, provider setup, local-model workflows, per-system runtime contracts, and the API reference.

## Development

```bash
python -m pip install -e ".[dev,cpu]"
pytest -q
python docs/make.py html
```

## License

The source code is Apache-2.0. Downloaded models, voices, and model cards retain their own licenses and terms. Installing an asset through `onnxvoice` does not relicense it.
