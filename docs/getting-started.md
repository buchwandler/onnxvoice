# Getting started

`onnxvoice` manages and executes ONNX voice-model assets through a shared local store and a common Python API. The typical flow is:

```text
catalog -> install -> verify -> open -> infer
```

Text normalization, G2P/phonemization, sentence planning, and higher-level voice policy belong to a producer or frontend package, not `onnxvoice`.

## Install

The base package does not install ONNX Runtime. Install the runtime distribution appropriate for the target machine:

```bash
python -m pip install onnxvoice
python -m pip install "onnxvoice[cpu]"
python -m pip install "onnxvoice[gpu]"
python -m pip install "onnxvoice[directml]"
python -m pip install "onnxvoice[openvino]"
```

Inspect the imported ONNX Runtime build and its reported providers after installation:

```bash
onnxvoice runtime status
```

To persist a default provider policy for future sessions, use `onnxvoice runtime use cuda`, `onnxvoice runtime use cpu`, or `onnxvoice runtime use auto`. This selects a provider; it does not change the installed ONNX Runtime distribution. See [ONNX Runtime providers](providers.md) for selection precedence and environment overrides.

Hugging Face download support is an engine-neutral transport extra; it does not install an inference runtime:

```bash
python -m pip install "onnxvoice[huggingface]"       # transport only
python -m pip install "onnxvoice[cpu,huggingface]"   # CPU runtime plus Hub downloads
```

Pocket's existing extra remains compatible and includes `huggingface_hub` plus `safetensors`; add the runtime provider needed by your installation:

```bash
python -m pip install "onnxvoice[pocket,cpu]"
```

Development installation:

```bash
python -m pip install -e ".[dev,cpu]"
```

See [ONNX Runtime providers](providers.md) for aliases and platform-specific provider notes. See [Pocket downloads](pocket-downloads.md) for gated voice-state access.

## First CLI workflow

List voices, install one catalog item, then inspect the local installation and cache:

```bash
onnxvoice list --kind voice --lang en-US
onnxvoice install piper:en_US-lessac-medium
onnxvoice installed
onnxvoice info piper:en_US-lessac-medium
onnxvoice cache info
```

### Chatterbox runtime

Install a managed Chatterbox bundle with the CPU profile, then follow the [Chatterbox system guide](systems/chatterbox.md) for the frontend/runtime boundary and Python example:

```bash
onnxvoice install chatterbox:nano-v2 --quality cpu
```

`list` combines catalog and installed inventory. `installed` is local-only and does not access the network. Use `onnxvoice --offline installed` to make the no-network intent explicit.

## First Python workflow

```python
from onnxvoice import OnnxVoice

voice = OnnxVoice()
installation = voice.install("piper:en_US-lessac-medium")
runtime = voice.open(installation, provider="cpu")

try:
    result = runtime.infer(
        [1, 20, 14, 5, 2],
        length_scale=1.0,
        noise_scale=0.667,
        noise_w=0.8,
    )
    print(result.audio.shape, result.sample_rate)
finally:
    runtime.close()
```

The Piper example supplies model-ready token IDs and a numeric speaker ID when required by the model. `onnxvoice` executes the graph. It does not convert text to phonemes or select a speaker by name.

## Offline mode

The CLI's global `--offline` option belongs before the command:

```bash
onnxvoice --offline installed
onnxvoice --offline list --status installed
```

Offline mode can read local installations and cached catalogs or assets. It cannot fetch a missing catalog, model artifact, or Pocket voice state. For the Python API, pass `offline=True` to `OnnxVoice`.
