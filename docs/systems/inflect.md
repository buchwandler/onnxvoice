# Inflect v2

ONNXVoice provides a low-level runtime adapter for Inflect Nano v2 and Inflect Micro v2. Both models use a fixed `default` voice and produce 24 kHz audio. The catalog carries the voice description (including locale and gender); the adapter does not infer or invent voice metadata.

Inflect v2 is a two-graph runtime:

| Artifact role | File            |
| ------------- | --------------- |
| `duration`    | `duration.onnx` |
| `decode`      | `decode.onnx`   |

The adapter creates and owns one lazy ONNX Runtime session per graph. ONNXVoice handles catalog discovery, pinned artifact downloads and SHA-256 verification, provider selection, session options, and runtime diagnostics. Model discovery and voice listing use catalog metadata and do not require installing the ONNX files.

## Runtime boundary

`InflectAdapter.infer()` accepts a non-empty sequence of model-ready integer token IDs. It does not accept text and does not import InflectG2P. Text normalization, G2P, and conversion to the model's token IDs belong to InflectSynth / InflectG2P.

Supported controls:

| Control     | Default |          Range | Effect                            |
| ----------- | ------: | -------------: | --------------------------------- |
| `speed`     |   `1.0` |    `0.5`–`2.0` | Scales duration as `1 / speed`    |
| `variation` | `0.667` |    `0.0`–`1.0` | Decoder noise scale               |
| `seed`      |     `0` | Python integer | Seeds request-local decoder noise |

A repeated request with the same tokens, controls, model, runtime stack, and seed produces the same decoder-noise tensor. The result is finite mono `float32` PCM at 24 kHz. ONNXVoice returns raw model audio: it does not clip, normalize loudness, fade edges, split text, concatenate chunks, or insert pauses.

## CLI

```bash
onnxvoice list --system inflect
onnxvoice install inflect:nano-v2
onnxvoice voices list --system inflect
onnxvoice voices show inflect:nano-v2/default
```

The catalog can be overridden with `ONNXVOICE_INFLECT_CATALOG`, set to a local JSON path or an HTTP(S) URL. The default source is the Inflect ONNX bundles catalog. Catalog aliases such as `inflect:nano` and `inflect:Inflect-Nano-v2-ONNX` resolve to the canonical `inflect:nano-v2` item.

## Catalog schema compatibility

The default catalog URL tracks the bundle repository's `main` branch, so schema 1 is kept backward compatible for the lifetime of ONNXVoice 0.2.x. If an incompatible schema is introduced, publish it at a separately versioned catalog endpoint and add explicit client support rather than changing the schema behind the existing URL.

## Managed Python usage

```python
from onnxvoice import OnnxVoice

manager = OnnxVoice()
installation = manager.install("inflect:nano-v2")
with manager.open(installation, providers="cpu") as runtime:
    result = runtime.infer(
        token_ids,
        speed=1.0,
        variation=0.667,
        seed=0,
    )
    audio = result.audio
    sample_rate = result.sample_rate  # 24000
```

`token_ids` must already be encoded by the producer frontend. The managed install contains both verified artifact roles; `open()` does not download or resolve a catalog entry.

## Local Python usage

Use the existing multi-artifact `open_local()` API when the graph files are already available. These unmanaged files are not copied into the ONNXVoice store.

```python
from onnxvoice import OnnxVoice

runtime = OnnxVoice.open_local(
    system="inflect",
    artifacts={
        "duration": "duration.onnx",
        "decode": "decode.onnx",
    },
    sample_rate=24000,
    runtime={
        "profile": "inflect-v2-split-v1",
        "layout": "split",
        "precision": "fp32",
    },
    metadata={"default_voice": "default", "voices": ["default"]},
    providers="cpu",
)
try:
    result = runtime.infer(token_ids, speed=1.1, variation=0.5, seed=42)
finally:
    runtime.close()
```

## Providers and diagnostics

Provider choice and provider/session options are managed by ONNXVoice and passed to both graph sessions. For example:

```python
runtime = manager.open(
    installation,
    providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    provider_options=[{"device_id": "0"}, {}],
    session_options=session_options,
)
```

The runtime diagnostic reports the split layout and each created component (`duration` and `decode`), its model path, provider selection, and graph inputs/outputs. Missing artifacts raise `CapabilityError`; incompatible graph names, types, shapes, or inference values raise `RuntimeContractError`.

The CLI's one-model import command is not a split-graph import workflow. Use the Python `open_local()` API for local Inflect files.
