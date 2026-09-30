# ONNX Runtime providers

ONNX Runtime is optional. The base `onnxvoice` package can manage catalog and storage data without installing a runtime. Install a compatible runtime distribution for inference:

```bash
python -m pip install "onnxvoice[cpu]"
python -m pip install "onnxvoice[gpu]"
python -m pip install "onnxvoice[directml]"
python -m pip install "onnxvoice[openvino]"
```

The `coreml`, `nnapi`, `xnnpack`, and `mobile` extras are markers. They do not necessarily install a provider implementation. Compatible platform ONNX Runtime distributions supply those providers.

## Provider names

`OnnxVoice.open()` and `OnnxVoice.open_local()` accept `provider` or `providers`. Supported aliases include:

| Alias             | ONNX Runtime provider       |
| ----------------- | --------------------------- |
| `cpu`             | `CPUExecutionProvider`      |
| `cuda`, `gpu`     | `CUDAExecutionProvider`     |
| `tensorrt`        | `TensorrtExecutionProvider` |
| `rocm`            | `ROCMExecutionProvider`     |
| `directml`, `dml` | `DmlExecutionProvider`      |
| `openvino`        | `OpenVINOExecutionProvider` |
| `coreml`          | `CoreMLExecutionProvider`   |
| `nnapi`           | `NnapiExecutionProvider`    |
| `xnnpack`         | `XnnpackExecutionProvider`  |

Canonical ONNX Runtime provider names are accepted as well. A comma-separated string or a sequence can request an ordered provider list. Requested providers must be available in the installed ONNX Runtime build.

`auto` selects the first available provider in this deterministic order: TensorRT, CUDA, ROCm, DirectML, OpenVINO, CoreML, XNNPACK, NNAPI, then CPU. It selects one provider.

## Configuration and precedence

An explicit Python `provider` or `providers` argument takes precedence. Otherwise, the runtime reads `ONNXVOICE_PROVIDERS`, then `ONNXVOICE_PROVIDER`, and defaults to `cpu`.

```python
from onnxvoice import OnnxVoice

voice = OnnxVoice()
runtime = voice.open("piper:en_US-lessac-medium", provider="cpu")
```

Use environment variables when an application needs a process-wide default. Prefer an explicit argument when a particular session needs a specific provider.

## Diagnostics

Runtime diagnostics report the provider request and the active provider for each session. Multi-session adapters expose one record per session:

```python
from onnxvoice import OnnxVoice

voice = OnnxVoice()
runtime = voice.open("piper:en_US-lessac-medium")
try:
    diagnostics = runtime.diagnostics()
    for session in diagnostics.sessions:
        print(session.providers_requested)
        print(session.providers_active)
finally:
    runtime.close()
```

Provider diagnostics describe the session that was created. They do not validate model quality or benchmark throughput.
