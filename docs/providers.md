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

## Runtime distributions and execution providers

An ONNX Runtime **distribution** is a Python package/build that supplies the `onnxruntime` import (for example, `onnxruntime`, `onnxruntime-gpu`, or a vendor build). An **execution provider** is an inference backend reported by that imported build (for example, `CUDAExecutionProvider` or `CPUExecutionProvider`). They are different: choosing CUDA does not replace the installed Python distribution.

Inspect the imported build, installed distribution metadata, and providers reported by ORT with:

```bash
onnxvoice runtime status
onnxvoice runtime list
```

“Reported” or “available” means the imported build lists the provider. It is not a hardware, driver, or native-library self-test; a provider can still fail when a session initializes. ONNXVoice does not benchmark providers or perform network checks as part of status.

## Select a default provider policy

`runtime use` persists an ordered provider policy for new ONNXVoice sessions. It validates the requested policy against providers reported by the currently imported build and never installs or uninstalls a distribution:

```bash
onnxvoice runtime use cuda
onnxvoice runtime use cpu
onnxvoice runtime use auto
onnxvoice runtime use cuda,cpu
onnxvoice runtime reset
```

The saved JSON preference is stored at `platformdirs.user_config_path("onnxvoice") / "runtime.json"`. Reset removes that preference but does not unset environment variables. Existing initialized sessions are not modified; close and reopen them or pass an explicit provider when opening a runtime.

## Configuration and precedence

The effective policy follows this order:

1. Explicit Python `provider=` or `providers=` argument.
2. `ONNXVOICE_PROVIDERS` environment variable (ordered comma-separated policy).
3. `ONNXVOICE_PROVIDER` environment variable.
4. Saved ONNXVoice user configuration.
5. Built-in `cpu` default.

An explicit Python argument remains the most deterministic choice. Environment variables continue to override saved configuration, so one user-level saved preference can be adjusted per process or environment. Ordered multi-provider lists preserve the order supplied; ONNXVoice does not add an implicit CPU fallback. `auto` selects one provider using the deterministic priority described above.

The same controls are available through the stable Python API:

```python
import onnxvoice

status = onnxvoice.runtime_status()
print(status.imported_distribution)
print(status.available_providers)
print(status.configured_providers, status.provider_source)
print(status.resolved_providers)

onnxvoice.set_default_providers("cuda")
# Remove the persisted preference and return to environment/default precedence:
onnxvoice.clear_default_providers()
```

The base `onnxvoice` package remains usable without ONNX Runtime. Selecting a provider requires an importable runtime that reports that provider; package/distribution changes are handled by the Python environment's package manager.

## Active session diagnostics

Runtime status describes the policy for **new** sessions; it does not imply a provider is already active. `SessionDiagnostic` distinguishes the requested policy and its source, the resolved provider list, and providers actually registered by an initialized session:

```python
from onnxvoice import OnnxVoice

voice = OnnxVoice()
runtime = voice.open("piper:en_US-lessac-medium")
try:
    for session in runtime.diagnostics().sessions:
        print(session.providers_requested)
        print(session.provider_source)
        print(session.providers_resolved)
        print(session.providers_active)
finally:
    runtime.close()
```

`providers_active` is session-specific and comes from `InferenceSession.get_providers()`; there is no global active provider before a session exists. A preference change affects future sessions only. For a workload-specific override, pass `provider="cuda"` to `open()` or `open_local()`.

## KittenTTS 2 component sessions

KittenTTS 2 creates four lazy ONNX sessions for its LM, speaker, flow, and mel2wav components. The selected provider list and provider options are passed to each component session without adding an implicit CPU fallback. A bundle LM profile that requires low-memory exact-ternary execution is CPU-only and uses its required component-specific session settings; conflicting caller-provided LM options are rejected. See [KittenTTS 2](systems/kitten2.md).

Provider diagnostics describe the session that was created. They do not validate model quality or benchmark throughput.
