# Runtime

The runtime module resolves provider aliases and creates ONNX Runtime sessions lazily. Importing this module does not itself import the optional `onnxruntime` dependency.

```{automodule} onnxvoice.runtime
:members: OnnxSession, available_providers, normalize_provider_name, resolve_providers
:show-inheritance:
```
