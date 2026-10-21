# Runtime

The runtime module separates provider-policy capture from resolution against providers reported by ONNX Runtime. Importing it does not import the optional `onnxruntime` dependency; status inspection catches absent or broken imports and returns structured diagnostics. `runtime_status()` is read-only and does not probe inference readiness.

```{automodule} onnxvoice.runtime
:members: OnnxSession, available_providers, normalize_provider_name, provider_preference, resolve_provider_preference, resolve_providers, runtime_distributions, runtime_status, get_default_providers, set_default_providers, clear_default_providers
:show-inheritance:
```
