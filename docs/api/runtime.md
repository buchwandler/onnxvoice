# Runtime

The runtime module separates provider-policy capture from resolution against providers reported by ONNX Runtime. Importing it does not import the optional `onnxruntime` dependency; status inspection catches absent or broken imports and returns structured diagnostics. `runtime_status()` is read-only and does not probe inference readiness.
ONNX session-creation errors include the graph component when known, the resolved execution providers, and the model filename. The original ONNX Runtime exception remains available as the error's cause.

## Phonon runtime API

`PhononAdapter`, `PhononVoiceState`, and `PhononStreamChunk` are public runtime types defined in `onnxvoice.systems.phonon` and re-exported from `onnxvoice`. The adapter accepts model-ready token IDs and returns streaming mono PCM; see the [Phonon system guide](../systems/phonon.md) for the generation controls and low-level boundary.

```{automodule} onnxvoice.runtime
:members: OnnxSession, available_providers, normalize_provider_name, provider_preference, resolve_provider_preference, resolve_providers, runtime_distributions, runtime_status, get_default_providers, set_default_providers, clear_default_providers
:show-inheritance:
```
