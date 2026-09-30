# Systems

Each system adapter owns the model-specific ONNX graph contract. It does not provide text normalization, G2P/phonemization, sentence planning, or application-level voice policy.

| System     | Built-in adapter | Built-in catalog | Stable selector support                              | Typical runtime inputs                                  |
| ---------- | ---------------- | ---------------- | ---------------------------------------------------- | ------------------------------------------------------- |
| Piper      | Yes              | Yes              | Yes                                                  | Token IDs and optional numeric speaker ID               |
| Kokoro     | Yes              | Yes              | Yes                                                  | Token IDs and caller-provided model-ready style         |
| Pocket     | Yes              | Yes              | Conditional on explicit voice-state identity records | Token IDs and prepared voice state                      |
| Supertonic | Yes              | Yes              | No                                                   | Token IDs, text mask, and caller-provided style tensors |

Adapter registration, catalog availability, and stable-selector assignment are independent capabilities. A built-in adapter does not by itself imply a built-in catalog or selector namespace.

```{toctree}
:maxdepth: 1

piper
kokoro
pocket
supertonic
```
