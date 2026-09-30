# onnxvoice

`onnxvoice` manages, verifies, stores, and executes ONNX voice-model assets through a shared local cache and a common Python API.

It provides:

- external model catalogs;
- deterministic model installation;
- content-addressed storage and checksum verification;
- installed and available inventory, plus model update checks;
- stable voice selectors;
- ONNX Runtime provider selection;
- system-specific runtime adapters;
- local unmanaged model opening.

It deliberately does not own text normalization, G2P/phonemization, sentence planning, or higher-level voice policy.

## Start here

- [Getting started](getting-started.md)
- [CLI reference and common workflows](cli.md)
- [Voice selectors and discovery](voices.md)
- [Python usage](python-api.md)
- [Storage and cache management](storage.md)
- [ONNX Runtime providers](providers.md)

```{toctree}
:maxdepth: 2
:caption: User guide
:hidden:

getting-started
concepts
cli
voices
storage
updates
providers
local-models
systems/index
python-api
pocket-downloads
```

```{toctree}
:maxdepth: 2
:caption: Reference
:hidden:

api/index
architecture
changelog
```
