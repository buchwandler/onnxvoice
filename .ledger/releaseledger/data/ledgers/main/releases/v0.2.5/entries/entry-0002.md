---
schema_version: 2
object_type: release_entry
versioning:
  schema_version: 1
  revision: 2
entry_id: entry-0002
release_version: v0.2.5
kind: added
summary: Added native ONNXVoice runtime support for Inflect v2 Nano and Micro models
status: draft
audience: null
scopes: []
source_refs:
  - git:61a10912b71732b9917c4886298b4e1822ee6ec6
paths:
  - MANIFEST.in
  - README.md
  - docs/cli.md
  - docs/concepts.md
  - docs/index.md
  - docs/systems/index.md
  - docs/systems/inflect.md
  - onnxvoice/catalog.py
  - onnxvoice/store.py
  - onnxvoice/systems/__init__.py
  - onnxvoice/systems/inflect.py
  - pyproject.toml
  - tests/fixtures/inflect_models.json
  - tests/test_inflect_adapter.py
  - tests/test_inflect_catalog.py
  - tests/test_inflect_local_open.py
  - tests/test_inflect_voices.py
issues: []
prs: []
sources:
  - git:61a10912b71732b9917c4886298b4e1822ee6ec6
  - tl:task-0028
contributors:
  - "@holgern"
breaking: false
internal: false
order: 2
---

The adapter parses the pinned Inflect bundle catalog, installs and opens the split duration/decode graphs, and accepts model-ready token IDs with speed, variation, and deterministic seed controls. Fixed default-voice discovery and 24 kHz mono output validation use ONNXVoice's common provider and session lifecycle. Real CPU inference on both pinned models also verified the frontend/runtime path and led to stricter validation that correctly accepts the graphs' singleton-channel y_mask while preserving latent batch/frame checks.
