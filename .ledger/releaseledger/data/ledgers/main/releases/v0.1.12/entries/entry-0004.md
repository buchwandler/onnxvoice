---
schema_version: 2
object_type: release_entry
versioning:
  schema_version: 1
  revision: 2
entry_id: entry-0004
release_version: v0.1.12
kind: added
summary:
  Added Pocket to stable voice selectors and exposed predefined voice cache
  usage and cleanup
status: accepted
audience: null
scopes: []
source_refs:
  - git:5b474f750fc7311edae9c205621805de9f3ccb2c
paths:
  - README.md
  - docs/architecture.md
  - docs/pocket-downloads.md
  - onnxvoice/catalog_tools/voice_selectors.py
  - onnxvoice/cli.py
  - onnxvoice/data/voice_selectors.json
  - onnxvoice/manager.py
  - onnxvoice/store.py
  - onnxvoice/systems/pocket.py
  - onnxvoice/types.py
  - onnxvoice/voice_selectors.py
  - pyproject.toml
  - tests/test_cache_usage.py
  - tests/test_pocket_adapter.py
  - tests/test_store.py
  - tests/test_voice_selectors.py
issues: []
prs: []
sources:
  - git:5b474f750fc7311edae9c205621805de9f3ccb2c
contributors:
  - "@holgern"
breaking: false
internal: false
order: 4
---
