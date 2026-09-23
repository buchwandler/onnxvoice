---
schema_version: 2
object_type: release_entry
versioning:
  schema_version: 1
  revision: 2
entry_id: entry-0001
release_version: v0.1.12
kind: added
summary:
  Added built-in Pocket voice conditioning with downloadable predefined voices
  and validated voice-state data
status: accepted
audience: null
scopes: []
source_refs:
  - git:6b3080a162b310688c5e486fad23a0da29029289
paths:
  - onnxvoice/catalog.py
  - onnxvoice/data/voice_selectors.json
  - onnxvoice/errors.py
  - onnxvoice/manager.py
  - onnxvoice/systems/pocket.py
  - pyproject.toml
  - tests/test_catalog.py
  - tests/test_pocket_adapter.py
  - tests/test_pocket_catalog.py
  - tests/test_pocket_network_boundaries.py
  - tests/test_voice_selectors.py
issues: []
prs: []
sources:
  - git:6b3080a162b310688c5e486fad23a0da29029289
contributors:
  - "@holgern"
breaking: false
internal: false
order: 1
---
