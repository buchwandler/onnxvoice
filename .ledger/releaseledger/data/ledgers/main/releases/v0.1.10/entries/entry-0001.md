---
schema_version: 2
object_type: release_entry
versioning:
  schema_version: 1
  revision: 1
entry_id: entry-0001
release_version: v0.1.10
kind: added
summary:
  Added validation for canonical Pocket catalog integrity and support for declared
  Pocket voice states
status: accepted
audience: null
scopes: []
source_refs:
  - git:d4e7f5166c7efc9501f44a64d3060993dac514dc
paths:
  - onnxvoice/catalog.py
  - onnxvoice/catalog_tools/pocket.py
  - onnxvoice/manager.py
  - onnxvoice/store.py
  - onnxvoice/voice_selectors.py
  - tests/test_pocket_catalog.py
  - tests/test_pocket_catalog_tools.py
  - tests/test_pocket_real_managed.py
  - tests/test_store.py
  - tests/test_voice_selectors.py
issues: []
prs: []
sources:
  - git:d4e7f5166c7efc9501f44a64d3060993dac514dc
contributors:
  - "@holgern"
breaking: false
internal: false
order: 1
---
