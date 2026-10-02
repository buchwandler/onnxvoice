---
schema_version: 2
object_type: release_entry
versioning:
  schema_version: 1
  revision: 1
entry_id: entry-0001
release_version: v0.2.1
kind: added
summary:
  Added reference-conditioned Kokoro cloning with reusable model-bound reference
  states
status: accepted
audience: null
scopes: []
source_refs:
  - git:37bb0b89a4e86c423baed90baf64f420a1b731d5
paths:
  - docs/systems/kokoro.md
  - onnxvoice/__init__.py
  - onnxvoice/catalog.py
  - onnxvoice/manager.py
  - onnxvoice/systems/__init__.py
  - onnxvoice/systems/kokoro.py
  - onnxvoice/systems/kokoro_cloning.py
  - tests/fixtures/kokoro_cloning_reference_mel.npz
  - tests/test_catalog.py
  - tests/test_kokoro_cloning.py
  - tests/test_local_runtime.py
issues: []
prs: []
sources:
  - git:37bb0b89a4e86c423baed90baf64f420a1b731d5
contributors:
  - "@holgern"
breaking: false
internal: false
order: 1
---
