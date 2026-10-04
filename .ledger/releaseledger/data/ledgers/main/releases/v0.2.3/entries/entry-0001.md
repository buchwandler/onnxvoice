---
schema_version: 2
object_type: release_entry
versioning:
  schema_version: 1
  revision: 1
entry_id: entry-0001
release_version: v0.2.3
kind: fixed
summary:
  Fixed Kokoro catalogs to expose Inno voice enrollment only when the selected
  distribution includes both required assets
status: accepted
audience: null
scopes: []
source_refs:
  - git:2ee72dfb38b0cd3d60cd97bf673955836e4fbd91
paths:
  - onnxvoice/catalog.py
  - tests/test_catalog.py
issues: []
prs: []
sources:
  - git:2ee72dfb38b0cd3d60cd97bf673955836e4fbd91
contributors:
  - "@holgern"
breaking: false
internal: false
order: 1
---
