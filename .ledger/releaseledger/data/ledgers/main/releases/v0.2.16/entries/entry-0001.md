---
schema_version: 2
object_type: release_entry
versioning:
  schema_version: 1
  revision: 1
entry_id: entry-0001
release_version: v0.2.16
kind: fixed
summary:
  Fixed Phonon streaming to accept legacy Mimi state records by inferring dtype
  from ONNX tensor metadata
status: accepted
audience: null
scopes: []
source_refs:
  - git:6d5651117d75efbc90d3d675fc387a8e9ed849fa
paths:
  - onnxvoice/systems/phonon.py
  - tests/test_phonon_runtime.py
issues: []
prs: []
sources:
  - git:6d5651117d75efbc90d3d675fc387a8e9ed849fa
contributors:
  - "@holgern"
breaking: false
internal: false
order: 1
---
