---
schema_version: 2
object_type: release_entry
versioning:
  schema_version: 1
  revision: 1
entry_id: entry-0002
release_version: v0.2.16
kind: fixed
summary:
  Fixed Phonon streaming to handle scalar decoder offsets and validate Mimi
  state tensor dtypes
status: accepted
audience: null
scopes: []
source_refs:
  - git:6658962fb7f91edd9bc4db81abd38fd41eedc9ee
paths:
  - onnxvoice/systems/phonon.py
  - tests/test_phonon_runtime.py
issues: []
prs: []
sources:
  - git:6658962fb7f91edd9bc4db81abd38fd41eedc9ee
contributors:
  - "@holgern"
breaking: false
internal: false
order: 2
---
