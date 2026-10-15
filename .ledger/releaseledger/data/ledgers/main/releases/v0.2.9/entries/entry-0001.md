---
schema_version: 2
object_type: release_entry
versioning:
  schema_version: 1
  revision: 1
entry_id: entry-0001
release_version: v0.2.9
kind: added
summary:
  Added Piper sequence-axis metadata without treating fixed ONNX dimensions
  as phoneme limits
status: accepted
audience: null
scopes: []
source_refs:
  - git:fe650b312fdac938b584d84ed712b6584fb646d0
paths:
  - docs/systems/piper.md
  - onnxvoice/__init__.py
  - onnxvoice/systems/piper.py
  - onnxvoice/types.py
  - tests/test_piper_contract.py
  - tests/test_piper_integration.py
issues: []
prs: []
sources:
  - git:fe650b312fdac938b584d84ed712b6584fb646d0
contributors:
  - "@holgern"
breaking: false
internal: false
order: 1
---
