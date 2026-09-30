---
schema_version: 2
object_type: release_entry
versioning:
  schema_version: 1
  revision: 1
entry_id: entry-0002
release_version: v0.1.13
kind: added
summary:
  Added Supertonic multi-component ONNX inference with configurable steps,
  speed, and deterministic seeding
status: accepted
audience: null
scopes: []
source_refs:
  - git:54301b0d50639a64e705e53356f2ea7c0bb8a7a3
paths:
  - docs/Makefile
  - docs/conf.py
  - docs/make.bat
  - docs/make.py
  - docs/requirements.txt
  - onnxvoice/manager.py
  - onnxvoice/systems/__init__.py
  - onnxvoice/systems/supertonic.py
  - pyproject.toml
  - tests/test_supertonic_adapter.py
  - tests/test_supertonic_local_open.py
  - tests/test_supertonic_parity.py
issues: []
prs: []
sources:
  - git:54301b0d50639a64e705e53356f2ea7c0bb8a7a3
contributors:
  - "@holgern"
breaking: false
internal: false
order: 2
---
