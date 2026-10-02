---
schema_version: 2
object_type: release_entry
versioning:
  schema_version: 1
  revision: 1
entry_id: entry-0002
release_version: v0.2.1
kind: added
summary:
  Added optional Inno v0.2 enrollment and offline ONNX export for reusable
  Kokoro voice packs
status: accepted
audience: null
scopes: []
source_refs:
  - git:ae2684c5382921e48a11833b666c99284709f2e8
paths:
  - docs/systems/kokoro.md
  - onnxvoice/__init__.py
  - onnxvoice/catalog.py
  - onnxvoice/manager.py
  - onnxvoice/systems/__init__.py
  - onnxvoice/systems/kokoro.py
  - onnxvoice/systems/kokoro_cloning.py
  - onnxvoice/systems/kokoro_inno.py
  - pyproject.toml
  - tests/fixtures/inno_fbank.npz
  - tests/fixtures/inno_prosody.json
  - tests/test_catalog.py
  - tests/test_kokoro_cloning.py
  - tests/test_kokoro_inno.py
  - tests/test_kokoro_inno_export.py
  - tests/test_local_runtime.py
  - tools/export_kokoro_inno.py
issues: []
prs: []
sources:
  - git:ae2684c5382921e48a11833b666c99284709f2e8
contributors:
  - "@holgern"
breaking: false
internal: false
order: 2
---
