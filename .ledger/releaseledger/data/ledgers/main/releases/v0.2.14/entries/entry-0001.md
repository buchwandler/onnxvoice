---
schema_version: 2
object_type: release_entry
versioning:
  schema_version: 1
  revision: 1
entry_id: entry-0001
release_version: v0.2.14
kind: added
summary:
  Added structured Hugging Face transport errors and a safe pinned-HTTPS fallback
  when optional Xet support is unavailable
status: accepted
audience: null
scopes: []
source_refs:
  - git:0ed88350f75c0b538007d1cba634b8a50c979b83
paths:
  - onnxvoice/errors.py
  - onnxvoice/huggingface.py
  - tests/test_huggingface.py
  - tests/test_store.py
issues: []
prs: []
sources:
  - git:0ed88350f75c0b538007d1cba634b8a50c979b83
contributors:
  - "@holgern"
breaking: false
internal: false
order: 1
---
