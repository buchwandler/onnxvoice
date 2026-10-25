---
schema_version: 2
object_type: release_entry
versioning:
  schema_version: 1
  revision: 1
entry_id: entry-0002
release_version: v0.2.14
kind: fixed
summary:
  Fixed Hugging Face downloads by retrying without Xet before using a pinned
  HTTPS fallback
status: accepted
audience: null
scopes: []
source_refs:
  - git:89bc62fb3ce0a66ffd4bddeb8a0f8f56278a54b2
paths:
  - onnxvoice/errors.py
  - onnxvoice/huggingface.py
  - tests/test_huggingface.py
  - tests/test_store.py
issues: []
prs: []
sources:
  - git:89bc62fb3ce0a66ffd4bddeb8a0f8f56278a54b2
contributors:
  - "@holgern"
breaking: false
internal: false
order: 2
---
