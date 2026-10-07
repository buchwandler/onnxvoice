---
schema_version: 2
object_type: release_entry
versioning:
  schema_version: 1
  revision: 1
entry_id: entry-0001
release_version: v0.2.5
kind: fixed
summary:
  Fixed cross-process model-store locking and coalesced concurrent catalog
  refreshes
status: draft
audience: null
scopes: []
source_refs:
  - git:e01f3b37ff07d24b13d373dce59ff0376df9bfd8
paths:
  - onnxvoice/catalog.py
  - onnxvoice/store.py
  - tests/test_concurrency.py
issues: []
prs: []
sources:
  - git:e01f3b37ff07d24b13d373dce59ff0376df9bfd8
  - tl:task-0028
contributors:
  - "@holgern"
breaking: false
internal: false
order: 1
---

File-backed OS advisory locks now coordinate thread/process access and are released automatically on process exit. Concurrent catalog fetches share a cache generation rather than issuing duplicate requests.
