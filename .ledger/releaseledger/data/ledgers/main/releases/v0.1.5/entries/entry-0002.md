---
schema_version: 2
object_type: release_entry
versioning:
  schema_version: 1
  revision: 1
entry_id: entry-0002
release_version: v0.1.5
kind: internal
summary: Improved consistency across inventory, CLI, manager, store, and test code
status: accepted
audience: null
scopes: []
source_refs:
  - git:7357ca4b357aa6e04dbae20d27845c14063f6d89
paths:
  - onnxvoice/_table.py
  - onnxvoice/catalog.py
  - onnxvoice/cli.py
  - onnxvoice/inventory.py
  - onnxvoice/manager.py
  - onnxvoice/store.py
  - tests/test_cache_usage.py
  - tests/test_cli.py
  - tests/test_inventory.py
  - tests/test_updates.py
issues: []
prs: []
sources:
  - git:7357ca4b357aa6e04dbae20d27845c14063f6d89
contributors:
  - "@holgern"
breaking: false
internal: true
order: 2
---
