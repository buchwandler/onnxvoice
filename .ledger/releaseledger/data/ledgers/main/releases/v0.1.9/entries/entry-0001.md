---
schema_version: 2
object_type: release_entry
versioning:
  schema_version: 1
  revision: 1
entry_id: entry-0001
release_version: v0.1.9
kind: changed
summary:
  Changed Pocket catalog handling to enforce canonical schema and validate
  discovered artifacts
status: accepted
audience: null
scopes: []
source_refs:
  - git:5ef09c9a26b9ac51d7e08a9c786a10c5deccf15d
paths:
  - onnxvoice/catalog.py
  - onnxvoice/catalog_tools/pocket.py
  - tests/test_pocket_catalog.py
  - tests/test_pocket_catalog_tools.py
  - tests/test_pocket_repository_contract.py
issues: []
prs: []
sources:
  - git:5ef09c9a26b9ac51d7e08a9c786a10c5deccf15d
contributors:
  - "@holgern"
breaking: false
internal: false
order: 1
---
