---
schema_version: 2
object_type: release_entry
versioning:
  schema_version: 1
  revision: 1
entry_id: entry-0004
release_version: v0.2.2
kind: changed
summary:
  Changed predefined Pocket voice states to require the safetensors format
  at catalog and runtime parse time
status: accepted
audience: null
scopes: []
source_refs: []
paths:
  - onnxvoice/catalog.py
  - onnxvoice/systems/pocket.py
  - onnxvoice/catalog_tools/pocket.py
issues: []
prs: []
sources:
  - git:99acd679c22f68420ef50ea3bae5db04fe453107
contributors:
  - "@holgern"
breaking: false
internal: false
order: 4
---
