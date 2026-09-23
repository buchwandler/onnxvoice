---
schema_version: 2
object_type: release_entry
versioning:
  schema_version: 1
  revision: 2
entry_id: entry-0002
release_version: v0.1.12
kind: added
summary:
  Added pinned Hugging Face asset downloads with integrity checks, cache reuse,
  and setup guidance for Pocket assets
status: accepted
audience: null
scopes: []
source_refs:
  - git:577fe9840787e07ff8fc6291a379d81357661836
paths:
  - README.md
  - docs/pocket-downloads.md
  - onnxvoice/catalog.py
  - onnxvoice/catalog_tools/pocket.py
  - onnxvoice/cli.py
  - onnxvoice/errors.py
  - onnxvoice/huggingface.py
  - onnxvoice/store.py
  - onnxvoice/systems/pocket.py
  - tests/test_cli.py
  - tests/test_huggingface.py
  - tests/test_pocket_adapter.py
  - tests/test_pocket_catalog.py
  - tests/test_pocket_catalog_tools.py
  - tests/test_pocket_contract.py
  - tests/test_store.py
issues: []
prs: []
sources:
  - git:577fe9840787e07ff8fc6291a379d81357661836
contributors:
  - "@holgern"
breaking: false
internal: false
order: 2
---
