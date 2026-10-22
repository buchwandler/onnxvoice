---
schema_version: 2
object_type: release_entry
versioning:
  schema_version: 1
  revision: 1
entry_id: entry-0005
release_version: v0.2.13
kind: internal
summary: Improved static typing across runtime, adapter, catalog, and test code
status: accepted
audience: null
scopes: []
source_refs:
  - git:3566b429f5b980db31d69d0f62b04ba9ce0200dd
paths:
  - onnxvoice/runtime.py
  - onnxvoice/systems/pocket.py
  - tests/test_cli_voices.py
  - tests/test_huggingface.py
  - tests/test_inflect_adapter.py
  - tests/test_kitten2_generation.py
  - tests/test_kitten2_integration.py
  - tests/test_kitten2_tokenizer.py
  - tests/test_pocket_adapter.py
  - tests/test_pocket_catalog.py
  - tests/test_qwen3_catalog.py
  - tests/test_qwen3_contract.py
  - tests/test_sopro_local_open.py
  - tests/test_store.py
  - tests/test_supertonic_adapter.py
  - tests/test_supertonic_catalog.py
  - tests/test_voices.py
issues: []
prs: []
sources:
  - git:3566b429f5b980db31d69d0f62b04ba9ce0200dd
contributors:
  - "@holgern"
breaking: false
internal: true
order: 5
---
