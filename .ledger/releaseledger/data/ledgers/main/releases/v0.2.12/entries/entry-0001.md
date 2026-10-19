---
schema_version: 2
object_type: release_entry
versioning:
  schema_version: 1
  revision: 1
entry_id: entry-0001
release_version: v0.2.12
kind: added
summary:
  Added native KittenTTS 2 support with expressive, chunked synthesis and an
  expanded voice catalog
status: accepted
audience: null
scopes: []
source_refs:
  - git:aff57a6a0e8402bfc177bb474e93e73aba2a0811
  - git:5fec35fcc982440ccb00de21b1cb30a4dfb7855b
  - git:930ae87af698e19ab4dd4f32af46b0c8bb2be44c
paths:
  - README.md
  - docs/architecture.md
  - docs/cli.md
  - docs/providers.md
  - docs/python-api.md
  - docs/systems/kitten2.md
  - onnxvoice/catalog.py
  - onnxvoice/catalog_tools/kitten2.py
  - onnxvoice/manager.py
  - onnxvoice/systems/kitten2.py
  - onnxvoice/systems/kitten2_chunking.py
  - onnxvoice/systems/kitten2_contract.py
  - onnxvoice/systems/kitten2_expression.py
  - onnxvoice/systems/kitten2_frontend.py
  - onnxvoice/systems/kitten2_tokenizer.py
  - tests/test_kitten2_catalog.py
  - tests/test_kitten2_chunking.py
  - tests/test_kitten2_chunking_integration.py
  - tests/test_kitten2_contract.py
  - tests/test_kitten2_expression.py
  - tests/test_kitten2_frontend.py
  - tests/test_kitten2_integration.py
  - tests/test_kitten2_local_open.py
  - tests/test_kitten2_prompt.py
  - tests/test_kitten2_real_bundle.py
  - tests/test_kitten2_tokenizer.py
issues: []
prs: []
sources:
  - git:aff57a6a0e8402bfc177bb474e93e73aba2a0811
  - git:5fec35fcc982440ccb00de21b1cb30a4dfb7855b
  - git:930ae87af698e19ab4dd4f32af46b0c8bb2be44c
contributors:
  - "@holgern"
breaking: false
internal: false
order: 1
---
