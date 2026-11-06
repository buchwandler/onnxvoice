---
schema_version: 2
object_type: release_entry
versioning:
  schema_version: 1
  revision: 1
entry_id: entry-0001
release_version: v0.2.17
kind: added
summary: Added Kokoro language-router support for English and German model targets
status: accepted
audience: null
scopes: []
source_refs:
  - git:547aafa73268181e3e91db5226461c4f9b1282dd
paths:
  - onnxvoice/catalog.py
  - onnxvoice/engine_options.py
  - onnxvoice/inventory.py
  - onnxvoice/systems/kokoro.py
  - onnxvoice/systems/kokoro_language_router.py
  - onnxvoice/systems/kokoro_static.py
  - tests/test_catalog.py
  - tests/test_engine_options.py
  - tests/test_inventory.py
  - tests/test_kokoro_contract.py
  - tests/test_kokoro_language_router.py
  - tests/test_manager_cache.py
issues: []
prs: []
sources:
  - git:547aafa73268181e3e91db5226461c4f9b1282dd
contributors:
  - "@holgern"
breaking: false
internal: false
order: 1
---
