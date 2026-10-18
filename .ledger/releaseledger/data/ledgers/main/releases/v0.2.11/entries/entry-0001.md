---
schema_version: 2
object_type: release_entry
versioning:
  schema_version: 1
  revision: 1
entry_id: entry-0001
release_version: v0.2.11
kind: added
summary: Added Chatterbox Nano v2 and Turbo catalog/runtime support
status: accepted
audience: null
scopes: []
source_refs:
  - git:2836d4831f799139878289c5cc0b6a0e3e71512f
paths:
  - onnxvoice/catalog.py
  - onnxvoice/catalog_tools/chatterbox.py
  - onnxvoice/catalog_tools/chatterbox_seed.json
  - onnxvoice/cli.py
  - onnxvoice/systems/__init__.py
  - onnxvoice/systems/chatterbox.py
  - pyproject.toml
  - tests/test_chatterbox_catalog.py
  - tests/test_chatterbox_catalog_tools.py
  - tests/test_chatterbox_runtime.py
issues: []
prs: []
sources:
  - git:2836d4831f799139878289c5cc0b6a0e3e71512f
contributors:
  - "@holgern"
breaking: false
internal: false
order: 1
---

Adds revision-pinned bundle tooling for Nano v2 and Turbo, managed install/open, a cleanroom four-graph runtime, reusable prepared reference voice state, and catalog build/verify CLI support. Tokenization and the public synthesis API remain the frontend's responsibility.
