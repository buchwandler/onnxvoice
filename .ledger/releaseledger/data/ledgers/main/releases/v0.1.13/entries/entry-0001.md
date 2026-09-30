---
schema_version: 2
object_type: release_entry
versioning:
  schema_version: 1
  revision: 1
entry_id: entry-0001
release_version: v0.1.13
kind: added
summary:
  Added per-voice language, locale, label, and gender metadata with generic-compatible
  locale filtering
status: accepted
audience: null
scopes: []
source_refs:
  - git:eb0bb57ec96ceeb98908faf6cc0ec54e58c5dbaa
paths:
  - README.md
  - docs/architecture.md
  - onnxvoice/__init__.py
  - onnxvoice/catalog.py
  - onnxvoice/catalog_tools/piper.py
  - onnxvoice/catalog_tools/pocket.py
  - onnxvoice/catalog_tools/voice_selectors.py
  - onnxvoice/inventory.py
  - onnxvoice/manager.py
  - onnxvoice/types.py
  - onnxvoice/voice_selectors.py
  - tests/test_catalog_tools.py
  - tests/test_inventory.py
  - tests/test_pocket_catalog.py
  - tests/test_pocket_catalog_tools.py
  - tests/test_voice_selectors.py
issues: []
prs: []
sources:
  - git:eb0bb57ec96ceeb98908faf6cc0ec54e58c5dbaa
contributors:
  - "@holgern"
breaking: false
internal: false
order: 1
---
