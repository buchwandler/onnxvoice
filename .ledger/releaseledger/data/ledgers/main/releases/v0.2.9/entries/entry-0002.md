---
schema_version: 2
object_type: release_entry
versioning:
  schema_version: 1
  revision: 1
entry_id: entry-0002
release_version: v0.2.9
kind: added
summary:
  Added install progress, configurable shared cache, and interruption-safe
  asset replacement
status: accepted
audience: null
scopes: []
source_refs:
  - git:12de613c204e78d2ffafe6da4fc355c6c0ce49a8
paths:
  - README.md
  - docs/cli.md
  - onnxvoice/_install_progress.py
  - onnxvoice/cli.py
  - onnxvoice/manager.py
  - onnxvoice/store.py
  - pyproject.toml
  - tests/test_install_progress_cli.py
  - tests/test_manager_cache.py
  - tests/test_store_progress.py
issues: []
prs: []
sources:
  - git:12de613c204e78d2ffafe6da4fc355c6c0ce49a8
contributors:
  - "@holgern"
breaking: false
internal: false
order: 2
---
