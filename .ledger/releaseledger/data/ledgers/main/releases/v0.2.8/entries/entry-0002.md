---
schema_version: 2
object_type: release_entry
versioning:
  schema_version: 1
  revision: 1
entry_id: entry-0002
release_version: v0.2.8
kind: fixed
summary: Fixed WaveKat cache graph contracts for Base and VoiceDesign CPU inference
status: accepted
audience: null
scopes: []
source_refs:
- git:237a15ee8ac0d2b257a8bf9c221a17252ad198c3
paths:
- .gitignore
- docs/systems/index.md
- docs/systems/qwen3-release-validation.md
- docs/systems/qwen3.md
- onnxvoice/catalog_tools/qwen3.py
- onnxvoice/systems/qwen3_contract.py
- pyproject.toml
- tests/test_qwen3_contract.py
- tests/test_qwen3_integration.py
issues: []
prs: []
sources:
- git:237a15ee8ac0d2b257a8bf9c221a17252ad198c3
contributors:
- '@holgern'
breaking: false
internal: false
order: 2
---
