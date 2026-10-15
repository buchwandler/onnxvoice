---
schema_version: 2
object_type: release_entry
versioning:
  schema_version: 1
  revision: 1
entry_id: entry-0001
release_version: v0.2.8
kind: added
summary: Added managed Qwen3 WaveKat inference for Base and VoiceDesign profiles
status: accepted
audience: null
scopes: []
source_refs:
  - git:c44e2129afd3085da32ce22c9704d3d1f8f4078d
paths:
  - README.md
  - docs/cli.md
  - docs/index.md
  - docs/local-models.md
  - docs/python-api.md
  - docs/systems/index.md
  - docs/systems/qwen3.md
  - onnxvoice/__init__.py
  - onnxvoice/catalog.py
  - onnxvoice/catalog_tools/__init__.py
  - onnxvoice/catalog_tools/qwen3.py
  - onnxvoice/catalog_tools/qwen3_seed.json
  - onnxvoice/cli.py
  - onnxvoice/manager.py
  - onnxvoice/runtime.py
  - onnxvoice/systems/__init__.py
  - onnxvoice/systems/qwen3.py
  - onnxvoice/systems/qwen3_audio.py
  - onnxvoice/systems/qwen3_contract.py
  - onnxvoice/systems/qwen3_generation.py
  - onnxvoice/systems/qwen3_sampling.py
  - onnxvoice/types.py
  - pyproject.toml
  - tests/fixtures/qwen3_log_mel_golden.json
  - tests/test_qwen3_audio.py
  - tests/test_qwen3_catalog.py
  - tests/test_qwen3_contract.py
  - tests/test_qwen3_generation.py
  - tests/test_qwen3_integration.py
  - tests/test_qwen3_local_open.py
  - tests/test_runtime.py
issues: []
prs: []
sources:
  - git:c44e2129afd3085da32ce22c9704d3d1f8f4078d
contributors:
  - "@holgern"
breaking: false
internal: false
order: 1
---
