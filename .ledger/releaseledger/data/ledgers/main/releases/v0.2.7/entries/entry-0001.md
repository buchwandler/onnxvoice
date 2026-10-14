---
schema_version: 2
object_type: release_entry
versioning:
  schema_version: 1
  revision: 2
entry_id: entry-0001
release_version: v0.2.7
kind: added
summary:
  Added Sopro zero-shot voice support with managed bundles, reference preparation,
  and parity-gated native synthesis
status: accepted
audience: null
scopes: []
source_refs:
  - git:102ef950b0b3eb78f8ae1f55024a2f94954ce9c2
paths:
  - README.md
  - docs/local-models.md
  - docs/systems/index.md
  - docs/systems/sopro.md
  - onnxvoice/__init__.py
  - onnxvoice/catalog.py
  - onnxvoice/manager.py
  - onnxvoice/store.py
  - onnxvoice/systems/__init__.py
  - onnxvoice/systems/sopro.py
  - onnxvoice/systems/sopro_frontend.py
  - tests/fixtures/sopro/offline_pipeline_golden.json
  - tests/fixtures/sopro/reference_golden.json
  - tests/fixtures/sopro/semantic_golden.json
  - tests/fixtures/sopro/tokenizer-v2-turbo.json.gz
  - tests/test_manifest.py
  - tests/test_sopro_adapter.py
  - tests/test_sopro_catalog.py
  - tests/test_sopro_frontend.py
  - tests/test_sopro_local_open.py
  - tests/test_sopro_real_bundle.py
issues: []
prs: []
sources:
  - git:102ef950b0b3eb78f8ae1f55024a2f94954ce9c2
contributors:
  - "@holgern"
breaking: false
internal: false
order: 1
---
