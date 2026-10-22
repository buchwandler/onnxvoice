---
schema_version: 2
object_type: release_entry
versioning:
  schema_version: 1
  revision: 1
entry_id: entry-0002
release_version: v0.2.13
kind: added
summary:
  Added engine discovery commands for target-specific options, capabilities,
  and limits
status: accepted
audience: null
scopes: []
source_refs:
  - git:716c576a4f7b8358d27e51c0afaf737dbcf0c129
paths:
  - docs/cli.md
  - docs/systems/chatterbox.md
  - docs/systems/index.md
  - docs/systems/inflect.md
  - docs/systems/kitten.md
  - docs/systems/kitten2.md
  - docs/systems/kokoro.md
  - docs/systems/piper.md
  - docs/systems/pocket.md
  - docs/systems/qwen3.md
  - docs/systems/sopro.md
  - docs/systems/supertonic.md
  - onnxvoice/cli.py
  - onnxvoice/engine_options.py
  - onnxvoice/systems/pocket.py
  - tests/test_cli_engines.py
  - tests/test_docs_cli_coverage.py
  - tests/test_engine_options.py
  - tests/test_pocket_adapter.py
issues: []
prs: []
sources:
  - git:716c576a4f7b8358d27e51c0afaf737dbcf0c129
contributors:
  - "@holgern"
breaking: false
internal: false
order: 2
---
