---
schema_version: 2
object_type: release_entry
versioning:
  schema_version: 1
  revision: 1
entry_id: entry-0001
release_version: v0.2.4
kind: fixed
summary:
  Fixed Pocket EOS handling with minimum-frame gating, tails, diagnostics,
  and Pocket-compatible temperature scaling
status: accepted
audience: null
scopes: []
source_refs:
  - git:b5380a29e5a39a5e6388a437e47baa54d0142ec6
paths:
  - docs/systems/pocket.md
  - onnxvoice/systems/pocket.py
  - tests/test_pocket_adapter.py
  - tests/test_pocket_real_bundle.py
issues: []
prs: []
sources:
  - git:b5380a29e5a39a5e6388a437e47baa54d0142ec6
contributors:
  - "@holgern"
breaking: false
internal: false
order: 1
---
