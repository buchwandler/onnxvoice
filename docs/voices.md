# Voice selectors and discovery

## Asset references and voice selectors are different

An asset reference identifies an installable catalog item, using the canonical `system:id` form:

```text
piper:en_US-lessac-medium
kokoro:v1.0
pocket:english_2026-04
```

A stable voice selector is a short persisted alias for a logical voice identity:

```text
de-ko-1
de-pi-1
en_us-ko-4
```

The selector does not identify a model installation, and resolving a Kokoro selector does not create or choose a model-ready style tensor.

## Canonical form

```text
<language-key>-<engine-code>-<slot>
```

| Engine code | System |
| ----------- | ------ |
| `ko`        | Kokoro |
| `pi`        | Piper  |
| `po`        | Pocket |

Language keys are normalized to lowercase with underscores, so `en-us-ko-4` canonicalizes to `en_us-ko-4`. Catalog locale, base language, and selector namespace are related but separate values. For example, a catalog locale can be `en-US`, its base language can be `en`, and its selector namespace can be `en_us`.

## Identity stability

Selectors are registry assignments, not positions in a live catalog response:

- Slots are append-only. Removed voices do not make their slots reusable.
- Sorting, filtering, installation state, or network availability cannot renumber a selector.
- Catalog voices can exist without a selector assignment. They are reported as unassigned rather than assigned a generated number.
- A selector resolves to a system, backing asset, and logical voice ID. It does not change the asset reference used for installation.
- Retired assignments remain part of the registry and can be included explicitly when listing or resolving.

Language filters use compatible language tags. `en` can match `en-US` and `en-GB`; `en-US` does not match `en-GB`. Gender comes only from authoritative metadata. Missing gender is represented as `unknown`.

## Pocket identity

Pocket selector identities are bundle-scoped. They require explicit catalog `voice_states` records. A name appearing only in `predefined_voice_names` or descriptive `voice_details` is not enough to allocate a stable selector. The current catalog may therefore expose Pocket voices without selector assignments.

## CLI discovery

```bash
onnxvoice voices list
onnxvoice voices list --lang en-US
onnxvoice voices list --system kokoro
onnxvoice voices list --include-retired
onnxvoice voices list --no-unassigned
onnxvoice voices show en_us-ko-4
```

Listing accepts `--refresh` and `--format table|plain|json|tsv`. `show` resolves a selector to its complete identity. See the [CLI reference](cli.md) for inventory filters.

## Python resolution

```python
from onnxvoice import resolve_voice_selector

identity = resolve_voice_selector("de-ko-1")
print(identity.system)
print(identity.backing_ref)
print(identity.voice_id)
```

The selector API resolves identity only. Producer/front-end packages remain responsible for semantic voice policy and model-specific inputs such as Kokoro style tensors.
