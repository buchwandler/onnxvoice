# Catalog voices and semantic references

## Voice identity

OnnxVoice derives each voice's identity from normalized catalog data: `(system, asset_id, voice_id)`. Its semantic reference is `<system>:<asset-id>[/<voice-id>]`.

For catalog items that are themselves voices, the item ID is the voice ID and the reference has no suffix:

```text
piper:en_US-lessac-medium
```

For bundles or models that expose child voice IDs, include the voice ID after `/`:

```text
kokoro:v1.0/af_heart
pocket:english_2026-04/alba
```

The reference identifies a catalog voice, not an installation. `backing_ref` is the asset reference used to install or open the underlying asset. Resolving a Kokoro voice does not construct or choose a model-ready style tensor.

## Discovery and metadata

`OnnxVoice.list_voices()` reads the available catalog items and returns the voice IDs explicitly exposed by each normalized item. Voice discovery therefore follows catalog contents. It does not assign IDs or infer extra voices from descriptive names.

```python
from onnxvoice import OnnxVoice

voice = OnnxVoice()
for record in voice.list_voices(language="en-US"):
    print(record.ref, record.system, record.asset_id, record.voice_id)
```

`VoiceRecord` includes the semantic `ref`, `system`, `asset_id`, `voice_id`, and `backing_ref`, along with descriptive language, locale, label, and gender metadata. Language and gender describe a voice. They do not form part of its identity. Missing authoritative gender metadata is represented as `unknown`.

Pocket discovery is limited to voice IDs already present in normalized `CatalogItem.voices`. Predefined names or descriptive details outside that field are not added as voices by discovery.

## Resolve a voice

Use `resolve_voice()` to look up a semantic reference in the current catalog:

```python
from onnxvoice import OnnxVoice

voice = OnnxVoice()
record = voice.resolve_voice("kokoro:v1.0/af_heart")
print(record.backing_ref, record.voice_id)
```

For an asset that is itself a voice, resolve its asset reference without a suffix:

```python
from onnxvoice import OnnxVoice

voice = OnnxVoice()
record = voice.resolve_voice("piper:en_US-lessac-medium")
```

The resolver validates the asset and, when present, the child voice ID against normalized catalog data. See the [Python usage guide](python-api.md) for manager configuration and the [CLI reference](cli.md) for command-line discovery.
