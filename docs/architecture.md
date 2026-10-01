# OnnxVoice architecture

OnnxVoice executes installed voice-model runtimes and derives voice identity from normalized catalog data. Producer packages prepare token IDs, model-ready style rows, speed, and other producer-owned policy. Text normalization, language detection, G2P, phoneme segmentation, producer semantic voice aliases, sentence handling, trace objects, and final audio composition remain outside this package.

## Catalog-backed voice identity

A voice is identified by `(system, asset_id, voice_id)`. Its semantic reference is `<system>:<asset-id>[/<voice-id>]`: `piper:en_US-lessac-medium` identifies an asset that is itself a voice, while `kokoro:v1.0/af_heart`, `supertonic:supertonic-3/F1`, and `kitten:nano-0.8-int8/Bella` identify child voices in model or bundle catalog items. The reference identifies catalog data, while the backing asset reference remains the `system:asset_id` used for installation.

`OnnxVoice.list_voices()` discovers voices from normalized catalog items. It uses the item ID for items whose kind is `voice` and uses the explicitly exposed child IDs for items with `CatalogItem.voices`. `resolve_voice()` validates a semantic reference against current catalog data. Voice identity and discovery derive directly from the catalog.

Resolving a Kokoro voice identifies the catalog voice. It does not select or construct a model-ready style tensor; the producer remains responsible for that operation.

Pocket discovery follows the current normalized `CatalogItem.voices` field. It does not infer voice identities from descriptive or predefined names outside that field.

## Descriptive language and voice metadata

Language, locale, labels, and gender are descriptive metadata, not parts of voice identity. `VoiceRecord.languages` separately describes supported synthesis-language capabilities. A multilingual Supertonic style keeps one semantic ref, such as `supertonic:supertonic-3/F1`, across those languages. `locale` records the most specific source-backed language tag, while `language` is its lowercase base language. Language filters support generic and specific compatible tags; conflicting specific locales such as `en-US` and `en-GB` do not match.

Language labels prefer authoritative human-readable source names, then locale/base tags, and never use a bare region code such as `US`.

`VoiceRecord.metadata` is the normalized projection for language, locale, language label, and gender; the existing `languages` and `gender` fields remain available. Gender is taken only from explicit authoritative metadata, with `unknown` as the fallback. Pocket `voice_details` describes declared voices independently of identity.

## Runtime boundary

```text
producer frontend
    | token_ids + style + speed + seed
    v
OnnxVoice adapter
    | catalog / installation / providers / sessions / graph execution
    v
InferenceResult(audio, sample_rate, timings, outputs, metadata)
```

The Kokoro adapter dispatches from catalog runtime metadata. A single layout owns one ONNX session. `split-onnx-v1` owns prosody, curves, and decoder sessions and keeps source generation, duration expansion, and STFT preparation inside OnnxVoice. Both layouts expose the same `infer(token_ids, style=..., speed=..., seed=...)` contract.
The Kitten adapter accepts model-ready token IDs, a caller-selected style tensor, and speed. It validates the graph ABI, applies the required 5,000-sample output trim, and returns mono audio at 24 kHz; KittenSynth remains responsible for text processing and style selection.

## Installation metadata

Catalog and manifest metadata retain the runtime layout, selected distribution, sample rate, maximum token count, required components, style dimensions, and declared timing output. Artifacts are queried by role, component, and quality rather than by filename conventions. Explicit distribution selection receives a distinct cache identity.

## Pocket auxiliary state cache

Pocket predefined voice states keep their existing `pocket-voice-states` cache layout separate from content-addressed model blobs. Explicit catalog size and SHA-256 pins participate in the cache identity and are checked before a downloaded state is published. Valid legacy cache entries remain reusable offline when their recorded integrity matches the current explicit pin; a changed pin cannot reuse the old state.

`AssetStore.usage()` exposes `auxiliary_bytes`, `pocket_voice_state_count`, `pocket_voice_state_bytes`, `orphan_auxiliary_count`, and `orphan_auxiliary_bytes`. Garbage collection derives reachability from installed Pocket bundle variants and removes a state only after no installed matching variant references it. `GcReport.removed_bytes` remains the blob-only byte total for compatibility; `removed_auxiliary_count` and `removed_auxiliary_bytes` report state-cache reclamation separately.

## Providers and diagnostics

Provider aliases are normalized without importing ONNX Runtime. The explicit `auto` policy checks TensorRT, CUDA, ROCm, DirectML, OpenVINO, CoreML, XNNPACK, NNAPI, and CPU in that order. Platform-managed CoreML and mobile providers use marker extras because the compatible ONNX Runtime build supplies the provider implementation.

`runtime.diagnostics()` returns producer-neutral session records containing model paths, requested and active providers, graph inputs, and graph outputs. Consumers do not need to know whether an adapter owns one or several sessions.

## Local resources

`open_local(system="kokoro", artifacts={...}, runtime={"layout": "split-onnx-v1"})` creates the same `Installation` representation used by managed assets. Paths are resolved and verified before sessions are created. Missing components, invalid manifests, unsafe paths, unsupported layouts, and graph contract mismatches use OnnxVoice errors.

## Adapter extension boundary

A system adapter owns its model-specific ONNX input/output contract and runtime lifecycle. Adapter registration does not itself define a catalog source or catalog contents. Applications can register a `SystemAdapter` implementation with `register_adapter`; catalog tooling remains a separate integration.

## Validation layers

`verify_installation()` checks installed-file presence, size, and SHA-256. `validate_onnx()` checks that a model loads and exposes input/output metadata through ONNX Runtime. `validate_audio()` checks returned audio is numeric, finite, and non-silent. These inexpensive checks do not constitute waveform parity or release-grade quality gates.

## Package versioning

Package versions are generated with `setuptools_scm` from version-control metadata. Source distributions without SCM metadata use the configured fallback version.

## Further reading

- [Concepts](concepts.md)
- [Storage](storage.md)
- [Catalog voices](voices.md)
- [Provider handling](providers.md)
- [System adapters](systems/index.md)
