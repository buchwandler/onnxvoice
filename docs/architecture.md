# OnnxVoice architecture

OnnxVoice executes installed voice-model runtimes and owns stable catalog voice identity selectors. Producer packages prepare token IDs, model-ready style rows, speed, and other producer-owned policy. Text normalization, language detection, G2P, phoneme segmentation, producer semantic voice aliases, sentence handling, trace objects, and final audio composition remain outside this package.

## Stable catalog voice identity

Short selectors such as `de-ko-1` and `de-pi-1` are persisted, append-only aliases for complete canonical identities. Permanent engine codes are `ko` for Kokoro, `pi` for Piper, and `po` for Pocket. Selectors resolve to a system, backing asset/model, and logical voice ID while leaving `system:id` asset references unchanged. Catalog projection may report an unassigned voice, but runtime discovery never invents a numeric slot.

This identity alias is distinct from producer semantic voice aliases and style policy. In particular, resolving a Kokoro selector does not select or construct a style tensor; the producer remains responsible for that operation.

Pocket selector identity is bundle-scoped as `(pocket, bundle_id, voice_id)` and is allocated only when canonical catalog metadata includes an explicit matching `voice_states` record. `predefined_voice_names` alone is not sufficient to construct a stable identity. The current canonical Pocket catalog has no explicit `voice_states`, so its predefined voices remain unassigned. Selector capability is independent of runtime adapter registration.

## Descriptive language and voice metadata

Descriptive metadata does not define selector identity. `locale` records the most specific source-backed language tag, while `language` is its lowercase base language. Selector namespaces remain permanent: English Piper locales use `en`, while Kokoro `en-US` and `en-GB` use `en_us` and `en_gb`. Existing selector mappings are not rewritten when descriptive tags are normalized.

Language compatibility is deliberately symmetric for generic capabilities: a generic `en` capability satisfies an `en-US` request, and a generic `en` request matches specific English locales. Conflicting specific locales such as `en-US` and `en-GB` do not match.

Language labels prefer authoritative human-readable source names, then locale/base tags, and never use a bare region code such as `US`.
`VoiceRecord.metadata` is the normalized projection for language, locale, language label, and gender; the existing `languages` and `gender` fields remain available. Gender is taken only from explicit authoritative metadata, with `unknown` as the fallback. Pocket `voice_details` describes declared voices independently of asset identity. Only explicit `voice_states` can qualify a Pocket voice for permanent selector assignment.

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

A system adapter owns its model-specific ONNX input/output contract and runtime lifecycle. The adapter registry does not itself define a catalog source, parser, or stable selector namespace. Applications can register a `SystemAdapter` implementation with `register_adapter`; catalog tooling and selector support remain separate integrations.

## Validation layers

`verify_installation()` checks installed-file presence, size, and SHA-256. `validate_onnx()` checks that a model loads and exposes input/output metadata through ONNX Runtime. `validate_audio()` checks returned audio is numeric, finite, and non-silent. These inexpensive checks do not constitute waveform parity or release-grade quality gates.

## Package versioning

Package versions are generated with `setuptools_scm` from version-control metadata. Source distributions without SCM metadata use the configured fallback version.

## Further reading

- [Concepts](concepts.md)
- [Storage](storage.md)
- [Stable voice selectors](voices.md)
- [Provider handling](providers.md)
- [System adapters](systems/index.md)
