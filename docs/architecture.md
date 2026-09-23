# OnnxVoice architecture

OnnxVoice executes installed voice-model runtimes and owns stable catalog voice identity selectors. Producer packages prepare token IDs, model-ready style rows, speed, and other producer-owned policy. Text normalization, language detection, G2P, phoneme segmentation, producer semantic voice aliases, sentence handling, trace objects, and final audio composition remain outside this package.

## Stable catalog voice identity

Short selectors such as `de-ko-1` and `de-pi-1` are persisted, append-only aliases for complete canonical identities. Permanent engine codes are `ko` for Kokoro, `pi` for Piper, and `po` for Pocket. Selectors resolve to a system, backing asset/model, and logical voice ID while leaving `system:id` asset references unchanged. Catalog projection may report an unassigned voice, but runtime discovery never invents a numeric slot.

This identity alias is distinct from producer semantic voice aliases and style policy. In particular, resolving a Kokoro selector does not select or construct a style tensor; the producer remains responsible for that operation.

Pocket selector identity is bundle-scoped as `(pocket, bundle_id, voice_id)` and is allocated only when canonical catalog metadata includes an explicit matching `voice_states` record. `predefined_voice_names` alone is not sufficient to construct a stable identity. The current canonical Pocket catalog has no explicit `voice_states`, so its predefined voices remain unassigned. Selector capability is independent of runtime adapter registration.

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
