# OnnxVoice architecture

OnnxVoice executes installed voice-model runtimes. Producer packages prepare token IDs, model-ready style rows, speed, and other producer-owned policy. Text normalization, language detection, G2P, phoneme segmentation, voice aliases, sentence handling, trace objects, and final audio composition remain outside this package.

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

## Providers and diagnostics

Provider aliases are normalized without importing ONNX Runtime. The explicit `auto` policy checks TensorRT, CUDA, ROCm, DirectML, OpenVINO, CoreML, XNNPACK, NNAPI, and CPU in that order. Platform-managed CoreML and mobile providers use marker extras because the compatible ONNX Runtime build supplies the provider implementation.

`runtime.diagnostics()` returns producer-neutral session records containing model paths, requested and active providers, graph inputs, and graph outputs. Consumers do not need to know whether an adapter owns one or several sessions.

## Local resources

`open_local(system="kokoro", artifacts={...}, runtime={"layout": "split-onnx-v1"})` creates the same `Installation` representation used by managed assets. Paths are resolved and verified before sessions are created. Missing components, invalid manifests, unsafe paths, unsupported layouts, and graph contract mismatches use OnnxVoice errors.
