# Sopro

Sopro is a manifest-backed, zero-shot voice-cloning system. The pinned `sopro:v2-turbo` catalog bundle declares a 24 kHz output rate and English (`en`), European Portuguese (`pt`), French (`fr`), and German (`de`) language tags. It does not provide predefined named voices; reference audio supplies the reusable voice state.

## Current implementation status

The adapter supports catalog-managed/local opening, graph inspection, token preparation, reusable reference preparation, one-segment offline synthesis, and stateful sequential `SoproRun` segment generation with automatic semantic carry. `SoproRun.stream_segment()` performs synchronous incremental inference on installed split acoustic ODE graphs, keeps acoustic/vocoder/ISTFT state, and yields raw mono float32 chunks; references too short for the streaming prompt safely fall back to one offline chunk. Bounded prompt/vocoder caches, synchronous `prepare_streaming()`, and final-chunk stream timings are available. The gated `wasm-uint8`/CPU real-bundle integration verifies incremental output, exact generated sample count, streaming graph use, and recovery after an early consumer stop.

**Native Python `infer()` is enabled only for the `wasm-uint8` profile with `CPUExecutionProvider`.** This is the only profile/provider pair exercised by the complete native parity gate; `webgpu-fp16`, other profiles, and other requested providers remain unsupported for Python inference. `infer()` accepts already-prepared model token IDs plus a reusable `SoproVoiceState` and returns raw mono float32 PCM. It does not split sentences, apply application-level loudness policy, or encode WAV.

`runtime.generation_defaults` exposes a frozen `SoproGenerationDefaults` snapshot; `fixed_steps`, `max_segment_chars`, `token_samples`, and `hop_ratio` provide validated manifest values without callers reading raw dictionaries. Sampling accepts `top_p` in the inclusive range `[0, 1]` (`0` retains the highest-probability candidate); `temperature=0` remains greedy, and `top_k=0` disables top-k filtering. Reference preparation accepts a finite positive `seconds` override; omitting it uses the manifest reference duration.

## Prepare text and reference voice

```python
import numpy as np
from onnxvoice import OnnxVoice

voice = OnnxVoice()
installation = voice.install("sopro:v2-turbo", quality="wasm-uint8")
runtime = voice.open(installation, provider="cpu")
try:
    text_input = runtime.prepare_text("olá, mundo!", language="pt")
    reference = runtime.prepare_voice(
        np.asarray(reference_audio, dtype=np.float32),
        sample_rate=24000,
    )
    result = runtime.infer(
        text_input.token_ids,
        voice_state=reference,
        language=text_input.language,
        temperature=0.7,
        top_p=0.95,
        top_k=50,
        max_seconds=10.0,
        seed=42,
    )
    print(result.audio.shape, result.sample_rate)
finally:
    runtime.close()
```

For raw synchronous streaming, create a run from the prepared voice and iterate one text segment:

```python
runtime.prepare_streaming(reference, seed=42)
run = runtime.begin_run(reference, language="pt", seed=42)
try:
    for chunk in run.stream_segment("olá, mundo!"):
        consume(chunk.audio, sample_rate=chunk.sample_rate)
finally:
    run.close()
```

Each chunk is contiguous mono float32 model PCM. The final chunk sets `is_final=True`; the stream enforces an exact generated-token sample count. Short references use a one-chunk offline fallback.

`prepare_streaming()` synchronously warms the acoustic and vocoder prompt states; repeated calls with the same voice/seed are idempotent. The adapter keeps a bounded one-entry LRU by default. Set `runtime.stream_cache_capacity = 0` to disable and clear the caches. Final chunk metadata includes preparation/stream timings, cache-hit flags, chunk count, and generated sample/token counts.

`quality` selects the Sopro bundle profile; it is not an ONNX Runtime execution provider. See [local models](../local-models.md) for opening explicit local bundles without installing them into the shared store.

`prepare_text()` returns a `SoproTextInput` containing normalized text, exact SentencePiece token IDs (including model BOS/EOS IDs), and the effective language. The supported values are `en`, `pt`, `fr`, `de`, and `auto`; `auto` follows the web runtime's untagged behavior. ONNXVoice does not split text into application-level sentences.

`prepare_voice()` accepts a finite, mono floating-point waveform and its source sample rate. It applies the official resampling, pause crop, reference-level normalization, three mel frontends, and the reference graph. The returned immutable `SoproVoiceState` contains `cond_vec`, semantic reference tokens, normalized reference mel, sample rate, and reference level; reuse it across one-segment inference requests.

## Parity and support boundaries

The deterministic fixtures under `tests/fixtures/sopro/` pin tokenization, prepared reference tensors, semantic generation, acoustic intermediates, vocoder features, and final PCM. The text fixtures cover all four declared languages and the official `auto` mode.

The optional native reference-graph check runs when `ONNXVOICE_SOPRO_REFERENCE_GRAPH` points to the pinned `reference.onnx` beside its `weights-reference.bin` shard:

```bash
ONNXVOICE_SOPRO_REFERENCE_GRAPH=/path/to/common/reference.onnx \
  pytest -q tests/test_sopro_local_open.py::test_real_reference_graph_matches_pinned_onnx_outputs
```

The full pinned offline parity check uses the native ONNX Runtime and the pinned `wasm-uint8` graphs/shards required by the inference pipeline. Set `ONNXVOICE_SOPRO_E2E_ROOT` to the cache root containing `manifest.json` and `model/`:

```bash
ONNXVOICE_SOPRO_E2E_ROOT=/path/to/pinned-cache \
  pytest -q tests/test_sopro_local_open.py::test_real_native_offline_pipeline_matches_pinned_intermediate_fixture
```

The managed-install/offline-reopen gate also exercises reference preparation, one-segment inference, session diagnostics, idempotent close, and a second offline reopen/inference cycle:

```bash
ONNXVOICE_RUN_REAL_SOPRO_TESTS=1 \
ONNXVOICE_SOPRO_E2E_ROOT=/path/to/pinned-cache \
  pytest -q tests/test_sopro_real_bundle.py
```

The offline parity check validates pinned artifact hashes and acoustic/vocoder intermediates as well as final audio. The managed gate verifies installation integrity, offline alias-based reopening, reference preparation, inference output, provider diagnostics, and close/reopen behavior. Both passed with the tested Android CPU runtime; no other native profile is claimed. ONNXVoice leaves sentence planning, loudness calibration, and WAV encoding to a higher-level package such as SoproSynth.
