# Phonon

Phonon support is a low-level, four-graph ONNX runtime for already-tokenized integer IDs to streaming mono `float32` PCM. ONNXVoice does not tokenize or normalize text, split long text, prepare application-level requests, or apply post-processing. Those tasks belong to a separate frontend.

The adapter is registered independently of the currently published model catalog. The catalog is authoritative: discover current availability instead of assuming a model, language, voice, or quality exists. It represents one active release set, not a historical-version resolver. When present, `release_version`, `release_tags`, and source revisions are provenance metadata; inspect the selected target and quality rather than relying on a fixed version.

```bash
onnxvoice list --system phonon
onnxvoice engines list --engine phonon --format json
onnxvoice engines options --engine phonon
```

Use only target-reported `quality_profiles`; do not infer additional precisions from the source model registry. Catalog results expose the active `release_version` and per-quality release tags. After quality resolution, `metadata` includes `selected_quality`, `release_version`, `release_tag`, and `source_revision`; aggregated targets also expose `release_versions` and `release_tags` maps.

## Runtime graphs and boundary

Catalog-managed bundles use the canonical published contract: `runtime.kind` must be `phonon-streaming-onnx-v1`, the declared graph inventory must be exact, and dimensions, state, voices, config/catalog equality, and graph signatures remain validated.
The adapter validates the installed config and each graph signature, then creates provider-configured ONNX Runtime sessions lazily as inference needs them.

| Component          | Relative graph path          |
| ------------------ | ---------------------------- |
| `text_conditioner` | `onnx/text_conditioner.onnx` |
| `flow_lm_main`     | `onnx/flow_lm_main.onnx`     |
| `flow_lm_flow`     | `onnx/flow_lm_flow.onnx`     |
| `mimi_decoder`     | `onnx/mimi_decoder.onnx`     |

Each utterance gets fresh Flow key/value state and Mimi decoder state. A caller-provided seed controls request-local flow noise. The stream yields `PhononStreamChunk` values with one mono PCM frame, sample rate, frame index, and EOS information. `infer()` collects that stream into one `InferenceResult`; it does not convert text to IDs or alter audio levels.

A bundle declares its sample rate, frame size, generation defaults, voice prompts, and frame limits. ONNXVoice does not fill missing controls with hidden family-wide defaults. Check target options before inference; supply any control marked required, including `max_frames` when the bundle declares no default. The frame cap is enforced by the runtime.

Controls accepted by `stream()` and `infer()` are `voice`, `duration_delta`, `temperature`, `seed`, `eos_threshold`, `frames_after_eos`, `min_frames_before_eos`, and `max_frames`. Seed is optional. A declared voice selects a bundled projected prompt; this adapter does not support arbitrary reference-audio cloning (`voice_cloning` is false).

## Managed install and streaming

Once a target is published, use its canonical ID and one of its catalog-declared qualities:

```python
from onnxvoice import OnnxVoice

manager = OnnxVoice()
targets = manager.list("phonon")
if not targets:
    raise RuntimeError("No Phonon targets are published in the current catalog")

target = targets[0]
quality = target.metadata["quality_profiles"][0]
installation = manager.install(f"phonon:{target.id}", quality=quality)

with manager.open(installation, providers="cpu") as runtime:
    # These are model-ready IDs supplied by the frontend, not text.
    token_ids = [1, 2, 3, 4]
    for chunk in runtime.stream(token_ids, voice=installation.default_voice, seed=17):
        consume_pcm(chunk.audio, sample_rate=chunk.sample_rate)
```

The example chooses from live catalog results. `onnxvoice install phonon:<target-id> --quality <published-quality>` uses the ordinary managed, per-file artifact installer with direct HTTPS URLs, declared byte sizes, and SHA-256 checksums; GitHub Release URLs do not require Hugging Face transport or ZIP extraction. Installed files are cached, and `open()` is local-only after installation. `--offline` prevents catalog or artifact network access; cached artifacts can still be reopened.

A streaming consumer should preserve chunk order and use each chunk's `sample_rate`. `runtime.infer(token_ids, ...)` is available when a caller wants all frames in one contiguous result instead.

## Local files

`OnnxVoice.open_local()` can open explicit Phonon files without copying them to or registering them in the managed store. Supply the config, four graph components, and a declared voice prompt using the same artifact-role names:

For `open_local(system="phonon")`, a canonical config works as-is, and an exporter-native config may omit the runtime-kind marker when all other runtime fields and graph signatures are valid. A supplied conflicting `runtime.kind`, legacy `runtime_kind`, or top-level `kind` is rejected. Compatibility reads only explicit Phonon runtime fields and known aliases; unrelated nested `kind` values do not select the runtime.

The example assumes the local config declares the required generation defaults. Otherwise, supply explicit caller-selected values for each control that engine options report as required.

```python
from onnxvoice import OnnxVoice

runtime = OnnxVoice.open_local(
    system="phonon",
    files={
        "config": "/models/phonon/config.json",
        "model:text_conditioner": "/models/phonon/onnx/text_conditioner.onnx",
        "model:flow_lm_main": "/models/phonon/onnx/flow_lm_main.onnx",
        "model:flow_lm_flow": "/models/phonon/onnx/flow_lm_flow.onnx",
        "model:mimi_decoder": "/models/phonon/onnx/mimi_decoder.onnx",
        "voice_prompt:<declared-voice>": "/models/phonon/voices/<declared-voice>.bin",
    },
    providers="cpu",
)
try:
    chunks = runtime.stream(token_ids, voice="<declared-voice>", seed=17)
    for chunk in chunks:
        consume_pcm(chunk.audio, sample_rate=chunk.sample_rate)
finally:
    runtime.close()
```

Replace the voice name and paths with files declared by the local config. The prompt file contains little-endian float32 projected embeddings. If the local config leaves controls unknown, supply explicit caller-selected values for every required field before streaming. Local opening does not establish licensing rights or validate that required upstream attribution has been retained.

## Voices, qualities, and licensing

Phonon voice IDs come only from the selected published bundle's declared prompt inventory. They appear as child voices in `OnnxVoice.list_voices(system="phonon")`; a semantic reference has the form `phonon:<catalog-target>/<declared-voice-id>`. Availability follows the active catalog and selected bundle. See [catalog voice discovery](../voices.md#phonon-voices) for how this relates to semantic voice records.

The bundle producer records different licenses for model lines and distributions; inspect each selected target's catalog license metadata rather than assuming a fixed model or quality is available. Keep bundled attribution and upstream notices. CC-BY-NC-4.0 is non-commercial; installation does not grant commercial-use rights. ONNXVoice source-code licensing is separate from model and voice licensing.

## Providers and diagnostics

Install an ONNX Runtime distribution and provider supported by the selected platform; ONNXVoice manages provider selection and session options but does not install the runtime. For example, the `cpu` extra provides CPU inference support. `runtime.diagnostics()` reports the Phonon streaming layout and per-graph session diagnostics for sessions that have been created. `close()` closes those sessions; they are recreated lazily if the adapter is used again.

## Real-bundle acceptance

The managed-install, CPU inference, and offline-reopen acceptance test is opt-in and uses only published `fp32` catalog entries. It is skipped by default, so ordinary tests do not download model artifacts. Run it when release assets and `onnxruntime` are available:

```bash
ONNXVOICE_PHONON_ACCEPTANCE=1 \
ONNXVOICE_REQUIRE_PHONON_ACCEPTANCE=1 \
pytest -q tests/test_phonon_real_bundle.py
```

Set both acceptance flags as shown to make the required Apache-2.0 31M case fail rather than skip when that target, its assets, or the runtime is unavailable. The 40M case remains optional and is tested only when its `fp32` variant is published. A passing unit or fake-session test does not substitute for this real-bundle check.
