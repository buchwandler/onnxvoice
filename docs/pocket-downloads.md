# Pocket downloads and Hugging Face access

Pocket model and voice-state downloads use the optional Hugging Face integration. Local bundles and assets already present in the OnnxVoice cache do not require `huggingface_hub` for opening or inference.

## Installation

For direct OnnxVoice use, install the Pocket extra and an ONNX Runtime provider:

```bash
python -m pip install "onnxvoice[pocket,cpu]"
```

PocketSynth's CPU and GPU extras install the required OnnxVoice Pocket support:

```bash
python -m pip install "pocketsynth[cpu]"
# or
python -m pip install "pocketsynth[gpu]"
```

## Public model files and gated voice states

The canonical Pocket ONNX bundle repository is public, so downloading its pinned files does not require Hugging Face credentials. Predefined voice-state files are separate assets and may be gated. For a gated voice, first accept or request access on the upstream repository page, then authenticate the account used by the application:

```bash
hf auth login
hf auth whoami
```

Alternatively, provide `HF_TOKEN` through the application's normal secret-management environment. OnnxVoice does not start an interactive login, save credentials, or include tokens in cache records, diagnostics, or errors. Do not print or paste token values while debugging.

## Diagnostics

The Pocket diagnostic reports whether the optional Hub package and credentials appear configured, whether offline mode is active, and whether implicit tokens are disabled. It does not make a network request and cannot confirm access to a gated repository:

```bash
onnxvoice doctor --system pocket
onnxvoice --offline doctor --system pocket --format json
```

For a gated-access failure, verify account access on the repository page and confirm the active Hugging Face account with `hf auth whoami`. If the application runs in a service, container, or a different user environment, verify that it sees the intended `HF_HOME`, `HF_TOKEN_PATH`, or `HF_TOKEN` configuration without displaying token contents.

## Reference-voice prompts

Managed reference-voice prompts, for example `kyutai-tts-voices:alba-mackenna/casual`, are public upstream WAV files from `kyutai/tts-voices`. The prompt catalog pins each prompt to an exact commit revision and records its size, SHA-256, dataset, variant, and license provenance. The audio stays on the Hugging Face repository and is cached lazily under `pocket-voice-prompts` in the OnnxVoice cache root.

Fetches verify size and SHA-256 before an atomic cache publication, and both values are part of the cache identity, so a changed pin cannot reuse an older prompt. Cached prompts work offline, and an offline miss reports an offline error instead of touching the network. Set `ONNXVOICE_POCKET_VOICE_PROMPTS_CATALOG` to override the prompt catalog source with a local path or an HTTP(S) URL.

Reference prompts never enter Pocket `voice_states`. Those records are predefined `.safetensors` states, and a catalog record with `"format": "wav"` is rejected at parse time, before any runtime loading.

## Offline and local use

An online managed run downloads the model bundle and, when selected, its predefined voice state into OnnxVoice-managed storage. Once all required assets are cached, repeat the operation in offline mode:

```bash
pocketsynth synthesize --bundle english_2026-04 --voice alba --offline \
  --output hello.wav "Hello from Pocket."
```

An offline cache miss is reported as an offline error. Explicit local bundles and previously installed assets remain usable without the Hub client. Canonical model artifacts are checked against their catalog size and SHA-256 metadata. For explicit Pocket `voice_states`, predefined state files are checked against the declared size and SHA-256 before atomic cache publication; both values are included in the cache identity. A changed integrity pin cannot reuse an older state.

Legacy Pocket state files stay in the existing `pocket-voice-states` layout. When a catalog supplies an explicit pin, a legacy entry is verified from its cache record and reused offline only if its actual size and SHA-256 match that pin. Hugging Face credentials are never part of the cache key or metadata.

`onnxvoice cache info` reports `auxiliary_bytes`, `pocket_voice_state_count`, `pocket_voice_state_bytes`, `orphan_auxiliary_count`, and `orphan_auxiliary_bytes`. `onnxvoice cache gc --dry-run` shows unreachable state records without deleting them. Actual GC preserves states referenced by any installed matching Pocket bundle variant and removes them after the last such variant is removed.
