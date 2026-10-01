# KittenTTS

OnnxVoice provides a built-in adapter and catalog for KittenTTS v0.8 ONNX bundles. It manages the pinned model and `voices.npz`, validates the ONNX tensor contract, and runs inference. It does not process text or choose a voice style.

## Available models and voices

The default catalog currently includes `nano-0.8-int8`, `nano-0.8-fp32`, `micro-0.8`, and `mini-0.8`. Each item has a `model` ONNX artifact and a `voices` NPZ artifact. Catalog voice aliases include Bella, Jasper, Luna, Bruno, Rosie, Hugo, Kiki, and Leo. Alias matching is case-sensitive. Use catalog discovery for the current set:

```bash
onnxvoice list --system kitten
onnxvoice voices list --system kitten --lang en-US
onnxvoice voices show kitten:nano-0.8-int8/Bella
```

A voice reference identifies the catalog voice, not an ONNX style tensor. The backing asset reference, such as `kitten:nano-0.8-int8`, is the model to install or open. The current catalog has no authoritative per-voice gender metadata, so discovered Kitten voices report `unknown` rather than inferring gender from names or style IDs.

## Managed model use

```python
from onnxvoice import OnnxVoice

voice = OnnxVoice()
installation = voice.install("kitten:nano-0.8-int8")
runtime = voice.open(installation, provider="cpu")
try:
    result = runtime.infer(token_ids, style=style, speed=1.0)
finally:
    runtime.close()
```

`token_ids` are already model-ready integer IDs. `style` is the model-ready style tensor selected by the caller, and `speed` must be a finite positive number. OnnxVoice does not add framing tokens, extract or select a style from the voices archive, or apply speed-prior policy. The adapter checks the model input names and tensor shapes against the graph, removes exactly 5,000 trailing output samples, and returns finite mono float32 audio at 24 kHz.

KittenSynth owns text normalization, G2P, voice alias/style selection, and synthesis policy. Its frontend and style selection prepare the values passed to this adapter. OnnxVoice owns catalog, artifact, provider, session, and tensor mechanics.

## Open local files

Use `open_local()` when the files should remain outside the managed cache:

```python
from onnxvoice import OnnxVoice

runtime = OnnxVoice.open_local(
    system="kitten",
    model="kitten_tts_nano_v0_8.onnx",
    voices="voices.npz",
    sample_rate=24000,
    provider="cpu",
)
try:
    result = runtime.infer(token_ids, style=style, speed=1.0)
finally:
    runtime.close()
```

Local opening does not access a catalog, copy files, or register them in the shared store. The `voices` artifact is retained as installation metadata; style extraction and selection remain with the caller.

## Catalog maintenance

The catalog builder takes the checked-in normalized catalog as its seed manifest. It resolves each listed Hugging Face repository to an immutable revision, reads that revision's `config.json` and file metadata, then writes the updated model catalog and deterministic source record. LFS size and SHA-256 metadata are used when available; otherwise the builder streams the artifact to calculate its digest. Build uses the network. Verify is offline.

```bash
onnxvoice catalog kitten build \
  --seed-catalog catalog/models.json \
  --output catalog/models.json \
  --source-output catalog/source.json \
  --revision main
onnxvoice catalog kitten verify \
  --catalog catalog/models.json \
  --source catalog/source.json
```

The seed supplies normalized model IDs, aliases, language, quality, and upstream repositories. Upstream config supplies model name/version, runtime profile, model filename, voice aliases, and speed priors. Review changes to the seed when adding a new model or changing its normalized identity. See [Local models](../local-models.md) for the local-file boundary.
