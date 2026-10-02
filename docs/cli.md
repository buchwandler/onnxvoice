# CLI reference and workflows

The CLI is organized around catalog discovery, local inventory, installation lifecycle, and cache maintenance. The global `--cache-dir` and `--offline` options come before the command.

## Top-level command map

| Command               | Purpose                                                               |
| --------------------- | --------------------------------------------------------------------- |
| `onnxvoice list`      | Merged installed and available inventory                              |
| `onnxvoice voices`    | Discover or resolve catalog voices                                    |
| `onnxvoice installed` | Local-only installed inventory                                        |
| `onnxvoice install`   | Install a catalog asset                                               |
| `onnxvoice updates`   | Compare installed assets with catalog entries                         |
| `onnxvoice update`    | Update one or all outdated installations                              |
| `onnxvoice info`      | Detailed installed-asset information, optionally with an update check |
| `onnxvoice doctor`    | Non-secret Pocket download diagnostics                                |
| `onnxvoice path`      | Print an installation path                                            |
| `onnxvoice show`      | Show an installed manifest                                            |
| `onnxvoice remove`    | Remove installation variants                                          |
| `onnxvoice verify`    | Verify installed files against checksums                              |
| `onnxvoice cache`     | Report usage or garbage-collect unreachable cache data                |
| `onnxvoice import`    | Import a model and optional config or voices file into the store      |
| `onnxvoice catalog`   | Maintainer commands to build and verify catalogs                      |

## Inventory and filters

```bash
# Installed entries, local-only
onnxvoice installed

# Installed US English voices
onnxvoice installed --kind voice --lang en-US

# Installed male US English voices
onnxvoice installed --kind voice --lang en-US --gender male

# Available male US English voices
onnxvoice list --kind voice --lang en-US --gender male --status available

# Filter a system and request machine-readable output
onnxvoice list --system piper --format json
onnxvoice installed --format tsv
```

| Flag                   | Accepted values or meaning                          |
| ---------------------- | --------------------------------------------------- |
| `--system`             | `piper`, `kokoro`, `pocket`, `supertonic`, `kitten` |
| `--kind`               | `voice`, `model`, `bundle`                          |
| `--lang`, `--language` | Language tag such as `en`, `en-US`, or `de-DE`      |
| `--gender`             | `male`, `female`, `neutral`, `unknown`              |
| `--quality`            | Quality selection                                   |
| `--distribution`       | Distribution identifier                             |
| `--status`             | `installed`, `available`, `local`                   |
| `--format`             | `table`, `plain`, `json`, `tsv`                     |

Generic language tags match compatible specific locales. An `en` filter matches English locales such as `en-US` and `en-GB`. An `en-US` filter matches a generic `en` capability, but not a conflicting specific locale such as `en-GB`. Case and hyphen normalization are supported.

Gender is source metadata and is never inferred from a voice name or ID. When the source does not provide authoritative gender, `unknown` is expected.

## Install and inspect

```bash
onnxvoice install piper:en_US-lessac-medium
onnxvoice install kokoro:v1.0 --quality fp16
onnxvoice info piper:en_US-lessac-medium
onnxvoice path piper:en_US-lessac-medium
onnxvoice show piper:en_US-lessac-medium
onnxvoice verify piper:en_US-lessac-medium
```

`install` accepts `--quality`, `--distribution`, `--refresh`, and `--force`. Catalog quality and distribution choices can produce separate managed installations.

## Check and apply updates

```bash
onnxvoice updates
onnxvoice updates --cached
onnxvoice updates --system piper
onnxvoice info piper:en_US-lessac-medium --check-updates
onnxvoice update piper:en_US-lessac-medium
onnxvoice update --all
```

Update checks normally refresh catalog data. `--cached` checks against cached catalogs only. The comparison is based on catalog and installed artifact identity and digests, not just a display version. Updating preserves the existing effective quality and distribution selection unless another selection is specified.

The comparison statuses are `current`, `update_available`, `unknown`, and `not_applicable`. `unknown` means there was not enough comparable artifact identity or digest information to determine whether the installation changed.

## Remove installations and reclaim storage

```bash
onnxvoice remove piper:en_US-lessac-medium --dry-run
onnxvoice remove piper:en_US-lessac-medium
onnxvoice remove piper:en_US-lessac-medium --gc
onnxvoice remove kokoro:v1.0 --all-variants --yes
```

Removing an installation does not necessarily delete its shared content blobs or Pocket voice-state records. Run garbage collection to reclaim unreachable cache data. Use `--dry-run` to inspect a removal or collection before applying it.

```bash
onnxvoice cache info
onnxvoice cache info --format json
onnxvoice cache gc --dry-run
onnxvoice cache gc
```

Cache reports distinguish installation logical bytes, blob bytes, unique file bytes, orphan blobs, auxiliary bytes, Pocket state usage, and orphan auxiliary data. Pocket state is retained while an installed matching bundle variant still references it.

## Catalog voices

```bash
onnxvoice voices list
onnxvoice voices list --lang en-US
onnxvoice voices list --system kokoro
onnxvoice voices show piper:en_US-lessac-medium
onnxvoice voices show kokoro:v1.0/af_heart
onnxvoice voices list --system supertonic --lang de
onnxvoice voices show supertonic:supertonic-3/F1
onnxvoice voices list --system kitten --lang en-US
onnxvoice voices show kitten:nano-0.8-int8/Bella
```

`voices list` accepts language and system filters, `--refresh`, and `--format`. Language filters match the voice's supported synthesis-language capabilities, not only its primary display locale. Each result identifies a normalized catalog voice. `show` resolves a semantic voice reference. Supertonic and Kitten child voices use refs such as `supertonic:supertonic-3/F1` and `kitten:nano-0.8-int8/Bella`. See [KittenTTS](systems/kitten.md) for the runtime boundary. See [Catalog voices](voices.md) for reference forms and record fields.

## Pocket diagnostics and offline use

```bash
onnxvoice doctor --system pocket
onnxvoice --offline doctor --system pocket --format json
```

The diagnostic does not reveal tokens or test access to a gated repository over the network. See [Pocket downloads](pocket-downloads.md) for authentication and offline-cache details.

## Import local files

The CLI `import` command accepts one model file and optional config and voices files:

```bash
onnxvoice import --system piper --id my-voice --model voice.onnx --config voice.onnx.json
```

This command does not describe every multi-component runtime layout. Use the Python local-file API for explicit multi-component models. See [Local models](local-models.md).

## Catalog maintenance

Catalog tooling is intended for maintainers rather than ordinary installation workflows:

```bash
onnxvoice catalog piper build --output catalog/voices.json --source-output catalog/source.json
onnxvoice catalog piper verify --catalog catalog/voices.json --source catalog/source.json
onnxvoice catalog pocket build --output catalog/bundles.json
onnxvoice catalog pocket verify --catalog catalog/bundles.json
onnxvoice catalog pocket prompts build \
  --repository kyutai/tts-voices \
  --revision main \
  --output catalog/voice-prompts.json \
  --source-output catalog/voice-prompts-source.json
onnxvoice catalog pocket prompts verify --catalog catalog/voice-prompts.json --source catalog/voice-prompts-source.json
onnxvoice catalog supertonic build --output catalog/supertonic.json
onnxvoice catalog supertonic verify --catalog catalog/supertonic.json
onnxvoice catalog kitten build \
  --seed-catalog catalog/models.json \
  --output catalog/models.json \
  --source-output catalog/source.json \
  --revision main
onnxvoice catalog kitten verify --catalog catalog/models.json --source catalog/source.json
```

Piper, Pocket, and Supertonic builders accept repository and revision options. The Kitten builder uses the normalized catalog as a seed manifest and refreshes each listed Hugging Face repository; it reads upstream `config.json` and artifact metadata. Build requires network access, while verification is offline. See [KittenTTS](systems/kitten.md) and `onnxvoice catalog <system> <action> --help` for details.

The Pocket prompts builder resolves the requested revision to an exact commit SHA, enumerates that pinned tree, selects `.wav` prompts, and applies checked-in license rules. A prompt without a matching license rule fails the build loudly, and every prompt record pins revision, size, SHA-256, and license provenance. See [Pocket downloads](pocket-downloads.md).
