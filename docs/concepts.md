# Concepts

## System

A system identifies an adapter and catalog format. The built-in systems are Piper, Kokoro, Pocket, Supertonic, and Kitten. Normalized catalog items determine which voices are exposed for each system.

## Catalog item and artifact

A catalog item describes an installable voice, model, or bundle. Its artifacts are the files and metadata needed by the runtime. Quality and distribution selections can affect which artifacts are installed and how an installation is identified.

For Kokoro, selecting a quality chooses a model variant rather than installing every quality variant. Pocket quality names select a complete bundle profile.

## Reference and installation

A reference is the canonical `system:id` name used to identify a catalog item, such as `piper:en_US-lessac-medium`, `kokoro:v1.0`, `pocket:english_2026-04`, and `kitten:nano-0.8-int8`. An installation is the verified managed local representation of a selected catalog item or imported model.

An installation manifest records the artifacts and selected metadata. An installation may share immutable content-addressed blobs with other installations.

## Voice record and semantic reference

A voice record identifies a normalized catalog voice with `(system, asset_id, voice_id)`. Its semantic reference has the form `<system>:<asset-id>[/<voice-id>]`. For example, `piper:en_US-lessac-medium` identifies an asset that is itself a voice, while `kokoro:v1.0/af_heart`, `supertonic:supertonic-3/F1`, and `kitten:nano-0.8-int8/Bella` identify child voices in model or bundle catalog items. See [Catalog voices](voices.md).

## Managed and local runtime opening

| Operation      | Catalog or network access                            | Shared store                   | Copies or registers files |
| -------------- | ---------------------------------------------------- | ------------------------------ | ------------------------- |
| `install()`    | Yes, when catalog or assets are missing or refreshed | Yes                            | Yes                       |
| `open()`       | No acquisition                                       | Reads an existing installation | No                        |
| `open_local()` | No                                                   | No                             | No                        |

Use `install()` to acquire catalog assets, `open()` to use an existing managed installation, and `open_local()` for explicit local files that should remain outside the shared store.

## Offline data

The cache can contain catalog data, installed assets, content-addressed blobs, and Pocket voice-state records. These are distinct resources. Offline mode uses cached data only and reports a miss rather than making a network request.

## Catalog sources

The default catalog client has sources for Piper, Kokoro, Pocket, Supertonic, and Kitten. Catalogs and model artifacts are external data and are not bundled with the package. Override a system source with `ONNXVOICE_PIPER_CATALOG`, `ONNXVOICE_KOKORO_CATALOG`, `ONNXVOICE_POCKET_CATALOG`, `ONNXVOICE_SUPERTONIC_CATALOG`, or `ONNXVOICE_KITTEN_CATALOG`. Each value may be a local path or an HTTP(S) URL. Python callers can use `catalog_sources` when constructing `OnnxVoice`.
