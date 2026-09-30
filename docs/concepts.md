# Concepts

## System

A system identifies an adapter and catalog format. The built-in systems are Piper, Kokoro, Pocket, and Supertonic. Adapter support, catalog support, and stable voice-selector support are separate capabilities.

## Catalog item and artifact

A catalog item describes an installable voice, model, or bundle. Its artifacts are the files and metadata needed by the runtime. Quality and distribution selections can affect which artifacts are installed and how an installation is identified.

For Kokoro, selecting a quality chooses a model variant rather than installing every quality variant. Pocket quality names select a complete bundle profile.

## Reference and installation

A reference is the canonical `system:id` name used to identify a catalog item, for example `piper:en_US-lessac-medium`, `kokoro:v1.0`, or `pocket:english_2026-04`. An installation is the verified managed local representation of a selected catalog item or imported model.

An installation manifest records the artifacts and selected metadata. An installation may share immutable content-addressed blobs with other installations.

## Voice record and stable selector

A voice record combines catalog voice information with a stable selector assignment when one exists. A selector such as `de-ko-1` identifies a logical catalog voice. It is not a model reference and does not select a model-ready style tensor. See [Voice selectors](voices.md).

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

The default catalog client has sources for Piper, Kokoro, Pocket, and Supertonic. Catalogs and model artifacts are external data and are not bundled with the package. Override a system source with `ONNXVOICE_PIPER_CATALOG`, `ONNXVOICE_KOKORO_CATALOG`, `ONNXVOICE_POCKET_CATALOG`, or `ONNXVOICE_SUPERTONIC_CATALOG`. Each value may be a local path or an HTTP(S) URL. Python callers can use `catalog_sources` when constructing `OnnxVoice`.
