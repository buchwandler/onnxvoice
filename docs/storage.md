# Storage and cache management

The shared cache is managed by `AssetStore` and normally resides in the platform cache location selected by `platformdirs`. Override it with the CLI's global `--cache-dir` option or `OnnxVoice(cache_dir=...)`.

By default, `platformdirs` selects an application cache directory such as `~/.cache/onnxvoice` on Linux. That directory itself is the cache root. A custom `cache_dir` is also used directly, without adding another `onnxvoice` directory.

A typical cache-root layout is:

```text
<cache-root>/
├── blobs/sha256/
├── catalogs/
├── installs/
├── locks/
└── pocket-voice-states/
```

## Content-addressed blobs and installations

Verified artifact bytes are stored by digest and can be reused by multiple installations. Installations link to the blobs where supported. If the filesystem cannot hard-link, the store can copy the verified file into the installation, which can increase physical disk usage.

An installation manifest records the selected item, artifact metadata, and verification information. It is separate from the shared blob objects.

## Understand cache usage

```bash
onnxvoice cache info
onnxvoice cache info --format json
```

The report distinguishes:

- installation logical bytes, which count files as represented by installations;
- blob apparent bytes, which count content-addressed objects;
- unique file bytes, which account for files not shared by hard links;
- orphan blob count and bytes;
- catalog bytes;
- auxiliary bytes, including Pocket voice-state data;
- Pocket voice-state count and bytes;
- orphan auxiliary count and bytes.

These values answer different questions. Logical installation size is not necessarily the amount of physical disk consumed.

## Remove and collect garbage

Removing an installation deletes its managed installation representation, but shared blobs may still be reachable from other installations. Pocket voice states are tracked separately and remain cached while any installed matching bundle variant references them.

Preview collection before deleting data:

```bash
onnxvoice cache gc --dry-run
```

Then collect unreachable blobs and auxiliary records:

```bash
onnxvoice cache gc
```

Removal and collection are deliberately separate operations. `onnxvoice remove REF --gc` combines them for convenience. Pocket state reachability is based on installed bundle variants.

## Offline cache behavior

Offline mode can read installed manifests, cached catalogs, blobs, and cached Pocket states. It cannot fill a cache miss. An installation can still be opened offline after all its required data is present locally.
