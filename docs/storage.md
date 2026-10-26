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

`blobs/sha256` is ONNXVoice's verified content-addressed artifact store. The blob is the shared, checksum-verified copy of an artifact and can be reused by multiple installations. Installation views normally hard-link to blobs; when hard links are unavailable, ONNXVoice copies the verified file instead. Do not delete `blobs/` manually: it may contain objects still referenced by installed assets, and direct deletion bypasses reachability checks and locks.

An installation manifest records the selected item, artifact metadata, and verification information. It is separate from the shared blob objects.

## Understand cache usage

## Understand cache usage

```bash
onnxvoice cache info
onnxvoice cache info --details
onnxvoice cache info --format json
```

The default report highlights filesystem free space, whole-cache disk allocation, and bytes reclaimable by safe GC. The detailed report provides category and top-level path totals, installation hard-link/copy counts, referenced and orphan blob data, and scan diagnostics. JSON always contains the full structured report and preserves existing flat keys; byte values remain integer counts.

Metrics have different meanings:

- **Logical installation bytes** count the represented artifact payloads.
- **Apparent bytes** sum file sizes; hard-linked names can appear more than once.
- **Unique apparent bytes** count each file inode once.
- **Allocated bytes** sum filesystem blocks once per inode where the platform reports them.
- **Reclaimable now** includes only current safe GC candidates, not unknown or unmanaged files.

Installation files are usually hard links to shared blobs. Copy-fallback installations consume additional space and may free those copied blocks immediately when removed. Consequently, adding blob bytes and installation logical size is not a valid estimate of disk usage or removal impact.

The scan includes every regular file below the cache root, does not follow symlinks, and reports unknown/legacy paths. Unknown cache content is visible but is never considered a GC candidate. The filesystem capacity fields describe the volume containing the cache, which may also contain unrelated files.

## Remove and collect garbage

Removing an installation deletes its managed installation view. Garbage collection removes shared content only after no remaining installation references it. Pocket voice-state records likewise remain while referenced by any installed matching bundle variant.

Preview safe GC or removal before deleting data:

```bash
onnxvoice cache gc --dry-run
onnxvoice remove REF --dry-run --gc
```

Removal preview distinguishes the logical installation size, immediate unique bytes, newly orphaned blobs/states, and total expected reclaimability after optional GC. For multiple refs, all selections are validated and planned before mutation. Actual `remove REF --gc` reports immediate reclaimed bytes and post-removal GC bytes separately from logical size.

Run standard garbage collection with:

```bash
onnxvoice cache gc
```

GC uses reachability rather than file age and revalidates candidates under its lock before deletion. It removes only unreachable content-addressed blobs and recognized unreferenced Pocket state. A stale preview is never trusted for destructive execution.

The retired `pocket-voice-prompts/` directory is reported as legacy cache data but is not removed by default. Preview or explicitly opt in to its cleanup with `--include-legacy`:

```bash
onnxvoice cache gc --include-legacy --dry-run
onnxvoice cache gc --include-legacy
```

This opt-in is limited to the recognized retired path; arbitrary unknown directories remain report-only.

## Offline cache behavior

Offline mode can read installed manifests, cached catalogs, blobs, and cached Pocket states. It cannot fill a cache miss. An installation can still be opened offline after all its required data is present locally.
