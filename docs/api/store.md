# Asset store

`AssetStore` keeps verified installations and immutable content-addressed blobs. Removing an installation and collecting unreachable blob or Pocket voice-state data are separate operations.

`usage()` returns typed whole-root accounting, including filesystem capacity, inode-deduplicated and allocated bytes, category totals, install hard-link/copy metrics, orphans, and unmanaged paths. `plan_gc()` previews categorized, conservative reachability-based candidates. `gc(plan=...)` revalidates under the GC lock before deleting and accepts `include_legacy=True` only for the recognized retired path. `remove()` reports immediate installation-view reclamation separately from logical size; `plan_remove()` estimates a multi-installation removal and optional post-removal GC without mutation. Unknown paths are report-only.

```{automodule} onnxvoice.store
:members: AssetStore
:show-inheritance:
```
