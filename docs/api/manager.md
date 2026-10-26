# Manager

`OnnxVoice` is the high-level facade for catalog operations, managed installations, inventory, updates, and local runtime creation.

Its `cache_usage()`, `plan_cache_gc()`, and `gc_cache()` methods expose typed, local-only cache accounting and collection. `plan_remove()` resolves all requested installed variants before computing immediate and post-GC reclamation; it does not mutate the cache. These operations do not load catalog data or make network requests.

```{automodule} onnxvoice.manager
:members: OnnxVoice
:show-inheritance:
```
