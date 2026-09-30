# Catalog

`CatalogClient` loads a system catalog from the configured source or local cache and normalizes it into `CatalogItem` objects. Loading can perform network I/O when data is missing, stale, or explicitly refreshed. Offline mode uses cached data only.

```{automodule} onnxvoice.catalog
:members: CatalogClient, DEFAULT_SOURCES, parse_ref, filter_items
:show-inheritance:
```
