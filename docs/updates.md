# Checking and applying updates

Update checks compare installed artifacts with the corresponding catalog entry. The comparison uses artifact roles and available digest identity rather than relying only on a human-readable version string.

## CLI

```bash
# Refresh catalogs and check all installations
onnxvoice updates

# Use cached catalogs without fetching
onnxvoice updates --cached

# Limit the check to a system
onnxvoice updates --system piper

# Update one item or all outdated items
onnxvoice update piper:en_US-lessac-medium
onnxvoice update --all
```

`onnxvoice info REF --check-updates` checks one installed item while displaying its details. `updates` normally refreshes catalogs; `--cached` prevents refresh. Offline mode also prevents network access.

When updating an installation, the selected quality and distribution are preserved unless new selections are supplied. Local or external imports may not have a corresponding catalog entry and may not have a meaningful remote update state.

## Comparison states

| Status             | Meaning                                                                                |
| ------------------ | -------------------------------------------------------------------------------------- |
| `current`          | Comparable installed and catalog artifacts match.                                      |
| `update_available` | One or more comparable artifact roles changed.                                         |
| `unknown`          | The catalog did not provide enough artifact identity or digest information to compare. |
| `not_applicable`   | The installation does not have an applicable catalog update comparison.                |

These are comparison results, not a promise that every remote source publishes a semantic release version.
