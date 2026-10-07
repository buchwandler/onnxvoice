# Systems

Each system adapter owns the model-specific ONNX graph contract. It does not provide text normalization, G2P/phonemization, sentence planning, or application-level voice policy.

| System     | Built-in adapter | Built-in catalog | Voice data in normalized catalog items | Typical runtime inputs                                  |
| ---------- | ---------------- | ---------------- | -------------------------------------- | ------------------------------------------------------- |
| Piper      | Yes              | Yes              | Voice item IDs                         | Token IDs and optional numeric speaker ID               |
| Kokoro     | Yes              | Yes              | Child voice IDs on model items         | Token IDs and caller-provided model-ready style         |
| Pocket     | Yes              | Yes              | IDs in each item's `voices` field      | Token IDs and prepared voice state                      |
| Supertonic | Yes              | Yes              | Child voice IDs on model items         | Token IDs, text mask, and caller-provided style tensors |
| Kitten     | Yes              | Yes              | Child voice IDs on model items         | Model-ready token IDs, style tensor, and speed          |
| Inflect    | Yes              | Yes              | One fixed `default` voice              | Model-ready token IDs, speed, variation, and seed       |

`OnnxVoice` discovers voices from the normalized catalog data shown above. A catalog item whose kind is `voice` contributes its own ID; items with child voices contribute the IDs already present in `CatalogItem.voices`. The catalog determines which voices are available.
Adapter registration and catalog availability are distinct. A built-in adapter does not guarantee that external catalog data is available.

```{toctree}
:maxdepth: 1

piper
kokoro
pocket
supertonic
kitten
inflect
```
