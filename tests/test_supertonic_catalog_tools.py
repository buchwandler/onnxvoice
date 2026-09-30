from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from onnxvoice.catalog_tools import supertonic

REPOSITORY = "supertone-oss-archive/supertonic-3"
REVISION = supertonic.CURRENT_SNAPSHOT
MODEL_PATHS = {
    "duration_predictor": "onnx/duration_predictor.onnx",
    "text_encoder": "onnx/text_encoder.onnx",
    "vector_estimator": "onnx/vector_estimator.onnx",
    "vocoder": "onnx/vocoder.onnx",
}
STYLE_COMPONENTS = tuple([f"F{i}" for i in range(1, 6)] + [f"M{i}" for i in range(1, 6)])
TTS_CONFIG = {
    "ae": {"sample_rate": 44100, "base_chunk_size": 512},
    "ttl": {"chunk_compress_factor": 6, "latent_dim": 24},
}
TTS_BYTES = json.dumps(TTS_CONFIG, sort_keys=True).encode()
SMALL_FILES = {
    "onnx/tts.json": TTS_BYTES,
    "onnx/unicode_indexer.json": b"[0, 1, 2]",
    **{
        f"voice_styles/{name}.json": json.dumps({"name": name}).encode()
        for name in STYLE_COMPONENTS
    },
}


def _tree() -> list[dict]:
    entries = []
    for index, path in enumerate(
        ["onnx/tts.json", "onnx/unicode_indexer.json", *MODEL_PATHS.values()]
    ):
        metadata = {
            "path": path,
            "size": len(SMALL_FILES[path]) if path in SMALL_FILES else 100 + index,
        }
        if path in MODEL_PATHS.values():
            metadata["lfs"] = {"oid": f"{index + 1:064x}", "size": 100 + index}
        entries.append(metadata)
    entries.extend(
        {"path": path, "size": len(payload)}
        for path, payload in SMALL_FILES.items()
        if path.startswith("voice_styles/")
    )
    return entries


def _mock_upstream(monkeypatch) -> list[str]:
    requested: list[str] = []
    monkeypatch.setattr(supertonic, "_repository_tree", lambda repository, revision: _tree())

    def read_url(url: str) -> bytes:
        requested.append(url)
        for path, payload in SMALL_FILES.items():
            if url.endswith(f"/{path}?download=true"):
                return payload
        raise AssertionError(f"Unexpected download: {url}")

    monkeypatch.setattr(supertonic, "_read_url", read_url)
    return requested


def _build(monkeypatch) -> tuple[dict, list[str]]:
    requested = _mock_upstream(monkeypatch)
    catalog = supertonic.build_catalog(
        repository=REPOSITORY,
        revision="main",
        resolved_revision=REVISION,
    )
    return catalog, requested


class TestBuildCatalog:
    def test_build_is_deterministic_sorted_and_uses_remote_integrity(self, monkeypatch) -> None:
        first, requested = _build(monkeypatch)
        second = supertonic.build_catalog(
            repository=REPOSITORY,
            revision="main",
            resolved_revision=REVISION,
        )

        assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)
        bundle = first["bundles"]["supertonic-3"]
        assert [artifact["role"] for artifact in bundle["artifacts"]] == [
            "config",
            "unicode_indexer",
            *(["model"] * 4),
            *(["voice_style"] * 10),
        ]
        assert [artifact["component"] for artifact in bundle["artifacts"][2:6]] == sorted(
            MODEL_PATHS
        )
        assert bundle["sample_rate"] == 44100
        assert bundle["runtime"] == {
            "layout": "supertonic-3-v1",
            "sample_rate": 44100,
            "base_chunk_size": 512,
            "chunk_compress_factor": 6,
            "latent_dim": 24,
            "chunk_size": 3072,
            "latent_width": 144,
        }
        assert bundle["languages"] == [*supertonic._LANGUAGES]
        assert bundle["voices"] == [
            {"name": name, "artifact_component": name} for name in STYLE_COMPONENTS
        ]
        assert len(bundle["artifacts"]) == 16
        assert bundle["artifacts"][0]["sha256"] == hashlib.sha256(TTS_BYTES).hexdigest()
        assert bundle["artifacts"][0]["size"] == len(TTS_BYTES)
        assert bundle["artifacts"][2]["sha256"] == f"{3:064x}"
        assert not any(Path(path).name in url for path in MODEL_PATHS.values() for url in requested)
        assert len(requested) == 24

    def test_resolves_named_revision_to_exact_sha(self, monkeypatch) -> None:
        urls: list[str] = []

        def read_url(url: str) -> bytes:
            urls.append(url)
            return json.dumps({"sha": REVISION}).encode()

        monkeypatch.setattr(supertonic, "_read_url", read_url)
        assert supertonic.resolve_revision(REPOSITORY, "main") == REVISION
        assert urls == [supertonic.huggingface_api_url(REPOSITORY, "main")]
        assert supertonic.resolve_revision(REPOSITORY, REVISION) == REVISION

    def test_requires_all_models_and_at_least_one_style(self, monkeypatch) -> None:
        monkeypatch.setattr(supertonic, "_repository_tree", lambda repository, revision: [])
        with pytest.raises(supertonic.CatalogError, match="Missing upstream asset"):
            supertonic.build_catalog(resolved_revision=REVISION)

        tree = [
            {"path": path, "size": 1, "lfs": {"oid": "1" * 64}}
            for path in [
                "onnx/tts.json",
                "onnx/unicode_indexer.json",
                *MODEL_PATHS.values(),
            ]
        ]
        monkeypatch.setattr(supertonic, "_repository_tree", lambda repository, revision: tree)
        with pytest.raises(supertonic.CatalogError, match="No voice_styles"):
            supertonic.build_catalog(resolved_revision=REVISION)

    def test_runtime_is_taken_from_upstream_config(self) -> None:
        sample_rate, runtime = supertonic._runtime_from_tts(TTS_BYTES)
        assert sample_rate == 44100
        assert runtime["chunk_size"] == 3072
        assert runtime["latent_width"] == 144
        with pytest.raises(supertonic.CatalogError, match="positive integer"):
            supertonic._runtime_from_tts(b'{"ae":{"sample_rate":0}}')


class TestVerifyCatalog:
    def test_verifier_accepts_builder_output_and_matching_source_without_network(
        self, monkeypatch
    ) -> None:
        catalog, _ = _build(monkeypatch)
        source = {"schema": 1, **catalog["source"]}
        monkeypatch.setattr(supertonic, "_read_url", lambda url: pytest.fail("network access"))
        monkeypatch.setattr(
            supertonic, "_repository_tree", lambda *args: pytest.fail("network access")
        )

        supertonic.verify_catalog(catalog, source)

    @pytest.mark.parametrize(
        ("mutate", "message"),
        [
            (lambda c: c["source"].update(bundle_count=2), "bundle_count"),
            (lambda c: c["source"].update(revision="b" * 40), "artifact URL is not pinned"),
            (
                lambda c: c["bundles"]["supertonic-3"]["artifacts"][0].update(path="../escape"),
                "unsafe artifact path",
            ),
            (
                lambda c: c["bundles"]["supertonic-3"]["artifacts"][0].update(size=0),
                "positive integer",
            ),
            (
                lambda c: c["bundles"]["supertonic-3"]["artifacts"][0].update(sha256="A" * 64),
                "lowercase hex",
            ),
            (lambda c: c["bundles"]["supertonic-3"]["artifacts"].pop(2), "missing model component"),
            (
                lambda c: c["bundles"]["supertonic-3"]["voices"][0].update(
                    artifact_component="absent"
                ),
                "missing style artifact",
            ),
            (
                lambda c: c["bundles"]["supertonic-3"].update(sample_rate=22050),
                "sample rate does not match",
            ),
            (
                lambda c: c["bundles"]["supertonic-3"]["runtime"].update(chunk_size=1),
                "chunk size does not match",
            ),
        ],
    )
    def test_rejects_integrity_and_contract_mutations(
        self, monkeypatch, mutate, message: str
    ) -> None:
        catalog, _ = _build(monkeypatch)
        damaged = copy.deepcopy(catalog)
        mutate(damaged)
        with pytest.raises(supertonic.CatalogError, match=message):
            supertonic.verify_catalog(damaged)

    def test_rejects_mismatched_source_json(self, monkeypatch) -> None:
        catalog, _ = _build(monkeypatch)
        source = {"schema": 1, **catalog["source"]}
        source["revision"] = "b" * 40
        with pytest.raises(supertonic.CatalogError, match="does not match catalog provenance"):
            supertonic.verify_catalog(catalog, source)

    def test_load_catalog_verifies_companion_source_file(self, monkeypatch, tmp_path: Path) -> None:
        catalog, _ = _build(monkeypatch)
        catalog_path = tmp_path / "bundles.json"
        source_path = tmp_path / "source.json"
        catalog_path.write_text(json.dumps(catalog), encoding="utf-8")
        source_path.write_text(json.dumps({"schema": 1, **catalog["source"]}), encoding="utf-8")

        loaded = supertonic.load_catalog(catalog_path, source_path)
        assert loaded == catalog
