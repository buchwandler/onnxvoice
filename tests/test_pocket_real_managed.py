"""Optional managed Pocket catalog-to-inference smoke test.

Set ``ONNXVOICE_POCKET_BUNDLE_DIR`` to a checked-out real bundle directory and
``ONNXVOICE_POCKET_CATALOG`` to its matching canonical catalog. The test rewrites
artifact URLs to local file URLs so it exercises catalog resolution and managed
store installation without requiring a second network download.
"""

from __future__ import annotations

import copy
import json
import os
from pathlib import Path

import numpy as np
import pytest

from onnxvoice import OnnxVoice


def test_real_managed_pocket_bundle_round_trip(tmp_path: Path) -> None:
    root_value = os.environ.get("ONNXVOICE_POCKET_BUNDLE_DIR")
    catalog_value = os.environ.get("ONNXVOICE_POCKET_CATALOG")
    if not root_value or not catalog_value:
        pytest.skip("ONNXVOICE_POCKET_BUNDLE_DIR and ONNXVOICE_POCKET_CATALOG are not configured")

    root = Path(root_value)
    catalog_path = Path(catalog_value)
    try:
        catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        pytest.fail(f"Unable to read managed Pocket catalog: {exc}")
    if not isinstance(catalog, dict) or not isinstance(catalog.get("bundles"), dict):
        pytest.fail("Managed Pocket catalog must contain a bundles mapping")

    requested_id = os.environ.get("ONNXVOICE_POCKET_BUNDLE_ID")
    if requested_id is None:
        requested_id = root.name
    entry = catalog["bundles"].get(requested_id)
    if not isinstance(entry, dict):
        if len(catalog["bundles"]) != 1:
            pytest.skip(f"Managed catalog has no unique bundle for {requested_id!r}")
        requested_id, entry = next(iter(catalog["bundles"].items()))
    entry = copy.deepcopy(entry)

    artifacts = entry.get("artifacts")
    if not isinstance(artifacts, list):
        pytest.fail("Managed Pocket catalog bundle artifacts must be a list")
    missing: list[str] = []
    for artifact in artifacts:
        if not isinstance(artifact, dict):
            pytest.fail("Managed Pocket catalog artifact must be an object")
        filename = artifact.get("filename")
        if not isinstance(filename, str):
            pytest.fail("Managed Pocket catalog artifact filename is missing")
        local_path = root / filename
        if not local_path.is_file():
            missing.append(filename)
        else:
            artifact["url"] = local_path.as_uri()
    if missing:
        pytest.skip(f"Managed Pocket bundle is incomplete; missing {', '.join(sorted(missing))}")

    managed_catalog = copy.deepcopy(catalog)
    managed_catalog["bundles"] = {requested_id: entry}
    local_catalog = tmp_path / "catalog.json"
    local_catalog.write_text(json.dumps(managed_catalog), encoding="utf-8")

    manager = OnnxVoice(
        cache_dir=tmp_path / "cache",
        catalog_sources={"pocket": str(local_catalog)},
    )
    quality = os.environ.get("ONNXVOICE_POCKET_QUALITY", "int8")
    try:
        installation = manager.install(f"pocket:{requested_id}", quality=quality)
        manager.store.verify(installation)
        with manager.open(installation, provider="CPUExecutionProvider") as runtime:
            voice = runtime.prepare_voice(
                np.zeros(24000, dtype=np.float32),
                sample_rate=installation.sample_rate or 24000,
            )
            result = runtime.infer(
                [1, 2, 3],
                voice_state=voice,
                temperature=0.7,
                lsd_steps=1,
                max_frames=2,
            )
            assert voice.embeddings.ndim == 3
            assert result.audio.ndim == 1
            assert result.audio.size > 0
            assert np.isfinite(result.audio).all()
            assert np.any(result.audio != 0)
            assert result.sample_rate == installation.sample_rate
    finally:
        for installation in manager.installed("pocket"):
            manager.store.remove(installation.system, installation.storage_id or installation.id)
