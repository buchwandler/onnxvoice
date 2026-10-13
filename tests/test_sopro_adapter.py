from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from onnxvoice import OnnxVoice
from onnxvoice.systems import SoproAdapter


def test_adapter_discovers_graph_contract_and_configures_each_lazy_session(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    profile = "native-test"
    root = tmp_path / "bundle"
    root.mkdir()
    tokenizer = root / "tokenizer.json"
    tokenizer.write_text("{}", encoding="utf-8")
    graph = root / profile / "custom_encoder.onnx"
    graph.parent.mkdir()
    graph.write_bytes(b"graph bytes")
    shard = root / profile / "custom-weights.bin"
    shard.write_bytes(b"external bytes")

    def declaration(path: Path) -> dict[str, Any]:
        payload = path.read_bytes()
        return {
            "file": path.name,
            "bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
        }

    graph_declaration = declaration(graph)
    graph_declaration["shard"] = "custom-data"
    manifest = {
        "format": 1,
        "model": "fixture",
        "revision": "fixture-revision",
        "sampleRate": 24000,
        "tokenizer": "tokenizer.json",
        "profiles": {
            profile: {
                "graphs": {"custom_encoder": graph_declaration},
                "shards": {"custom-data": declaration(shard)},
            }
        },
    }
    manifest_path = root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    files = {
        "manifest": manifest_path,
        "tokenizer": tokenizer,
        "graph:custom_encoder": graph,
        "shard:custom-data": shard,
    }

    created: list[tuple[Path, dict[str, Any], Any]] = []
    session_options = object()

    class Session:
        def __init__(self, path: Path, **kwargs: Any) -> None:
            self.closed = 0
            created.append((path, kwargs, self))

        def close(self) -> None:
            self.closed += 1

    monkeypatch.setattr("onnxvoice.systems.sopro.OnnxSession", Session)
    runtime = OnnxVoice.open_local(
        system="sopro",
        files=files,
        sample_rate=24000,
        runtime={"layout": "multi", "profile": profile},
        providers=["CPUExecutionProvider"],
        provider_options=[{"device_id": "0"}],
        session_options=session_options,
    )
    assert isinstance(runtime, SoproAdapter)
    try:
        assert runtime.graphs == ("custom_encoder",)
        assert runtime.shards == ("custom-data",)
        assert runtime._sessions == {}
        assert created == []

        runtime.session("custom_encoder")
        assert len(created) == 1
        created_path, kwargs, session = created[0]
        assert created_path == graph.resolve()
        assert kwargs == {
            "component": "custom_encoder",
            "providers": ["CPUExecutionProvider"],
            "provider_options": [{"device_id": "0"}],
            "session_options": session_options,
        }
        runtime.close()
        runtime.close()
        assert session.closed == 1
    finally:
        runtime.close()
