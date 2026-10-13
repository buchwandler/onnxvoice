from __future__ import annotations

import gzip
import hashlib
import json
import math
import os
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from onnxvoice import OnnxVoice, registered_systems
from onnxvoice.errors import (
    CapabilityError,
    IntegrityError,
    ManifestError,
    RuntimeContractError,
    UnsafePathError,
)
from onnxvoice.systems import SoproAdapter, SoproVoiceState
from onnxvoice.systems.sopro import _sopro_istft
from onnxvoice.types import RuntimeDiagnostic, SessionDiagnostic, TensorSpec

PROFILE = "wasm-uint8"
REVISION = "7a645ef2aaecfea9"


def _write_bundle(root: Path) -> tuple[dict[str, Path], dict[str, Any]]:
    payloads = {
        "tokenizer.json": b'{"model":{"type":"wordpiece"}}',
        "common/reference.onnx": b"reference graph fixture",
        "common/weights-reference.bin": b"reference shard fixture",
        f"{PROFILE}/semantic_core.onnx": b"semantic graph fixture",
        f"{PROFILE}/weights-semantic.bin": b"semantic shard fixture",
    }
    paths: dict[str, Path] = {}
    for relative, payload in payloads.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
        paths[relative] = path

    def record(
        relative: str, *, url: str | None = None, shard: str | None = None
    ) -> dict[str, Any]:
        payload = payloads[relative]
        result: dict[str, Any] = {
            "file": Path(relative).name,
            "bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
        }
        if url is not None:
            result["url"] = url
        if shard is not None:
            result["shard"] = shard
        return result

    manifest = {
        "format": 1,
        "model": "sopro-v2-turbo",
        "revision": REVISION,
        "sampleRate": 24000,
        "tokenizer": "tokenizer.json",
        "profiles": {
            PROFILE: {
                "provider": "wasm",
                "dtype": "float32",
                "graphs": {
                    "reference": record(
                        "common/reference.onnx", url="../common/reference.onnx", shard="reference"
                    ),
                    "semantic_core": record(f"{PROFILE}/semantic_core.onnx", shard="semantic"),
                },
                "shards": {
                    "reference": record(
                        "common/weights-reference.bin", url="../common/weights-reference.bin"
                    ),
                    "semantic": record(f"{PROFILE}/weights-semantic.bin"),
                },
            }
        },
    }
    manifest_path = root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    paths["manifest.json"] = manifest_path
    files = {
        "manifest": manifest_path,
        "tokenizer": paths["tokenizer.json"],
        "graph:reference": paths["common/reference.onnx"],
        "graph:semantic_core": paths[f"{PROFILE}/semantic_core.onnx"],
        "shard:reference": paths["common/weights-reference.bin"],
        "shard:semantic": paths[f"{PROFILE}/weights-semantic.bin"],
    }
    return files, manifest


def _open_local(files: dict[str, Path], *, profile: str = PROFILE, **kwargs: Any) -> SoproAdapter:
    runtime = {"layout": "multi", "profile": profile, "revision": REVISION}
    return OnnxVoice.open_local(
        system="sopro",
        files=files,
        sample_rate=24000,
        runtime=runtime,
        metadata={"model": "sopro-v2-turbo", "source_commit": "a" * 40},
        **kwargs,
    )


class _FakeSession:
    def __init__(self, path: Path, **kwargs: Any) -> None:
        self.model = path
        self.kwargs = kwargs
        self.closed = 0

    @property
    def input_specs(self) -> tuple[TensorSpec, ...]:
        return (TensorSpec(name="input", ort_type="tensor(float)", shape=(1, 2)),)

    @property
    def output_specs(self) -> tuple[TensorSpec, ...]:
        return (TensorSpec(name="output", ort_type="tensor(float)", shape=(1, 2)),)

    @property
    def resolved_providers(self) -> tuple[str, ...]:
        return ("CPUExecutionProvider",)

    def run(self, inputs: dict[str, Any]) -> list[np.ndarray]:
        return [np.asarray(inputs["input"], dtype=np.float32)]

    def diagnostics(self) -> SessionDiagnostic:
        return SessionDiagnostic(
            component=self.kwargs["component"],
            model_path=self.model,
            providers_requested=("CPUExecutionProvider",),
            providers_active=("CPUExecutionProvider",),
            inputs=("input",),
            outputs=("output",),
        )

    def close(self) -> None:
        self.closed += 1


def test_sopro_is_builtin_and_local_open_maps_manifest_graphs_and_shards(tmp_path: Path) -> None:
    files, _ = _write_bundle(tmp_path)
    runtime = _open_local({key: files[key] for key in reversed(files)})
    try:
        assert "sopro" in registered_systems()
        assert isinstance(runtime, SoproAdapter)
        assert runtime.installation.path == tmp_path.resolve()
        assert runtime.profile == PROFILE
        assert runtime.graphs == ("reference", "semantic_core")
        assert runtime.shards == ("reference", "semantic")
        assert runtime.graph_shards == {
            "reference": (tmp_path / "common/weights-reference.bin").resolve(),
            "semantic_core": (tmp_path / f"{PROFILE}/weights-semantic.bin").resolve(),
        }
        assert runtime.installation.artifact("manifest").path == files["manifest"].resolve()
        assert runtime.installation.artifact("tokenizer").path == files["tokenizer"].resolve()
        assert runtime._sessions == {}
        assert runtime.diagnostics() == RuntimeDiagnostic(
            system="sopro", ref=runtime.installation.ref, layout="multi", sessions=()
        )
    finally:
        runtime.close()


def test_manifest_graph_resolves_shard_filename_alias(tmp_path: Path) -> None:
    files, manifest = _write_bundle(tmp_path)
    manifest["profiles"][PROFILE]["graphs"]["reference"]["shard"] = "weights-reference.bin"
    files["manifest"].write_text(json.dumps(manifest), encoding="utf-8")
    runtime = _open_local(files)
    try:
        assert (
            runtime.graph_shards["reference"]
            == (tmp_path / "common/weights-reference.bin").resolve()
        )
    finally:
        runtime.close()


def test_graph_sessions_are_lazy_provider_configured_and_close_idempotently(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    files, _ = _write_bundle(tmp_path)
    created: list[_FakeSession] = []
    provider_options = {"CPUExecutionProvider": {"arena_extend_strategy": "kSameAsRequested"}}
    session_options = object()

    def make_session(path: Path, **kwargs: Any) -> _FakeSession:
        session = _FakeSession(path, **kwargs)
        created.append(session)
        return session

    monkeypatch.setattr("onnxvoice.systems.sopro.OnnxSession", make_session)
    runtime = _open_local(
        files,
        providers=["CPUExecutionProvider"],
        provider_options=provider_options,
        session_options=session_options,
    )
    try:
        assert created == []
        outputs = runtime.run_graph("reference", {"input": np.asarray([[1, 2]], dtype=np.float32)})
        assert outputs[0].tolist() == [[1.0, 2.0]]
        _ = runtime.graph_specs("semantic_core")
        assert [session.model for session in created] == [
            (tmp_path / "common/reference.onnx").resolve(),
            (tmp_path / f"{PROFILE}/semantic_core.onnx").resolve(),
        ]
        for session in created:
            assert session.kwargs == {
                "component": "reference"
                if session.model.name == "reference.onnx"
                else "semantic_core",
                "providers": ["CPUExecutionProvider"],
                "provider_options": provider_options,
                "session_options": session_options,
            }
        diagnostic = runtime.diagnostics()
        assert [item.component for item in diagnostic.sessions] == ["reference", "semantic_core"]
        assert [item.model_path for item in diagnostic.sessions] == [
            session.model for session in created
        ]
        runtime.close()
        runtime.close()
        assert [session.closed for session in created] == [1, 1]
    finally:
        runtime.close()


@pytest.mark.parametrize(
    ("change", "error", "message"),
    [
        (lambda manifest: manifest.update(format=2), ManifestError, "format=1"),
        (
            lambda manifest: manifest["profiles"][PROFILE]["graphs"]["reference"].update(
                url="../../outside.onnx"
            ),
            UnsafePathError,
            "escapes the bundle",
        ),
        (
            lambda manifest: manifest["profiles"][PROFILE]["graphs"]["reference"].update(
                sha256="A" * 64
            ),
            ManifestError,
            "lowercase",
        ),
    ],
)
def test_local_open_rejects_invalid_manifest_and_unsafe_paths(
    tmp_path: Path, change, error: type[Exception], message: str
) -> None:
    files, manifest = _write_bundle(tmp_path)
    change(manifest)
    files["manifest"].write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(error, match=message):
        _open_local(files)


def test_local_open_rejects_unknown_or_unselected_profile(tmp_path: Path) -> None:
    files, _ = _write_bundle(tmp_path)
    with pytest.raises(CapabilityError, match="Unknown Sopro profile"):
        _open_local(files, profile="unknown")
    with pytest.raises(CapabilityError, match="selected explicitly"):
        OnnxVoice.open_local(system="sopro", files=files, sample_rate=24000)


def test_local_open_verifies_manifest_hashes_and_path_mapping(tmp_path: Path) -> None:
    files, _ = _write_bundle(tmp_path)
    (tmp_path / "common/reference.onnx").write_bytes(b"x" * len(b"reference graph fixture"))
    with pytest.raises(IntegrityError, match="SHA-256 mismatch"):
        _open_local(files)

    files, _ = _write_bundle(tmp_path / "other")
    files["graph:reference"] = files["graph:semantic_core"]
    with pytest.raises(ManifestError, match="does not match the manifest path"):
        _open_local(files)


def _configure_reference_frontend(
    files: dict[str, Path], manifest: dict[str, Any]
) -> dict[str, Any]:
    fixture_root = Path(__file__).parent / "fixtures" / "sopro"
    golden = json.loads((fixture_root / "reference_golden.json").read_text(encoding="utf-8"))
    tokenizer = gzip.decompress((fixture_root / "tokenizer-v2-turbo.json.gz").read_bytes())
    files["tokenizer"].write_bytes(tokenizer)
    config = golden["frontend_config"]
    config["model"].update(
        {
            "acoustic_mel_mean": [0.0] * 100,
            "acoustic_mel_std": [1.0] * 100,
            "cond_hidden_dim": 512,
            "semantic_vocab_size": 4375,
            "text_vocab_size": 8192,
            "max_text_len": 2048,
            "style_prefix_tokens": 8,
            "ar_model_dim": 512,
            "ar_blocks": 12,
            "ar_heads": 8,
            "ar_kv_heads": 8,
            "acoustic_dit_dim_head": 64,
            "acoustic_upsampler_kernel_size": 3,
        }
    )
    config["generation"] = {
        "temperature": 0.8,
        "top_p": 0.9,
        "top_k": 25,
        "steps": 2,
        "max_seconds": 30.0,
        "min_seconds": 0.4,
        "style_tokens": 160,
        "prompt_tokens": 120,
        "ref_seconds": 10.0,
    }
    config["vocoder"] = {
        "sample_rate": 24000,
        "n_fft": 1024,
        "hop_length": 256,
        "n_mels": 100,
        "dim": 512,
        "num_layers": 8,
        "max_magnitude": 100.0,
        "block_lookaheads": [3] * 8,
    }
    manifest["config"] = config
    manifest["tokenSamples"] = 1024
    manifest["hopRatio"] = 4
    manifest["steps"] = 2
    files["manifest"].write_text(json.dumps(manifest), encoding="utf-8")
    return golden


class _FakeReferenceSession:
    def __init__(self, path: Path, **kwargs: Any) -> None:
        self.model = path
        self.kwargs = kwargs
        self.inputs: dict[str, np.ndarray] = {}
        self.closed = 0

    @property
    def input_specs(self) -> tuple[TensorSpec, ...]:
        return (
            TensorSpec("semantic_mel", "tensor(float)", (1, 80, "frames")),
            TensorSpec("interp_left", "tensor(int64)", ("tokens",)),
            TensorSpec("interp_right", "tensor(int64)", ("tokens",)),
            TensorSpec("interp_weight", "tensor(float)", ("tokens",)),
            TensorSpec("speaker_mel", "tensor(float)", (1, 80, "frames")),
        )

    @property
    def output_names(self) -> tuple[str, ...]:
        return ("cond_vec", "semantic_tokens")

    def run(self, inputs: dict[str, np.ndarray]) -> list[np.ndarray]:
        self.inputs = {name: np.array(value, copy=True) for name, value in inputs.items()}
        token_count = inputs["interp_left"].size
        return [
            np.zeros((1, 512), dtype=np.float32),
            np.arange(token_count, dtype=np.int64)[np.newaxis, :],
        ]

    def diagnostics(self) -> SessionDiagnostic:
        return SessionDiagnostic(
            component="reference",
            model_path=self.model,
            providers_requested=("CPUExecutionProvider",),
            providers_active=("CPUExecutionProvider",),
            inputs=tuple(self.inputs),
            outputs=self.output_names,
        )

    def close(self) -> None:
        self.closed += 1


def test_prepare_voice_builds_typed_reference_state_and_official_graph_inputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    files, manifest = _write_bundle(tmp_path)
    golden = _configure_reference_frontend(files, manifest)
    created: list[_FakeReferenceSession] = []

    def make_session(path: Path, **kwargs: Any) -> _FakeReferenceSession:
        session = _FakeReferenceSession(path, **kwargs)
        created.append(session)
        return session

    monkeypatch.setattr("onnxvoice.systems.sopro.OnnxSession", make_session)
    runtime = _open_local(files)
    tone = golden["tones"]["steady"]
    indices = np.arange(tone["samples"], dtype=np.float64)
    waveform = (
        0.08 * np.sin(2 * math.pi * 220 * indices / tone["sample_rate"])
        + 0.025 * np.sin(2 * math.pi * 791 * indices / tone["sample_rate"])
    ).astype(np.float32)
    try:
        voice = runtime.prepare_voice(waveform, tone["sample_rate"])
        session = created[0]
        expected = tone["features"]
        assert set(session.inputs) == {
            "semantic_mel",
            "interp_left",
            "interp_right",
            "interp_weight",
            "speaker_mel",
        }
        assert session.inputs["semantic_mel"].shape == (1, *expected["semantic_shape"])
        assert session.inputs["speaker_mel"].shape == (1, *expected["speaker_shape"])
        assert session.inputs["interp_left"].dtype == np.int64
        assert session.inputs["interp_right"].dtype == np.int64
        for name in ("semantic", "speaker", "acoustic"):
            graph_name = f"{name}_mel" if name != "acoustic" else None
            if graph_name is not None:
                flat = session.inputs[graph_name].reshape(-1)
            else:
                flat = voice.mel.reshape(-1)
            expected_values = expected[name]
            positions = np.asarray([int(index) for index in expected_values], dtype=np.int64)
            values = np.asarray([expected_values[str(index)] for index in positions])
            np.testing.assert_allclose(flat[positions], values, rtol=1.0e-5, atol=5.0e-6)
        assert voice.cond_vec.shape == (512,)
        assert voice.semantic_tokens.tolist() == list(range(expected["token_count"]))
        assert voice.mel.shape == tuple(expected["acoustic_shape"])
        assert voice.sample_rate == 24000
        assert voice.level_db == pytest.approx(-19.8, abs=1.0e-6)
        assert not voice.cond_vec.flags.writeable
        assert not voice.semantic_tokens.flags.writeable
        assert not voice.mel.flags.writeable
        resampled = golden["tones"]["resampled_16k"]
        source_indices = np.arange(resampled["samples"], dtype=np.float64)
        source_audio = (
            0.08 * np.sin(2 * math.pi * 220 * source_indices / resampled["sample_rate"])
            + 0.025 * np.sin(2 * math.pi * 791 * source_indices / resampled["sample_rate"])
        ).astype(np.float32)
        resampled_voice = runtime.prepare_voice(source_audio, resampled["sample_rate"])
        expected_resampled = resampled["features"]
        assert session.inputs["semantic_mel"].shape == (1, *expected_resampled["semantic_shape"])
        assert session.inputs["speaker_mel"].shape == (1, *expected_resampled["speaker_shape"])
        for name, array in (
            ("semantic", session.inputs["semantic_mel"]),
            ("speaker", session.inputs["speaker_mel"]),
            ("acoustic", resampled_voice.mel),
        ):
            flat = array.reshape(-1)
            expected_values = expected_resampled[name]
            positions = np.asarray([int(index) for index in expected_values], dtype=np.int64)
            values = np.asarray([expected_values[str(index)] for index in positions])
            np.testing.assert_allclose(flat[positions], values, rtol=1.0e-5, atol=3.0e-5)
        assert resampled_voice.level_db == pytest.approx(-19.8, abs=1.0e-6)
        text = runtime.prepare_text("olá, mundo!", "pt")
        assert text.normalized_text == "<|lang_pt|> Olá, mundo!"
        assert list(text.token_ids) == [1, 276, 5, 423, 326, 731, 264, 2157, 702, 2]
    finally:
        runtime.close()
    assert created[0].closed == 1


def test_real_reference_graph_matches_pinned_onnx_outputs(
    tmp_path: Path,
) -> None:
    graph_setting = os.environ.get("ONNXVOICE_SOPRO_REFERENCE_GRAPH")
    if not graph_setting:
        pytest.skip("set ONNXVOICE_SOPRO_REFERENCE_GRAPH to the pinned reference.onnx")
    try:
        import onnxruntime as ort
    except ImportError:
        pytest.skip("onnxruntime is not installed")

    graph_source = Path(graph_setting).expanduser().resolve()
    shard_source = graph_source.parent / "weights-reference.bin"
    if not graph_source.is_file() or not shard_source.is_file():
        pytest.skip("the reference graph and its external-data shard are not available")
    files, manifest = _write_bundle(tmp_path)
    golden = _configure_reference_frontend(files, manifest)
    graph_target = files["graph:reference"]
    shard_target = files["shard:reference"]
    shutil.copyfile(graph_source, graph_target)
    shutil.copyfile(shard_source, shard_target)

    def digest(path: Path) -> str:
        hasher = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                hasher.update(chunk)
        return hasher.hexdigest()

    graph_golden = golden["reference_graph"]
    assert digest(graph_target) == graph_golden["graph_sha256"]
    assert digest(shard_target) == graph_golden["external_data_sha256"]
    profile = manifest["profiles"][PROFILE]
    profile["graphs"]["reference"].update(
        bytes=graph_target.stat().st_size,
        sha256=graph_golden["graph_sha256"],
    )
    profile["shards"]["reference"].update(
        bytes=shard_target.stat().st_size,
        sha256=graph_golden["external_data_sha256"],
    )
    files["manifest"].write_text(json.dumps(manifest), encoding="utf-8")
    runtime = _open_local(files, providers=["CPUExecutionProvider"])
    tone = golden["tones"]["steady"]
    indices = np.arange(tone["samples"], dtype=np.float64)
    waveform = (
        0.08 * np.sin(2 * math.pi * 220 * indices / tone["sample_rate"])
        + 0.025 * np.sin(2 * math.pi * 791 * indices / tone["sample_rate"])
    ).astype(np.float32)
    try:
        voice = runtime.prepare_voice(waveform, tone["sample_rate"])
        cond_samples = graph_golden["cond_vec"]["samples"]
        positions = np.asarray([int(index) for index in cond_samples], dtype=np.int64)
        values = np.asarray([cond_samples[str(index)] for index in positions])
        np.testing.assert_allclose(voice.cond_vec[positions], values, rtol=1.0e-5, atol=1.0e-5)
        assert voice.semantic_tokens.tolist() == graph_golden["semantic_tokens"]["values"]
        assert voice.cond_vec.shape == (512,)
        assert voice.mel.shape == (100, 76)
        assert ort.get_available_providers()
    finally:
        runtime.close()


def _configure_semantic_bundle(
    files: dict[str, Path],
    manifest: dict[str, Any],
    *,
    include_prefix: bool = True,
) -> None:
    _configure_reference_frontend(files, manifest)
    model = manifest["config"]["model"]
    model.update(
        {
            "semantic_vocab_size": 16,
            "text_vocab_size": 8192,
            "max_text_len": 2048,
            "style_prefix_tokens": 2,
            "ar_model_dim": 8,
            "acoustic_dit_dim_head": 4,
            "ar_blocks": 2,
            "ar_heads": 2,
            "ar_kv_heads": 1,
        }
    )
    manifest["config"]["generation"].update({"style_tokens": 3, "prompt_tokens": 4})
    manifest["config"]["vocoder"].update({"dim": 8, "num_layers": 2, "block_lookaheads": [3, 3]})
    if include_prefix:
        prefix_path = files["graph:semantic_core"].with_name("semantic_prefix.onnx")
        payload = b"semantic prefix fixture graph"
        prefix_path.write_bytes(payload)
        manifest["profiles"][PROFILE]["graphs"]["semantic_prefix"] = {
            "file": prefix_path.name,
            "bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
            "shard": "semantic",
        }
        files["graph:semantic_prefix"] = prefix_path
    files["manifest"].write_text(json.dumps(manifest), encoding="utf-8")


def _add_fake_acoustic_graphs(files: dict[str, Path], manifest: dict[str, Any]) -> None:
    profile = manifest["profiles"][PROFILE]
    for graph, shard in (
        ("acoustic_condition", "condition"),
        ("acoustic_offline_2", "acoustic"),
        ("vocoder", "vocoder"),
    ):
        path = files["graph:semantic_core"].with_name(f"{graph}.onnx")
        payload = f"{graph} fixture graph".encode()
        path.write_bytes(payload)
        profile["graphs"][graph] = {
            "file": path.name,
            "bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
            "shard": shard,
        }
        files[f"graph:{graph}"] = path
    for shard in ("condition", "acoustic", "vocoder"):
        path = files["shard:semantic"].with_name(f"weights-{shard}.bin")
        payload = f"{shard} fixture weights".encode()
        path.write_bytes(payload)
        profile["shards"][shard] = {
            "file": path.name,
            "bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
        }
        files[f"shard:{shard}"] = path
    files["manifest"].write_text(json.dumps(manifest), encoding="utf-8")


def _configure_pinned_native_bundle(
    files: dict[str, Path],
    manifest: dict[str, Any],
    source_root: Path,
    golden: dict[str, Any],
) -> None:
    actual = json.loads((source_root / "manifest.json").read_text(encoding="utf-8"))
    common = source_root / "model" / "common"
    profile_root = source_root / "model" / PROFILE
    profile = manifest["profiles"][PROFILE]

    def digest(path: Path) -> str:
        hasher = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                hasher.update(chunk)
        return hasher.hexdigest()

    def record(source: Path, target: Path, *, shard: str | None = None) -> dict[str, Any]:
        shutil.copyfile(source, target)
        result = {"file": target.name, "bytes": target.stat().st_size, "sha256": digest(target)}
        if shard is not None:
            result["shard"] = shard
        return result

    profile["graphs"]["reference"].update(
        bytes=(common / "reference.onnx").stat().st_size,
        sha256=golden["graphs"]["reference"],
    )
    profile["shards"]["reference"].update(
        bytes=(common / "weights-reference.bin").stat().st_size,
        sha256=golden["shards"]["reference"],
    )
    graph_sources = {
        "semantic_prefix": common / "semantic_prefix.onnx",
        "semantic_core": common / "semantic_core.onnx",
        "acoustic_condition": profile_root / "acoustic_condition.onnx",
        "acoustic_offline_2": profile_root / "acoustic_offline_2.onnx",
        "vocoder": profile_root / "vocoder.onnx",
    }
    graph_shards = {
        "semantic_prefix": "semantic",
        "semantic_core": "semantic",
        "acoustic_condition": "condition",
        "acoustic_offline_2": "acoustic",
        "vocoder": "vocoder",
    }
    for name, source in graph_sources.items():
        target = (
            files["graph:semantic_core"].with_name(f"{name}.onnx")
            if name != "semantic_core"
            else files["graph:semantic_core"]
        )
        item = record(source, target, shard=graph_shards[name])
        if item["sha256"] != golden["graphs"][name]:
            raise AssertionError(f"unexpected pinned Sopro graph digest for {name}")
        profile["graphs"][name] = item
        files[f"graph:{name}"] = target

    shard_sources = {
        "semantic": common / "weights-semantic.bin",
        "condition": profile_root / "weights-condition.bin",
        "acoustic": profile_root / "weights-acoustic.bin",
        "vocoder": profile_root / "weights-vocoder.bin",
    }
    for name, source in shard_sources.items():
        target = (
            files["shard:semantic"]
            if name == "semantic"
            else files["shard:semantic"].with_name(f"weights-{name}.bin")
        )
        item = record(source, target)
        if item["sha256"] != golden["shards"][name]:
            raise AssertionError(f"unexpected pinned Sopro shard digest for {name}")
        profile["shards"][name] = item
        files[f"shard:{name}"] = target

    reference_graph = record(common / "reference.onnx", files["graph:reference"])
    reference_shard = record(common / "weights-reference.bin", files["shard:reference"])
    if reference_graph["sha256"] != golden["graphs"]["reference"]:
        raise AssertionError("unexpected pinned Sopro reference graph digest")
    if reference_shard["sha256"] != golden["shards"]["reference"]:
        raise AssertionError("unexpected pinned Sopro reference shard digest")
    profile["graphs"]["reference"].update(reference_graph)
    profile["shards"]["reference"].update(reference_shard)
    manifest["config"] = actual["config"]
    for name in ("sampleRate", "steps", "tokenSamples", "hopRatio", "positionContext", "revision"):
        manifest[name] = actual[name]
    files["manifest"].write_text(json.dumps(manifest), encoding="utf-8")


def _semantic_test_voice() -> SoproVoiceState:
    return SoproVoiceState(
        cond_vec=np.zeros(512, dtype=np.float32),
        semantic_tokens=np.arange(1, 7, dtype=np.int64),
        mel=np.zeros((100, 1), dtype=np.float32),
        sample_rate=24000,
        level_db=-19.8,
    )


class _FakeSemanticPrefixSession:
    output_names = ("hidden", "bos_hidden")

    def __init__(self, path: Path, **kwargs: Any) -> None:
        self.model = path
        self.kwargs = kwargs
        self.inputs: dict[str, np.ndarray] = {}

    @property
    def input_specs(self) -> tuple[TensorSpec, ...]:
        return (
            TensorSpec("text_ids", "tensor(int64)", (1, "text")),
            TensorSpec("style_tokens", "tensor(int64)", (1, "style")),
            TensorSpec("prompt_tokens", "tensor(int64)", (1, "prompt")),
        )

    def run(self, inputs: dict[str, np.ndarray]) -> list[np.ndarray]:
        self.inputs = {name: value.copy() for name, value in inputs.items()}
        hidden_length = 2 + inputs["text_ids"].shape[1] + inputs["prompt_tokens"].shape[1]
        hidden = np.full((1, hidden_length, 8), 0.25, dtype=np.float32)
        bos_hidden = np.full((1, 1, 8), 0.5, dtype=np.float32)
        return [hidden, bos_hidden]

    def close(self) -> None:
        return None


class _FakeSemanticCoreSession:
    output_names = ("logits", "present_k", "present_v")

    def __init__(self, path: Path, *, mode: str, **kwargs: Any) -> None:
        self.model = path
        self.kwargs = kwargs
        self.mode = mode
        self.calls: list[dict[str, np.ndarray]] = []

    @property
    def input_specs(self) -> tuple[TensorSpec, ...]:
        return (
            TensorSpec("hidden", "tensor(float)", (1, "query", 8)),
            TensorSpec("token", "tensor(int64)", (1, 1)),
            TensorSpec("past_k", "tensor(float)", (2, 1, 1, "past", 4)),
            TensorSpec("past_v", "tensor(float)", (2, 1, 1, "past", 4)),
            TensorSpec("cos", "tensor(float)", ("query", 4)),
            TensorSpec("sin", "tensor(float)", ("query", 4)),
            TensorSpec("attention_bias", "tensor(float)", (1, 1, "query", "present")),
        )

    def run(self, inputs: dict[str, np.ndarray]) -> list[np.ndarray]:
        self.calls.append({name: value.copy() for name, value in inputs.items()})
        call_index = len(self.calls) - 1
        logits = np.full((1, 18), -100.0, dtype=np.float32)
        logits[0, 16] = 200.0
        if self.mode == "minimum-eos":
            logits[0, 3 + call_index] = 5.0
            logits[0, 17] = 100.0
        elif self.mode == "seeded":
            logits[0, 1] = 3.0
            logits[0, 2] = 2.5
            logits[0, 3] = 2.0
        elif self.mode == "bad-logits":
            logits[0, 1] = np.nan
        else:
            logits[0, 5] = 10.0
        present_length = inputs["attention_bias"].shape[-1]
        cache_length = present_length - 1 if self.mode == "bad-cache" else present_length
        cache = np.zeros((2, 1, 1, cache_length, 4), dtype=np.float32)
        return [logits, cache.copy(), cache.copy()]

    def close(self) -> None:
        return None


class _FakeAcousticConditionSession:
    output_names = ("mu",)

    def __init__(self, path: Path, *, mode: str, **kwargs: Any) -> None:
        self.model = path
        self.kwargs = kwargs
        self.mode = mode
        self.calls: list[dict[str, np.ndarray]] = []

    @property
    def input_specs(self) -> tuple[TensorSpec, ...]:
        return (
            TensorSpec("semantic_tokens", "tensor(int64)", (1, "tokens")),
            TensorSpec("frame_to_token", "tensor(int64)", ("frames",)),
        )

    def run(self, inputs: dict[str, np.ndarray]) -> list[np.ndarray]:
        self.calls.append({name: value.copy() for name, value in inputs.items()})
        frames = inputs["frame_to_token"].size
        values = np.broadcast_to(
            np.arange(frames, dtype=np.float32)[None, None, :], (1, 100, frames)
        ).copy()
        if self.mode == "bad-condition":
            values[0, 0, 0] = np.nan
        return [values]

    def close(self) -> None:
        return None


class _FakeAcousticOfflineSession:
    output_names = ("mel",)

    def __init__(self, path: Path, *, mode: str, **kwargs: Any) -> None:
        self.model = path
        self.kwargs = kwargs
        self.mode = mode
        self.calls: list[dict[str, np.ndarray]] = []

    @property
    def input_specs(self) -> tuple[TensorSpec, ...]:
        return (
            TensorSpec("x_init", "tensor(float)", (1, 100, "frames")),
            TensorSpec("mu", "tensor(float)", (1, 100, "frames")),
            TensorSpec("cond_vec", "tensor(float)", (1, 512)),
            TensorSpec("cond_mel", "tensor(float)", (1, 100, "frames")),
            TensorSpec("cond_mask", "tensor(float)", (1, 1, "frames")),
            TensorSpec("cos", "tensor(float)", ("frames", 4)),
            TensorSpec("sin", "tensor(float)", ("frames", 4)),
        )

    def run(self, inputs: dict[str, np.ndarray]) -> list[np.ndarray]:
        self.calls.append({name: value.copy() for name, value in inputs.items()})
        mel = inputs["mu"].copy()
        if self.mode == "bad-offline":
            mel[0, 0, 0] = np.nan
        return [mel]

    def close(self) -> None:
        return None


class _FakeVocoderSession:
    output_names = ("istft_features", "embed_state_out", "conv_state_out", "pending_out")

    def __init__(self, path: Path, *, mode: str, **kwargs: Any) -> None:
        self.model = path
        self.kwargs = kwargs
        self.mode = mode
        self.calls: list[dict[str, np.ndarray]] = []

    @property
    def input_specs(self) -> tuple[TensorSpec, ...]:
        return (
            TensorSpec("is_start", "tensor(bool)", ()),
            TensorSpec("is_flush", "tensor(bool)", ()),
            TensorSpec("mel", "tensor(float)", (1, 100, "frames")),
            TensorSpec("embed_state", "tensor(float)", (1, 100, 6)),
            TensorSpec("conv_state", "tensor(float)", (2, 1, 8, 6)),
            TensorSpec("pending", "tensor(float)", (2, 1, 8, 3)),
        )

    def run(self, inputs: dict[str, np.ndarray]) -> list[np.ndarray]:
        self.calls.append({name: value.copy() for name, value in inputs.items()})
        frames = inputs["mel"].shape[-1] or 3
        features = np.full((1, frames, 1026), -100.0, dtype=np.float32)
        features[:, :, 0] = -2.0
        features[:, :, 513:] = 0.0
        if self.mode == "bad-vocoder":
            features[0, 0, 0] = np.nan
        elif self.mode == "bad-vocoder-rank":
            features = features.reshape(1, frames, 513, 2)
        return [
            features,
            np.zeros((1, 100, 6), dtype=np.float32),
            np.zeros((2, 1, 8, 6), dtype=np.float32),
            np.zeros((2, 1, 8, 3), dtype=np.float32),
        ]

    def close(self) -> None:
        return None


def _install_semantic_fake_sessions(monkeypatch: pytest.MonkeyPatch, mode: str) -> dict[str, Any]:
    from onnxvoice.systems import sopro as sopro_module

    sessions: dict[str, Any] = {}

    def create_session(path: Path, **kwargs: Any) -> Any:
        component = kwargs["component"]
        if component == "semantic_prefix":
            session = _FakeSemanticPrefixSession(path, **kwargs)
        elif component == "semantic_core":
            session = _FakeSemanticCoreSession(path, mode=mode, **kwargs)
        elif component == "acoustic_condition":
            session = _FakeAcousticConditionSession(path, mode=mode, **kwargs)
        elif component == "acoustic_offline_2":
            session = _FakeAcousticOfflineSession(path, mode=mode, **kwargs)
        elif component == "vocoder":
            session = _FakeVocoderSession(path, mode=mode, **kwargs)
        else:
            raise AssertionError(f"unexpected Sopro graph session {component!r}")
        sessions[component] = session
        return session

    monkeypatch.setattr(sopro_module, "OnnxSession", create_session)
    return sessions


def test_semantic_generation_prefills_then_uses_cached_core_and_honors_minimum_eos(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    files, manifest = _write_bundle(tmp_path)
    _configure_semantic_bundle(files, manifest)
    sessions = _install_semantic_fake_sessions(monkeypatch, "minimum-eos")
    runtime = _open_local(files)
    try:
        text = runtime.prepare_text("hello, world!", "en")
        result = runtime._generate_semantic_tokens(
            text,
            _semantic_test_voice(),
            options={
                "temperature": 0.0,
                "top_p": 1.0,
                "top_k": 0,
                "min_seconds": 0.08,
                "max_seconds": 0.2,
            },
        )
        assert result.tokens.tolist() == [3]
        assert result.sampled_steps == 2
        assert result.stopped_on_eos
        assert result.max_steps == 5
        assert result.min_steps == 2
        assert not result.tokens.flags.writeable

        prefix = sessions["semantic_prefix"]
        np.testing.assert_array_equal(prefix.inputs["text_ids"], np.asarray([text.token_ids]))
        np.testing.assert_array_equal(prefix.inputs["style_tokens"], [[1, 2, 3]])
        np.testing.assert_array_equal(prefix.inputs["prompt_tokens"], [[1, 2, 3, 4]])
        core = sessions["semantic_core"]
        assert len(core.calls) == 2
        prefill, step = core.calls
        assert prefill["hidden"].shape == (1, 14, 8)
        assert prefill["token"].dtype == np.int64
        assert prefill["token"].tolist() == [[16]]
        assert prefill["past_k"].shape == (2, 1, 1, 0, 4)
        assert prefill["cos"].shape == (15, 4)
        assert prefill["attention_bias"].shape == (1, 1, 15, 15)
        assert prefill["attention_bias"][0, 0, 0, 1] == -1.0e4
        assert step["hidden"].shape == (1, 0, 8)
        assert step["token"].tolist() == [[3]]
        assert step["past_k"].shape == (2, 1, 1, 15, 4)
        assert step["cos"].shape == (1, 4)
        assert step["attention_bias"].shape == (1, 1, 1, 16)
    finally:
        runtime.close()


def test_semantic_sampling_is_seeded_and_respects_the_duration_step_guard(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    files, manifest = _write_bundle(tmp_path)
    _configure_semantic_bundle(files, manifest)
    sessions = _install_semantic_fake_sessions(monkeypatch, "seeded")
    runtime = _open_local(files)
    try:
        result = runtime._generate_semantic_tokens(
            runtime.prepare_text("hello, world!", "en"),
            _semantic_test_voice(),
            options={
                "temperature": 0.8,
                "top_p": 0.8,
                "top_k": 3,
                "max_seconds": 0.13,
                "min_seconds": 0.0,
                "seed": 123,
            },
        )
        assert result.tokens.tolist() == [1, 2, 1, 2]
        assert result.sampled_steps == 4
        assert result.max_steps == 4
        assert result.min_steps == 1
        assert not result.stopped_on_eos
        assert len(sessions["semantic_core"].calls) == 4
    finally:
        runtime.close()


def test_single_segment_orders_acoustic_condition_ode_and_vocoder_without_application_policy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    files, manifest = _write_bundle(tmp_path)
    _configure_semantic_bundle(files, manifest)
    _add_fake_acoustic_graphs(files, manifest)
    sessions = _install_semantic_fake_sessions(monkeypatch, "pipeline")
    runtime = _open_local(files, providers=["CPUExecutionProvider"])
    try:
        result = runtime._synthesize_one_segment(
            runtime.prepare_text("hello, world!", "en"),
            _semantic_test_voice(),
            options={"temperature": 0.0, "max_seconds": 0.05, "min_seconds": 0.0},
        )
        assert list(sessions) == [
            "semantic_prefix",
            "semantic_core",
            "acoustic_condition",
            "acoustic_offline_2",
            "vocoder",
        ]
        assert result.sample_rate == 24000
        assert result.audio.dtype == np.float32
        assert result.audio.ndim == 1
        assert result.audio.flags.c_contiguous
        assert result.audio.size == 2 * 1024
        assert np.isfinite(result.audio).all()
        assert np.any(result.audio != 0.0)
        assert result.metadata["steps"] == 2
        assert result.metadata["semantic_tokens"] == 2
        assert result.metadata["audio_samples"] == result.audio.size

        condition = sessions["acoustic_condition"].calls[0]
        np.testing.assert_array_equal(condition["semantic_tokens"], [[1, 2, 3, 4, 5, 6, 5, 5]])
        np.testing.assert_array_equal(condition["frame_to_token"], [0, 0, 1, 2, 3, 4, 5, 6, 7])
        acoustic = sessions["acoustic_offline_2"].calls[0]
        assert acoustic["x_init"].shape == (1, 100, 9)
        np.testing.assert_allclose(
            acoustic["x_init"].reshape(-1)[:4],
            [-0.33596167, -0.27266544, 0.63008156, 0.82158428],
            rtol=1.0e-6,
            atol=1.0e-7,
        )
        np.testing.assert_array_equal(acoustic["cond_mask"], [[[1.0] + [0.0] * 8]])
        assert acoustic["cos"].shape == acoustic["sin"].shape == (9, 4)
        vocoder = sessions["vocoder"].calls
        assert len(vocoder) == 2
        assert bool(vocoder[0]["is_start"])
        assert not bool(vocoder[0]["is_flush"])
        assert not bool(vocoder[1]["is_start"])
        assert bool(vocoder[1]["is_flush"])
        assert vocoder[1]["mel"].shape == (1, 100, 0)
        inferred = runtime.infer(
            runtime.prepare_text("hello, world!", "en").token_ids,
            voice_state=_semantic_test_voice(),
            language="en",
            temperature=0.0,
            max_seconds=0.05,
            min_seconds=0.0,
        )
        np.testing.assert_array_equal(inferred.audio, result.audio)
        assert inferred.metadata["language"] == "en"
        assert inferred.metadata["sample_rate"] == 24000
    finally:
        runtime.close()


@pytest.mark.parametrize(
    ("mode", "message"),
    [
        ("bad-condition", "acoustic conditioning output"),
        ("bad-offline", "offline acoustic output"),
        ("bad-vocoder", "ISTFT features"),
        ("bad-vocoder-rank", "ISTFT features"),
    ],
)
def test_single_segment_rejects_invalid_acoustic_and_vocoder_outputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mode: str, message: str
) -> None:
    files, manifest = _write_bundle(tmp_path)
    _configure_semantic_bundle(files, manifest)
    _add_fake_acoustic_graphs(files, manifest)
    _install_semantic_fake_sessions(monkeypatch, mode)
    runtime = _open_local(files)
    try:
        with pytest.raises(RuntimeContractError, match=message):
            runtime._synthesize_one_segment(
                runtime.prepare_text("hello, world!", "en"),
                _semantic_test_voice(),
                options={"temperature": 0.0, "max_seconds": 0.05, "min_seconds": 0.0},
            )
    finally:
        runtime.close()


def test_infer_validates_tokens_options_voice_and_profile(tmp_path: Path) -> None:
    files, manifest = _write_bundle(tmp_path)
    _configure_reference_frontend(files, manifest)
    runtime = _open_local(files, providers=["CPUExecutionProvider"])
    voice = _semantic_test_voice()
    try:
        with pytest.raises(RuntimeContractError, match="prepared SoproVoiceState"):
            runtime.infer([1])
        with pytest.raises(RuntimeContractError, match="Unsupported Sopro inference option"):
            runtime.infer([1], voice_state=voice, unexpected=True)
        with pytest.raises(RuntimeContractError, match="token_ids"):
            runtime.infer(np.asarray([1.0]), voice_state=voice)
        with pytest.raises(RuntimeContractError, match="token_ids"):
            runtime.infer([100000], voice_state=voice)
        with pytest.raises(CapabilityError, match="Unsupported Sopro language"):
            runtime.infer([1], voice_state=voice, language="es")
        runtime.providers = ("CUDAExecutionProvider",)
        with pytest.raises(CapabilityError, match="only for CPUExecutionProvider"):
            runtime.infer([1], voice_state=voice)
        runtime.providers = None
        with pytest.raises(CapabilityError, match="only for CPUExecutionProvider"):
            runtime.infer([1], voice_state=voice)
        runtime.providers = ("CPUExecutionProvider",)
        runtime.profile = "webgpu-fp16"
        with pytest.raises(CapabilityError, match="only parity-gated for wasm-uint8"):
            runtime.infer([1], voice_state=voice)
    finally:
        runtime.close()


@pytest.mark.parametrize(
    ("options", "error"),
    [
        ({"temperature": -0.1}, RuntimeContractError),
        ({"top_p": 0.0}, RuntimeContractError),
        ({"top_k": -1}, RuntimeContractError),
        ({"max_seconds": 0.0}, RuntimeContractError),
        ({"max_seconds": 0.4, "min_seconds": 0.5}, RuntimeContractError),
        ({"max_seconds": 31.0}, RuntimeContractError),
        ({"seed": -1}, RuntimeContractError),
        ({"steps": 1}, CapabilityError),
        ({"application_split": True}, RuntimeContractError),
        ({1: "invalid option key"}, RuntimeContractError),
    ],
)
def test_semantic_generation_rejects_invalid_controls(
    tmp_path: Path, options: dict[str, Any], error: type[Exception]
) -> None:
    files, manifest = _write_bundle(tmp_path)
    _configure_semantic_bundle(files, manifest, include_prefix=False)
    runtime = _open_local(files)
    try:
        with pytest.raises(error):
            runtime._semantic_generation_options(options)
    finally:
        runtime.close()


def test_semantic_generation_requires_prefix_graph_and_validates_outputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    files, manifest = _write_bundle(tmp_path / "missing")
    _configure_semantic_bundle(files, manifest, include_prefix=False)
    runtime = _open_local(files)
    try:
        with pytest.raises(CapabilityError, match="semantic_prefix"):
            runtime._generate_semantic_tokens(
                runtime.prepare_text("hello, world!", "en"), _semantic_test_voice()
            )
    finally:
        runtime.close()

    for mode, match in (("bad-logits", "logits"), ("bad-cache", "present_k")):
        case_root = tmp_path / mode
        files, manifest = _write_bundle(case_root)
        _configure_semantic_bundle(files, manifest)
        _install_semantic_fake_sessions(monkeypatch, mode)
        runtime = _open_local(files)
        try:
            with pytest.raises(RuntimeContractError, match=match):
                runtime._generate_semantic_tokens(
                    runtime.prepare_text("hello, world!", "en"), _semantic_test_voice()
                )
        finally:
            runtime.close()


def test_real_semantic_graphs_match_pinned_token_fixture(tmp_path: Path) -> None:
    source_root = os.environ.get("ONNXVOICE_SOPRO_SEMANTIC_ROOT")
    if not source_root:
        pytest.skip("set ONNXVOICE_SOPRO_SEMANTIC_ROOT to the pinned semantic graph directory")
    source = Path(source_root)
    sources = {
        "reference": source / "reference.onnx",
        "reference_shard": source / "weights-reference.bin",
        "semantic_prefix": source / "semantic_prefix.onnx",
        "semantic_core": source / "semantic_core.onnx",
        "semantic_shard": source / "weights-semantic.bin",
    }
    if not all(path.is_file() for path in sources.values()):
        pytest.skip("the pinned semantic graphs and external-data shards are not available")

    files, manifest = _write_bundle(tmp_path)
    _configure_reference_frontend(files, manifest)
    profile = manifest["profiles"][PROFILE]
    targets = {
        "reference": files["graph:reference"],
        "reference_shard": files["shard:reference"],
        "semantic_prefix": files["graph:semantic_core"].with_name("semantic_prefix.onnx"),
        "semantic_core": files["graph:semantic_core"],
        "semantic_shard": files["shard:semantic"],
    }
    for name, source_path in sources.items():
        shutil.copyfile(source_path, targets[name])

    def digest(path: Path) -> str:
        hasher = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                hasher.update(chunk)
        return hasher.hexdigest()

    golden_path = Path(__file__).parent / "fixtures" / "sopro" / "semantic_golden.json"
    golden = json.loads(golden_path.read_text(encoding="utf-8"))
    expected_hashes = {
        "semantic_prefix": golden["semantic_prefix_sha256"],
        "semantic_core": golden["semantic_core_sha256"],
        "semantic_shard": golden["semantic_weights_sha256"],
    }
    for name, expected_hash in expected_hashes.items():
        assert digest(targets[name]) == expected_hash
    profile["graphs"]["reference"].update(
        bytes=targets["reference"].stat().st_size,
        sha256=digest(targets["reference"]),
    )
    profile["shards"]["reference"].update(
        bytes=targets["reference_shard"].stat().st_size,
        sha256=digest(targets["reference_shard"]),
    )
    profile["graphs"]["semantic_core"].update(
        bytes=targets["semantic_core"].stat().st_size,
        sha256=expected_hashes["semantic_core"],
    )
    profile["graphs"]["semantic_prefix"] = {
        "file": targets["semantic_prefix"].name,
        "bytes": targets["semantic_prefix"].stat().st_size,
        "sha256": expected_hashes["semantic_prefix"],
        "shard": "semantic",
    }
    profile["shards"]["semantic"].update(
        bytes=targets["semantic_shard"].stat().st_size,
        sha256=expected_hashes["semantic_shard"],
    )
    files["graph:semantic_prefix"] = targets["semantic_prefix"]
    files["manifest"].write_text(json.dumps(manifest), encoding="utf-8")

    runtime = _open_local(files, providers=["CPUExecutionProvider"])
    reference_golden = json.loads(
        (Path(__file__).parent / "fixtures" / "sopro" / "reference_golden.json").read_text(
            encoding="utf-8"
        )
    )
    tone = reference_golden["tones"]["steady"]
    indices = np.arange(tone["samples"], dtype=np.float64)
    waveform = (
        0.08 * np.sin(2 * math.pi * 220 * indices / tone["sample_rate"])
        + 0.025 * np.sin(2 * math.pi * 791 * indices / tone["sample_rate"])
    ).astype(np.float32)
    try:
        voice = runtime.prepare_voice(waveform, tone["sample_rate"])
        text = runtime.prepare_text(golden["text"], golden["language"])
        assert list(text.token_ids) == golden["text_ids"]
        result = runtime._generate_semantic_tokens(
            text,
            voice,
            options={
                "temperature": golden["temperature"],
                "top_p": golden["top_p"],
                "top_k": golden["top_k"],
                "max_seconds": golden["max_seconds"],
                "min_seconds": golden["min_seconds"],
                "seed": golden["seed"],
            },
        )
        assert result.tokens.tolist() == golden["tokens"]
        assert result.sampled_steps == golden["sampled_steps"]
        assert not result.stopped_on_eos
    finally:
        runtime.close()


def test_vocoder_istft_matches_official_runtime_fixture() -> None:
    frames = 3
    features = np.empty((frames, 1026), dtype=np.float32)
    for frame in range(frames):
        for frequency in range(513):
            features[frame, frequency] = -2.0 - frequency * 0.002 - frame * 0.01
            features[frame, 513 + frequency] = 0.0001 * frequency + 0.02 * frame
    audio = _sopro_istft(features, frames, n_fft=1024, hop=256)
    assert audio.shape == (512,)
    assert audio.dtype == np.float32
    assert audio.flags.c_contiguous
    positions = [0, 1, 17, 127, 255, 256, 511]
    expected = [
        4.2541108769000857e-7,
        -1.984190930670593e-6,
        -2.150206455553416e-6,
        -2.7349024094291963e-6,
        -1.985644303204026e-6,
        -9.56238977778412e-7,
        2.776663677650504e-6,
    ]
    np.testing.assert_allclose(audio[positions], expected, rtol=1.0e-6, atol=1.0e-12)


def test_real_native_offline_pipeline_matches_pinned_intermediate_fixture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_value = os.environ.get("ONNXVOICE_SOPRO_E2E_ROOT")
    if not source_value:
        pytest.skip(
            "set ONNXVOICE_SOPRO_E2E_ROOT to the pinned model cache containing manifest.json and model/"
        )
    source_root = Path(source_value)
    required = [
        source_root / "manifest.json",
        source_root / "model" / "common" / "reference.onnx",
        source_root / "model" / "common" / "weights-reference.bin",
        source_root / "model" / "common" / "semantic_prefix.onnx",
        source_root / "model" / "common" / "semantic_core.onnx",
        source_root / "model" / "common" / "weights-semantic.bin",
        source_root / "model" / PROFILE / "acoustic_condition.onnx",
        source_root / "model" / PROFILE / "acoustic_offline_2.onnx",
        source_root / "model" / PROFILE / "vocoder.onnx",
        source_root / "model" / PROFILE / "weights-condition.bin",
        source_root / "model" / PROFILE / "weights-acoustic.bin",
        source_root / "model" / PROFILE / "weights-vocoder.bin",
    ]
    if not all(path.is_file() for path in required):
        pytest.skip("the pinned native offline graphs and external-data shards are not available")

    golden_path = Path(__file__).parent / "fixtures" / "sopro" / "offline_pipeline_golden.json"
    golden = json.loads(golden_path.read_text(encoding="utf-8"))
    files, manifest = _write_bundle(tmp_path)
    _configure_reference_frontend(files, manifest)
    _configure_pinned_native_bundle(files, manifest, source_root, golden)
    captured: dict[str, list[dict[str, Any]]] = {}
    original = SoproAdapter._run_contract_graph

    def capture_graph(
        adapter: SoproAdapter,
        graph: str,
        inputs: dict[str, np.ndarray],
        expected_outputs: tuple[str, ...],
    ) -> dict[str, np.ndarray]:
        outputs = original(adapter, graph, inputs, expected_outputs)
        if graph == "acoustic_condition":
            mu = outputs["mu"]
            captured.setdefault(graph, []).append(
                {
                    "shape": list(mu.shape),
                    "samples": [float(mu[0, 0, 0]), float(mu[0, 50, 46]), float(mu[0, 99, 91])],
                    "frame_map": inputs["frame_to_token"].tolist()[:12],
                    "tokens": inputs["semantic_tokens"].reshape(-1).tolist(),
                }
            )
        elif graph == "acoustic_offline_2":
            mel = outputs["mel"]
            captured.setdefault(graph, []).append(
                {
                    "shape": list(mel.shape),
                    "samples": [float(mel[0, 0, 0]), float(mel[0, 50, 45]), float(mel[0, 99, 91])],
                    "x_init": inputs["x_init"].reshape(-1)[:4].tolist(),
                    "cond_mask": inputs["cond_mask"][0, 0, [0, 75, 76, 91]].tolist(),
                }
            )
        elif graph == "vocoder":
            features = outputs["istft_features"]
            captured.setdefault(graph, []).append(
                {
                    "shape": list(features.shape),
                    "samples": features.reshape(-1)[[0, 1, 512, 513, 1025]].tolist(),
                }
            )
        return outputs

    monkeypatch.setattr(SoproAdapter, "_run_contract_graph", capture_graph)
    runtime = _open_local(files, providers=["CPUExecutionProvider"])
    reference_golden = json.loads(
        (Path(__file__).parent / "fixtures" / "sopro" / "reference_golden.json").read_text(
            encoding="utf-8"
        )
    )
    tone = reference_golden["tones"]["steady"]
    indices = np.arange(tone["samples"], dtype=np.float64)
    waveform = (
        0.08 * np.sin(2 * math.pi * 220 * indices / tone["sample_rate"])
        + 0.025 * np.sin(2 * math.pi * 791 * indices / tone["sample_rate"])
    ).astype(np.float32)
    try:
        voice = runtime.prepare_voice(waveform, tone["sample_rate"])
        text = runtime.prepare_text(golden["text"], golden["language"])
        assert list(text.token_ids) == golden["text_ids"]
        result = runtime.infer(
            text.token_ids,
            voice_state=voice,
            language=golden["language"],
            temperature=0.0,
            top_p=1.0,
            top_k=0,
            max_seconds=golden["max_seconds"],
            min_seconds=golden["min_seconds"],
            seed=golden["seed"],
        )
        assert captured["acoustic_condition"][0]["shape"] == [1, 100, 92]
        np.testing.assert_array_equal(
            captured["acoustic_condition"][0]["tokens"][-4:], golden["semantic_tokens"]
        )
        np.testing.assert_array_equal(
            captured["acoustic_condition"][0]["frame_map"], [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2]
        )
        np.testing.assert_allclose(
            captured["acoustic_condition"][0]["samples"],
            golden["condition_mu_samples"],
            rtol=1.0e-5,
            atol=1.0e-5,
        )
        assert captured["acoustic_offline_2"][0]["shape"] == [1, 100, 92]
        np.testing.assert_allclose(
            captured["acoustic_offline_2"][0]["x_init"],
            golden["noise_samples"],
            rtol=1.0e-6,
            atol=1.0e-7,
        )
        np.testing.assert_array_equal(
            captured["acoustic_offline_2"][0]["cond_mask"], [1.0, 1.0, 0.0, 0.0]
        )
        np.testing.assert_allclose(
            captured["acoustic_offline_2"][0]["samples"],
            golden["acoustic_mel_samples"],
            rtol=1.0e-5,
            atol=1.0e-5,
        )
        assert [item["shape"] for item in captured["vocoder"]] == [[1, 21, 1026], [1, 27, 1026]]
        np.testing.assert_allclose(
            [item["samples"] for item in captured["vocoder"]],
            golden["vocoder_feature_samples"],
            rtol=1.0e-5,
            atol=1.0e-5,
        )
        assert result.audio.shape == (golden["audio_length"],)
        assert result.audio.dtype == np.float32
        assert result.audio.flags.c_contiguous
        assert result.sample_rate == 24000
        assert np.isfinite(result.audio).all()
        np.testing.assert_allclose(
            [float(result.audio.min()), float(result.audio.max())],
            [golden["audio_min"], golden["audio_max"]],
            rtol=1.0e-5,
            atol=1.0e-5,
        )
        positions = [0, 1, 2, 17, 255, 1024, 2048, 3839]
        np.testing.assert_allclose(
            result.audio[positions], golden["audio_samples"], rtol=1.0e-5, atol=1.0e-5
        )
        assert result.metadata["steps"] == 2
        assert result.metadata["semantic_tokens"] == 4
        assert result.metadata["sample_rate"] == 24000
    finally:
        runtime.close()
