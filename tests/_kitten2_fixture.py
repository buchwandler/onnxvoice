from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def create_bundle(root: Path) -> Path:
    payloads = {
        "lm/lm.onnx": b"lm graph",
        "lm/lm.meta.json": b'{"component":"lm","runtime_profile":"test","io_dtype":"float32","input_mode":"inputs_embeds","embedding_storage_dtype":"float32","runtime_memory_profile":"default","runtime_provider":"CPUExecutionProvider","num_hidden_layers":1,"num_key_value_heads":1,"head_dim":1,"hidden_size":4}',
        "lm/embed_tokens.bf16.npy": b"embeddings",
        "lm/lm.ternary-exact-int8.onnx.data": b"lm data",
        "speaker/speaker.onnx": b"speaker graph",
        "speaker/speaker.meta.json": b'{"component":"speaker"}',
        "decoder/flow/flow.onnx": b"flow graph",
        "decoder/flow/flow.meta.json": b'{"component":"flow"}',
        "decoder/mel2wav/mel2wav.onnx": b"mel2wav graph",
        "decoder/mel2wav/mel2wav.onnx.data": b"mel2wav data",
        "decoder/mel2wav/mel2wav.meta.json": b'{"component":"mel2wav"}',
        "tokenizer/tokenizer.json": b"{}",
        "tokenizer/vocab.json": b"{}",
        "voices/Bruno.npz": b"voice data",
        "voices/Bruno.meta.json": b'{"reference_transcript":"hello"}',
    }
    for relative, payload in payloads.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)

    manifest: dict[str, Any] = {
        "format": "kitten2-onnx-bundle-v1",
        "sample_rate": 24000,
        "default_voice": "Bruno",
        "default_preset": "stable",
        "lm": {
            "model": "lm/lm.onnx",
            "metadata": "lm/lm.meta.json",
            "embeddings": "lm/embed_tokens.bf16.npy",
        },
        "speaker": {"model": "speaker/speaker.onnx", "metadata": "speaker/speaker.meta.json"},
        "decoder": {
            "flow": "decoder/flow/flow.onnx",
            "flow_metadata": "decoder/flow/flow.meta.json",
            "mel2wav": "decoder/mel2wav/mel2wav.onnx",
            "mel2wav_metadata": "decoder/mel2wav/mel2wav.meta.json",
        },
        "tokenizer": "tokenizer",
        "tokenizer_assets": ["tokenizer.json", "vocab.json"],
        "voices": {"Bruno": "voices/Bruno.npz"},
        "token_map": {
            "audio_id_base": 100,
            "num_audio_tokens": 10,
            "speech_start_id": 1,
            "speech_end_id": 2,
            "text_start_id": 3,
            "text_end_id": 4,
            "start_id": 5,
            "stop_id": 6,
            "final_seg_id": 110,
            "reference_speech_start_id": 11,
            "reference_speech_end_id": 12,
            "reference_text_start_id": 13,
            "reference_text_end_id": 14,
        },
        "decode_presets": {
            "stable": {"temperature": 0.8, "top_p": 0.8, "top_k": 50, "min_p": 0.0},
            "expressive": {"temperature": 0.9, "top_p": 0.9, "top_k": 50, "min_p": 0.0},
        },
        "generation": {
            "use_reference_prompt": True,
            "repetition_window": 50,
            "emotion_control": "{emo: 1}",
            "chunk_chars": 380,
            "chunk_min_chars": 130,
        },
        "files": {
            relative: hashlib.sha256(payload).hexdigest() for relative, payload in payloads.items()
        },
    }
    bundle_path = root / "bundle.json"
    bundle_path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")
    return bundle_path
