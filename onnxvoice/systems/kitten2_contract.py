from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..checksums import digest_file
from ..errors import RuntimeContractError
from ..types import validate_relative_path, validate_safe_component

_LAYOUT = "kitten2-onnx-bundle-v1"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_TOKEN_MAP_FIELDS = (
    "audio_id_base",
    "num_audio_tokens",
    "speech_start_id",
    "speech_end_id",
    "text_start_id",
    "text_end_id",
    "start_id",
    "stop_id",
    "final_seg_id",
    "reference_speech_start_id",
    "reference_speech_end_id",
    "reference_text_start_id",
    "reference_text_end_id",
)


@dataclass(frozen=True, slots=True)
class Kitten2Bundle:
    root: Path
    manifest: dict[str, Any]
    files: dict[str, str]


def _invalid(message: str) -> RuntimeContractError:
    return RuntimeContractError(f"Invalid Kitten2 bundle: {message}")


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise _invalid(f"{label} must be an object")
    return value


def _relative(value: Any, label: str) -> str:
    if not isinstance(value, str):
        raise _invalid(f"{label} must be a relative path string")
    try:
        return validate_relative_path(value, field_name=label)
    except ValueError as exc:
        raise _invalid(str(exc)) from exc


def _referenced_file(value: Any, label: str, files: Mapping[str, str]) -> str:
    relative = _relative(value, label)
    if relative not in files:
        raise _invalid(f"{label} is not declared in bundle.json files: {relative}")
    return relative


def _positive_int(value: Any, label: str) -> int:
    if type(value) is not int or value < 1:
        raise _invalid(f"{label} must be a positive integer")
    return value


def _validate_manifest(
    root: Path, manifest: Any, *, ignored_files: frozenset[str]
) -> Kitten2Bundle:
    data = _mapping(manifest, "bundle.json")
    if data.get("format") != _LAYOUT:
        raise _invalid(f"format must be {_LAYOUT!r}")
    if type(data.get("sample_rate")) is not int or data["sample_rate"] != 24000:
        raise _invalid("sample_rate must be 24000")

    raw_files = _mapping(data.get("files"), "files")
    if not raw_files:
        raise _invalid("files must be a non-empty path-to-SHA256 object")
    files: dict[str, str] = {}
    for raw_path, sha256 in raw_files.items():
        relative = _relative(raw_path, "bundle file path")
        if relative == "bundle.json":
            raise _invalid("files must not include bundle.json")
        if not isinstance(sha256, str) or _SHA256_RE.fullmatch(sha256) is None:
            raise _invalid(f"files[{relative!r}] must be a lowercase SHA-256 digest")
        files[relative] = sha256

    lm = _mapping(data.get("lm"), "lm")
    speaker = _mapping(data.get("speaker"), "speaker")
    decoder = _mapping(data.get("decoder"), "decoder")
    file_fields = (
        (lm, "model", "lm.model"),
        (lm, "metadata", "lm.metadata"),
        (lm, "embeddings", "lm.embeddings"),
        (speaker, "model", "speaker.model"),
        (speaker, "metadata", "speaker.metadata"),
        (decoder, "flow", "decoder.flow"),
        (decoder, "flow_metadata", "decoder.flow_metadata"),
        (decoder, "mel2wav", "decoder.mel2wav"),
        (decoder, "mel2wav_metadata", "decoder.mel2wav_metadata"),
    )
    referenced = {
        _referenced_file(section.get(key), label, files)
        for section, key, label in file_fields
    }

    tokenizer = _relative(data.get("tokenizer"), "tokenizer")
    raw_tokenizer_assets = data.get("tokenizer_assets")
    if not isinstance(raw_tokenizer_assets, list) or not raw_tokenizer_assets:
        raise _invalid("tokenizer_assets must be a non-empty list")
    for name in raw_tokenizer_assets:
        asset_path = _relative(name, "tokenizer asset")
        relative = f"{tokenizer}/{asset_path}"
        if relative not in files:
            raise _invalid(f"tokenizer asset is not declared in bundle.json files: {relative}")
        referenced.add(relative)

    raw_voices = _mapping(data.get("voices"), "voices")
    if not raw_voices:
        raise _invalid("voices must be a non-empty object")
    voices: dict[str, str] = {}
    for name, raw_path in raw_voices.items():
        try:
            validate_safe_component(name, field_name="voice name")
        except ValueError as exc:
            raise _invalid(str(exc)) from exc
        voice_path = _referenced_file(raw_path, f"voices.{name}", files)
        metadata_path = str(Path(voice_path).with_suffix(".meta.json").as_posix())
        if metadata_path not in files:
            raise _invalid(f"voice metadata is not declared in bundle.json files: {metadata_path}")
        voices[name] = voice_path
        referenced.update((voice_path, metadata_path))

    default_voice = data.get("default_voice")
    if not isinstance(default_voice, str) or default_voice not in voices:
        raise _invalid("default_voice must name a voice declared in voices")

    raw_presets = _mapping(data.get("decode_presets"), "decode_presets")
    if not {"stable", "expressive"}.issubset(raw_presets):
        raise _invalid("decode_presets must declare stable and expressive")
    for name, raw_preset in raw_presets.items():
        try:
            validate_safe_component(name, field_name="decode preset name")
        except ValueError as exc:
            raise _invalid(str(exc)) from exc
        preset = _mapping(raw_preset, f"decode_presets.{name}")
        for key in ("temperature", "top_p", "top_k", "min_p"):
            if key not in preset:
                raise _invalid(f"decode_presets.{name}.{key} is required")
        temperature = preset["temperature"]
        if (
            isinstance(temperature, bool)
            or not isinstance(temperature, (int, float))
            or not math.isfinite(float(temperature))
            or temperature <= 0
        ):
            raise _invalid(f"decode_presets.{name}.temperature must be finite and positive")
        for key in ("top_p", "min_p"):
            value = preset[key]
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or not 0 <= value <= 1
            ):
                raise _invalid(f"decode_presets.{name}.{key} must be between 0 and 1")
        if type(preset["top_k"]) is not int or preset["top_k"] < 0:
            raise _invalid(f"decode_presets.{name}.top_k must be a non-negative integer")
    default_preset = data.get("default_preset")
    if not isinstance(default_preset, str) or default_preset not in raw_presets:
        raise _invalid("default_preset must name a declared decode preset")

    token_map = _mapping(data.get("token_map"), "token_map")
    unknown_token_fields = set(token_map) - set(_TOKEN_MAP_FIELDS)
    if unknown_token_fields:
        raise _invalid(f"unknown token_map fields: {sorted(unknown_token_fields)}")
    token_ids: list[int] = []
    for field in _TOKEN_MAP_FIELDS:
        value = token_map.get(field)
        if type(value) is not int or value < 0:
            raise _invalid(f"token_map.{field} must be a non-negative integer")
        token_ids.append(value)
    if len(set(token_ids)) != len(token_ids):
        raise _invalid("token_map IDs must be distinct")
    if token_map["num_audio_tokens"] < 1:
        raise _invalid("token_map.num_audio_tokens must be positive")
    if token_map["final_seg_id"] != token_map["audio_id_base"] + token_map["num_audio_tokens"]:
        raise _invalid("token_map.final_seg_id must equal audio_id_base + num_audio_tokens")

    generation = _mapping(data.get("generation"), "generation")
    if "use_reference_prompt" in generation and type(generation["use_reference_prompt"]) is not bool:
        raise _invalid("generation.use_reference_prompt must be a boolean")
    for key in ("repetition_window", "chunk_chars", "chunk_min_chars"):
        if key in generation:
            _positive_int(generation[key], f"generation.{key}")
    if "emotion_control" in generation and not isinstance(generation["emotion_control"], str):
        raise _invalid("generation.emotion_control must be a string")


    actual_files: set[str] = set()
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        resolved = path.resolve()
        if not resolved.is_relative_to(root):
            raise _invalid(f"bundle file escapes its root: {path.relative_to(root).as_posix()}")
        relative = path.relative_to(root).as_posix()
        if relative != "bundle.json" and relative not in ignored_files:
            actual_files.add(relative)
    if actual_files != set(files):
        missing = sorted(set(files) - actual_files)
        extra = sorted(actual_files - set(files))
        raise _invalid(f"file inventory mismatch; missing={missing}, undeclared={extra}")

    graph_directories = {
        Path(relative).parent.as_posix()
        for relative in files
        if relative.endswith(".onnx")
    }
    for relative in files:
        if (
            relative.endswith(".onnx.data")
            and Path(relative).parent.as_posix() not in graph_directories
        ):
            raise _invalid(f"external ONNX data must be adjacent to a graph: {relative}")
    for relative, expected_sha256 in files.items():
        path = (root / relative).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise _invalid(f"bundle file is missing or outside its root: {relative}")
        digest = digest_file(path)
        if digest != expected_sha256:
            raise _invalid(f"SHA-256 mismatch for {relative}")

    metadata_paths = (
        lm["metadata"],
        speaker["metadata"],
        decoder["flow_metadata"],
        decoder["mel2wav_metadata"],
        *(str(Path(path).with_suffix(".meta.json").as_posix()) for path in voices.values()),
    )
    for metadata_path in metadata_paths:
        try:
            value = json.loads((root / metadata_path).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise _invalid(f"cannot read metadata file {metadata_path}: {exc}") from exc
        if not isinstance(value, Mapping):
            raise _invalid(f"metadata file {metadata_path} must contain a JSON object")
    return Kitten2Bundle(root=root, manifest=dict(data), files=files)


def load_kitten2_bundle(
    bundle_path: str | Path, *, ignored_files: frozenset[str] = frozenset()
) -> Kitten2Bundle:
    path = Path(bundle_path).expanduser().resolve()
    if path.name != "bundle.json" or not path.is_file():
        raise _invalid("bundle.json is required")
    root = path.parent
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise _invalid(f"cannot read bundle.json: {exc}") from exc
    return _validate_manifest(root, manifest, ignored_files=ignored_files)
