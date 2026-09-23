"""Pocket ONNX bundle adapter with explicit v2 graph contracts."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NoReturn

import numpy as np
from platformdirs import user_cache_path

from ..checksums import digest_file, verify_file
from ..errors import (
    AssetAccessError,
    AssetDownloadError,
    AssetNotFoundError,
    IntegrityError,
    OfflineError,
    OptionalDependencyError,
    PredefinedVoiceAccessError,
    PredefinedVoiceIntegrityError,
    PredefinedVoiceNotFoundError,
    RuntimeContractError,
)
from ..huggingface import HuggingFaceSource, download_huggingface_file
from ..runtime import OnnxSession
from ..store import FileLock
from ..types import InferenceResult, Installation, validate_safe_component
from .base import SystemAdapter

POCKET_REQUIRED_METADATA_FIELDS = frozenset(
    {
        "schema_version",
        "sample_rate",
        "samples_per_frame",
        "latent_dim",
        "conditioning_dim",
        "flow_lm_state_manifest",
        "mimi_state_manifest",
        "insert_bos_before_voice",
        "bos_before_voice_file",
    }
)
REQUIRED_FLOW_STATE_FIELDS = ("flow_lm_state_manifest", "mimi_state_manifest")
_FLOW_MAIN_STATE_ROLES = {"flow_lm_main", "mimi_decoder"}
_SUPPORTED_DTYPES = {
    "bool",
    "float16",
    "float32",
    "float64",
    "int8",
    "int16",
    "int32",
    "int64",
    "uint8",
}

PREDEFINED_VOICE_REPOSITORY = "kyutai/pocket-tts"
PREDEFINED_VOICE_REVISION = "d18466fc6d3ad070afee1bde2f0db8992007c48f"


def _load_safetensors(path: Path) -> Mapping[str, np.ndarray]:
    try:
        from safetensors.numpy import load_file
    except ImportError as exc:
        raise OptionalDependencyError(
            "Loading predefined Pocket voices requires safetensors. Install 'onnxvoice[pocket]'."
        ) from exc
    return load_file(str(path))


@dataclass(frozen=True, slots=True)
class StateSpec:
    """One recurrent state tensor declared by a Pocket bundle manifest."""

    index: int
    input_name: str
    output_name: str
    shape: tuple[int, ...]
    dtype: np.dtype
    fill: str
    module: str | None = None
    key: str | None = None


def parse_state_manifest(raw: Any, *, name: str) -> tuple[StateSpec, ...]:
    """Parse the v2 list-form state manifest into typed state specifications."""
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes, Mapping)):
        raise RuntimeContractError(f"{name} state manifest must be a sequence")
    specs: list[StateSpec] = []
    indexes: set[int] = set()
    input_names: set[str] = set()
    output_names: set[str] = set()
    for position, entry in enumerate(raw):
        if not isinstance(entry, Mapping):
            raise RuntimeContractError(f"{name} state manifest entry {position} must be an object")
        index = entry.get("index")
        if not isinstance(index, int) or isinstance(index, bool) or index < 0:
            raise RuntimeContractError(f"{name} state entry {position} has invalid index")
        if index in indexes:
            raise RuntimeContractError(f"{name} state manifest has duplicate index {index}")
        input_name = entry.get("input_name")
        output_name = entry.get("output_name")
        if not isinstance(input_name, str) or not input_name:
            raise RuntimeContractError(f"{name} state entry {index} has invalid input_name")
        if not isinstance(output_name, str) or not output_name:
            raise RuntimeContractError(f"{name} state entry {index} has invalid output_name")
        if input_name in input_names:
            raise RuntimeContractError(
                f"{name} state manifest has duplicate input_name {input_name!r}"
            )
        if output_name in output_names:
            raise RuntimeContractError(
                f"{name} state manifest has duplicate output_name {output_name!r}"
            )
        shape = entry.get("shape")
        if not isinstance(shape, list) or any(
            not isinstance(dimension, int) or isinstance(dimension, bool) or dimension < 0
            for dimension in shape
        ):
            raise RuntimeContractError(f"{name} state entry {index} has invalid shape")
        dtype_name = entry.get("dtype")
        if not isinstance(dtype_name, str) or dtype_name not in _SUPPORTED_DTYPES:
            raise RuntimeContractError(
                f"{name} state entry {index} has unsupported dtype {dtype_name!r}"
            )
        fill = entry.get("fill")
        if fill not in {"zeros", "ones", "nan", "empty"}:
            raise RuntimeContractError(f"{name} state entry {index} has unsupported fill {fill!r}")
        dtype = np.dtype(dtype_name)
        if fill == "nan" and not np.issubdtype(dtype, np.floating):
            raise RuntimeContractError(
                f"{name} state entry {index} uses nan with non-floating dtype"
            )
        module = entry.get("module")
        key = entry.get("key")
        if (module is None) != (key is None):
            raise RuntimeContractError(
                f"{name} state entry {index} must declare module and key together"
            )
        if module is not None and (not isinstance(module, str) or not module):
            raise RuntimeContractError(f"{name} state entry {index} has invalid module")
        if key is not None and (not isinstance(key, str) or not key):
            raise RuntimeContractError(f"{name} state entry {index} has invalid key")
        indexes.add(index)
        input_names.add(input_name)
        output_names.add(output_name)
        specs.append(
            StateSpec(index, input_name, output_name, tuple(shape), dtype, fill, module, key)
        )
    specs.sort(key=lambda spec: spec.index)
    if [spec.index for spec in specs] != list(range(len(specs))):
        raise RuntimeContractError(f"{name} state indexes must be contiguous and start at zero")
    return tuple(specs)


def initialize_state(specs: Sequence[StateSpec]) -> dict[str, np.ndarray]:
    """Create state inputs using each manifest entry's declared fill semantics."""
    state: dict[str, np.ndarray] = {}
    for spec in specs:
        if spec.fill == "zeros":
            value = np.zeros(spec.shape, dtype=spec.dtype)
        elif spec.fill == "ones":
            value = np.ones(spec.shape, dtype=spec.dtype)
        elif spec.fill == "nan":
            value = np.full(spec.shape, np.nan, dtype=spec.dtype)
        else:
            value = np.empty(spec.shape, dtype=spec.dtype)
        state[spec.input_name] = value
    return state


def _derive_step(module_state: Mapping[str, Any]) -> np.ndarray:
    if "step" in module_state:
        return np.asarray(module_state["step"], dtype=np.int64).reshape(1)
    if "offset" in module_state and "end_offset" not in module_state:
        return np.asarray(module_state["offset"], dtype=np.int64).reshape(1)
    if "current_end" in module_state:
        return np.asarray([np.asarray(module_state["current_end"]).shape[0]], dtype=np.int64)
    return np.asarray([0], dtype=np.int64)


def _adapt_state_tensor(tensor: Any, spec: StateSpec) -> np.ndarray:
    """Fit an imported tensor to a manifest's fixed state input shape."""
    source = np.asarray(tensor, dtype=spec.dtype)
    target = initialize_state((spec,))[spec.input_name]
    if source.shape == spec.shape:
        return source.copy()
    if source.size == np.prod(spec.shape, dtype=np.int64):
        return source.reshape(spec.shape).copy()
    if source.ndim != len(spec.shape):
        return target
    slices = tuple(
        slice(0, min(source_dim, target_dim))
        for source_dim, target_dim in zip(source.shape, spec.shape, strict=True)
    )
    if any(selection.stop == 0 for selection in slices):
        return target
    adapted = target.copy()
    adapted[slices] = source[slices]
    return adapted


def _flow_state_from_model_state(
    model_state: Mapping[str, Mapping[str, Any]],
    specs: Sequence[StateSpec],
    *,
    name: str,
) -> dict[str, np.ndarray]:
    if not specs:
        raise PredefinedVoiceIntegrityError(
            f"Pocket Flow-LM manifest cannot import predefined voice {name!r}: "
            "the manifest is empty"
        )
    state = initialize_state(specs)
    for spec in specs:
        if spec.module is None or spec.key is None:
            raise PredefinedVoiceIntegrityError(
                f"Pocket Flow-LM manifest cannot import predefined voice {name!r}: "
                f"state {spec.input_name!r} has no module/key mapping"
            )
        module_state = model_state.get(spec.module, {})
        tensor = module_state.get(spec.key)
        try:
            if tensor is None and spec.key == "step":
                tensor = _derive_step(module_state)
            if tensor is not None:
                state[spec.input_name] = _adapt_state_tensor(tensor, spec)
        except (TypeError, ValueError, OverflowError) as exc:
            raise PredefinedVoiceIntegrityError(
                f"Predefined Pocket Flow-LM tensor {spec.module}/{spec.key} "
                "could not be adapted to the bundle state manifest"
            ) from exc
    return state


def update_state_from_named_outputs(
    state: dict[str, np.ndarray],
    output_by_name: Mapping[str, Any],
    specs: Sequence[StateSpec],
) -> None:
    """Update recurrent state by manifest output names, never positional offsets."""
    for spec in specs:
        if spec.output_name not in output_by_name:
            raise RuntimeContractError(f"Missing state output {spec.output_name!r}")
        state[spec.input_name] = np.asarray(output_by_name[spec.output_name])


@dataclass(frozen=True, slots=True, init=False)
class PocketVoiceState:
    """Reusable Pocket conditioning, either embeddings or Flow-LM state."""

    embeddings: np.ndarray | None
    flow_state: Mapping[str, np.ndarray] | None
    sample_rate: int
    metadata: Mapping[str, Any]
    values: Mapping[str, np.ndarray]

    def __init__(
        self,
        embeddings: np.ndarray | None = None,
        sample_rate: int = 24000,
        metadata: Mapping[str, Any] | None = None,
        *,
        flow_state: Mapping[str, np.ndarray] | None = None,
        values: Mapping[str, np.ndarray] | None = None,
    ) -> None:
        legacy_values = dict(values or {})
        if embeddings is not None and flow_state is not None:
            raise ValueError("PocketVoiceState accepts embeddings or Flow-LM state, not both")
        if flow_state is not None and legacy_values:
            raise ValueError("PocketVoiceState values cannot be combined with Flow-LM state")
        if embeddings is None and flow_state is None:
            if not legacy_values:
                raise TypeError("PocketVoiceState requires embeddings or Flow-LM state")
            embeddings = np.asarray(next(iter(legacy_values.values())), dtype=np.float32)

        metadata_values = dict(metadata or {})
        if embeddings is not None:
            normalized_embeddings = np.asarray(embeddings, dtype=np.float32)
            normalized_flow_state = None
            compatible_values = legacy_values or {"embeddings": normalized_embeddings}
        else:
            if not isinstance(flow_state, Mapping):
                raise TypeError("PocketVoiceState Flow-LM state must be a mapping")
            normalized_embeddings = None
            normalized_flow_state = {name: np.asarray(value) for name, value in flow_state.items()}
            compatible_values = {}

        object.__setattr__(self, "embeddings", normalized_embeddings)
        object.__setattr__(self, "flow_state", normalized_flow_state)
        object.__setattr__(
            self, "sample_rate", int(metadata_values.get("sample_rate", sample_rate))
        )
        object.__setattr__(self, "metadata", metadata_values)
        object.__setattr__(self, "values", compatible_values)


class PocketAdapter(SystemAdapter):
    """Pocket ONNX bundle adapter with lazy, contract-validated sessions."""

    system = "pocket"
    DEFAULT_MAX_FRAMES = 200

    def __init__(
        self,
        installation: Installation,
        *,
        voice_cache_dir: str | Path | None = None,
        offline: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(installation, **kwargs)
        self._sessions: dict[str, OnnxSession] = {}
        self._bundle_metadata: dict[str, Any] | None = None
        self._bos_conditioning: np.ndarray | None = None
        self._flow_specs: tuple[StateSpec, ...] = ()
        self._mimi_specs: tuple[StateSpec, ...] = ()
        self._validated_contracts: set[str] = set()
        self._validated = False
        self._predefined_voice_states: dict[
            tuple[str, str, str, str, str, str, str], PocketVoiceState
        ] = {}
        self._voice_cache_dir = Path(
            voice_cache_dir
            if voice_cache_dir is not None
            else Path(user_cache_path("onnxvoice")) / "pocket-voice-states"
        )
        self._offline = offline

    def _validate_bundle_metadata(self) -> dict[str, Any]:
        if self._bundle_metadata is not None:
            return self._bundle_metadata
        try:
            metadata_path = self.installation.artifact("bundle_metadata").path
        except KeyError as err:
            runtime_metadata = self.installation.metadata.get("runtime") or {}
            if not runtime_metadata:
                raise RuntimeContractError(
                    "Pocket bundle metadata not found. Provide bundle_metadata artifact or runtime metadata."
                ) from err
            self._bundle_metadata = dict(runtime_metadata)
            return self._bundle_metadata
        try:
            bundle_data = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeContractError(
                f"Could not read Pocket bundle metadata {metadata_path}: {exc}"
            ) from exc
        if not isinstance(bundle_data, dict):
            raise RuntimeContractError("Pocket bundle metadata must be a JSON object")
        self._bundle_metadata = {**bundle_data, **self.installation.metadata}
        return self._bundle_metadata

    @staticmethod
    def _parse_manifest(metadata: Mapping[str, Any], key: str, name: str) -> tuple[StateSpec, ...]:
        raw = metadata.get(key)
        if (
            raw in (None, {}, "")
            or isinstance(raw, (Mapping, str, bytes))
            or not isinstance(raw, Sequence)
        ):
            # Accept legacy runtime metadata only at the adapter boundary; the public parser is strict.
            return ()
        return parse_state_manifest(raw, name=name)

    def _ensure_validated(self) -> dict[str, Any]:
        if self._validated:
            assert self._bundle_metadata is not None
            return self._bundle_metadata
        metadata = self._validate_bundle_metadata()
        missing = sorted(
            field for field in POCKET_REQUIRED_METADATA_FIELDS if field not in metadata
        )
        if missing:
            raise RuntimeContractError(
                f"Pocket bundle metadata is missing required fields: {', '.join(missing)}"
            )
        try:
            self._flow_specs = self._parse_manifest(metadata, "flow_lm_state_manifest", "flow_lm")
            self._mimi_specs = self._parse_manifest(metadata, "mimi_state_manifest", "mimi")
        except RuntimeContractError:
            raise
        self._validated = True
        return metadata

    @staticmethod
    def _predefined_voice_names(metadata: Mapping[str, Any]) -> tuple[str, ...]:
        def parse(raw: Any, field_name: str) -> tuple[str, ...] | None:
            if raw is None:
                return None
            if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes, Mapping)):
                raise RuntimeContractError(f"Pocket {field_name} must be a sequence")
            names: list[str] = []
            for name in raw:
                if not isinstance(name, str):
                    raise RuntimeContractError(f"Pocket {field_name} must contain strings")
                try:
                    validate_safe_component(name, field_name="predefined voice name")
                except ValueError as exc:
                    raise RuntimeContractError(str(exc)) from exc
                if name in names:
                    raise RuntimeContractError(f"Pocket {field_name} contains duplicate {name!r}")
                names.append(name)
            return tuple(names)

        bundle_names = parse(metadata.get("predefined_voices"), "predefined_voices")
        catalog_names = parse(metadata.get("predefined_voice_names"), "predefined_voice_names")
        if (
            bundle_names is not None
            and catalog_names is not None
            and set(bundle_names) != set(catalog_names)
        ):
            raise RuntimeContractError(
                "Pocket bundle metadata and catalog disagree about predefined voice names"
            )
        return bundle_names if bundle_names is not None else catalog_names or ()

    @staticmethod
    def _predefined_voice_source(
        metadata: Mapping[str, Any], name: str, bundle_name: str
    ) -> HuggingFaceSource:
        records = metadata.get("voice_states")
        if records is not None:
            if not isinstance(records, Sequence) or isinstance(records, (str, bytes, Mapping)):
                raise RuntimeContractError("Pocket voice_states metadata must be a sequence")
            for record in records:
                if not isinstance(record, Mapping):
                    raise RuntimeContractError("Pocket voice-state metadata must contain objects")
                if record.get("name") != name:
                    continue
                source = record.get("source")
                access = record.get("access")
                if not isinstance(source, Mapping) or not isinstance(access, Mapping):
                    raise RuntimeContractError(
                        f"Pocket predefined voice {name!r} has invalid source metadata"
                    )
                repository = source.get("repository")
                revision = source.get("revision")
                path = source.get("path")
                gated = access.get("gated")
                if (
                    source.get("provider") != "huggingface"
                    or not isinstance(repository, str)
                    or not isinstance(revision, str)
                    or re.fullmatch(r"[0-9a-f]{40}", revision) is None
                    or not isinstance(path, str)
                    or not isinstance(gated, bool)
                ):
                    raise RuntimeContractError(
                        f"Pocket predefined voice {name!r} has invalid Hugging Face source metadata"
                    )
                try:
                    return HuggingFaceSource(repository, revision, path, gated=gated)
                except ValueError as exc:
                    raise RuntimeContractError(str(exc)) from exc
        return HuggingFaceSource(
            PREDEFINED_VOICE_REPOSITORY,
            PREDEFINED_VOICE_REVISION,
            f"languages/{bundle_name}/embeddings/{name}.safetensors",
            gated=True,
        )

    @property
    def predefined_voices(self) -> tuple[str, ...]:
        return self._predefined_voice_names(self._ensure_validated())

    @staticmethod
    def _names(session: Any, attribute: str) -> tuple[str, ...]:
        value = getattr(session, attribute, ())
        if isinstance(value, (tuple, list)):
            return tuple(str(name) for name in value)
        return ()

    @staticmethod
    def _input_ranks(session: Any) -> dict[str, int]:
        specs = getattr(session, "input_specs", ())
        if not isinstance(specs, (tuple, list)):
            return {}
        return {spec.name: len(spec.shape) for spec in specs}

    @staticmethod
    def _fit_rank(value: Any, rank: int) -> np.ndarray:
        result = np.asarray(value, dtype=np.float32)
        while result.ndim < rank:
            result = np.expand_dims(result, axis=1)
        while result.ndim > rank:
            if result.shape[1] != 1:
                raise RuntimeContractError(
                    f"Cannot fit tensor shape {result.shape} to rank {rank}"
                )
            result = np.squeeze(result, axis=1)
        return result

    @staticmethod
    def _named_outputs(session: Any, outputs: Sequence[Any]) -> dict[str, np.ndarray]:
        names = PocketAdapter._names(session, "output_names")
        if len(names) != len(outputs):
            raise RuntimeContractError(
                f"Session returned {len(outputs)} outputs but exposes {len(names)} output names"
            )
        return {name: np.asarray(value) for name, value in zip(names, outputs, strict=True)}

    @staticmethod
    def _find_output(
        outputs: Mapping[str, np.ndarray], candidates: Sequence[str], *, label: str
    ) -> str:
        for candidate in candidates:
            if candidate in outputs:
                return candidate
        lowered = {name.casefold(): name for name in outputs}
        for candidate in candidates:
            if candidate.casefold() in lowered:
                return lowered[candidate.casefold()]
        if len(outputs) == 1:
            return next(iter(outputs))
        raise RuntimeContractError(
            f"{label} output is not identifiable; available outputs: {', '.join(outputs) or 'none'}"
        )

    def _validate_state_contract(
        self,
        role: str,
        input_names: tuple[str, ...],
        output_names: tuple[str, ...],
        specs: Sequence[StateSpec],
    ) -> None:
        missing_inputs = [spec.input_name for spec in specs if spec.input_name not in input_names]
        missing_outputs = [
            spec.output_name for spec in specs if spec.output_name not in output_names
        ]
        if missing_inputs or missing_outputs:
            details = []
            if missing_inputs:
                details.append(f"missing inputs: {', '.join(missing_inputs)}")
            if missing_outputs:
                details.append(f"missing outputs: {', '.join(missing_outputs)}")
            raise RuntimeContractError(f"{role} state contract mismatch ({'; '.join(details)})")

    def _validate_session_contract(self, role: str, session: OnnxSession) -> None:
        if role in self._validated_contracts:
            return
        input_names = self._names(session, "input_names")
        output_names = self._names(session, "output_names")
        if not input_names or not output_names:
            # Minimal fake sessions used by lifecycle tests do not expose graph metadata.
            return
        inputs = set(input_names)
        outputs = set(output_names)
        if role == "text_conditioner":
            if "token_ids" not in inputs:
                raise RuntimeContractError("text_conditioner graph must accept token_ids")
            self._find_output(
                {name: np.empty(0) for name in output_names},
                ("embeddings", "text_embeddings", "conditioning"),
                label="text conditioner embeddings",
            )
        elif role == "mimi_encoder":
            if "audio" not in inputs:
                raise RuntimeContractError("mimi_encoder graph must accept audio")
            self._find_output(
                {name: np.empty(0) for name in output_names},
                ("embeddings", "voice_embeddings", "encoded"),
                label="Mimi encoder embeddings",
            )
        elif role == "flow_lm_main":
            missing = {"sequence", "text_embeddings"} - inputs
            if missing:
                raise RuntimeContractError(
                    f"flow_lm_main graph is missing inputs: {', '.join(sorted(missing))}"
                )
            self._validate_state_contract(role, input_names, output_names, self._flow_specs)
            non_state_outputs = outputs - {spec.output_name for spec in self._flow_specs}
            self._find_output(
                {name: np.empty(0) for name in non_state_outputs},
                ("conditioning", "cond", "c", "latent"),
                label="Flow-LM main conditioning",
            )
        elif role == "flow_lm_flow":
            missing = {"c", "s", "t", "x"} - inputs
            if missing:
                raise RuntimeContractError(
                    f"flow_lm_flow graph is missing inputs: {', '.join(sorted(missing))}"
                )
            self._find_output(
                {name: np.empty(0) for name in output_names},
                ("velocity", "flow", "v", "output"),
                label="flow velocity",
            )
        elif role == "mimi_decoder":
            if "latent" not in inputs:
                raise RuntimeContractError("mimi_decoder graph must accept latent")
            self._validate_state_contract(role, input_names, output_names, self._mimi_specs)
            non_state_outputs = outputs - {spec.output_name for spec in self._mimi_specs}
            self._find_output(
                {name: np.empty(0) for name in non_state_outputs},
                ("audio", "waveform", "output"),
                label="Mimi decoder audio",
            )
        self._validated_contracts.add(role)

    def _get_session(self, role: str) -> OnnxSession:
        if role in self._sessions:
            return self._sessions[role]
        try:
            artifact = self.installation.artifact(role)
        except KeyError as err:
            raise RuntimeContractError(
                f"Pocket bundle is missing required artifact: {role!r}"
            ) from err
        session = OnnxSession(
            artifact.path,
            component=role,
            providers=self.providers,
            provider_options=self.provider_options,
            session_options=self.session_options,
        )
        self._sessions[role] = session
        try:
            self._validate_session_contract(role, session)
        except Exception:
            self._sessions.pop(role, None)
            session.close()
            raise
        return session

    def _load_bos_conditioning(self) -> np.ndarray:
        if self._bos_conditioning is not None:
            return self._bos_conditioning
        try:
            self._bos_conditioning = np.asarray(
                np.load(self.installation.artifact("bos_conditioning").path), dtype=np.float32
            )
        except (KeyError, OSError, ValueError) as exc:
            raise RuntimeContractError(f"Could not load BOS conditioning: {exc}") from exc
        return self._bos_conditioning

    @staticmethod
    def _normalize_embeddings(value: Any, *, label: str) -> np.ndarray:
        embeddings = np.asarray(value, dtype=np.float32)
        if embeddings.ndim == 2:
            embeddings = embeddings[None, ...]
        if embeddings.ndim != 3 or embeddings.shape[0] != 1:
            raise RuntimeContractError(
                f"{label} must have shape [1, sequence, dimension], got {embeddings.shape}"
            )
        return embeddings

    def _cached_predefined_voice_asset(
        self,
        name: str,
        bundle_id: str,
        source: HuggingFaceSource,
    ) -> tuple[Path, str]:
        cache_key = {
            "system": "pocket",
            "asset_kind": "predefined_voice_state",
            "model_repo": source.repository,
            "model_revision": source.revision,
            "asset_path": source.path,
            "bundle_id": bundle_id,
            "voice_name": name,
        }
        cache_digest = hashlib.sha256(
            json.dumps(cache_key, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        cache_dir = self._voice_cache_dir / cache_digest[:2] / cache_digest
        state_path = cache_dir / f"{name}.safetensors"
        metadata_path = cache_dir / f"{name}.json"
        lock_path = self._voice_cache_dir / "locks" / f"{cache_digest}.lock"
        with FileLock(lock_path):
            if state_path.exists() or metadata_path.exists():
                if not state_path.is_file() or not metadata_path.is_file():
                    raise PredefinedVoiceIntegrityError(
                        f"Incomplete cached state for predefined Pocket voice {name!r}"
                    )
                try:
                    cache_record = json.loads(metadata_path.read_text(encoding="utf-8"))
                    if not isinstance(cache_record, dict):
                        raise ValueError("invalid cached metadata")
                    if cache_record.get("key") != cache_key:
                        raise ValueError("cache key mismatch")
                    size = cache_record["size"]
                    sha256 = cache_record["sha256"]
                    if type(size) is not int or size <= 0 or not isinstance(sha256, str):
                        raise ValueError("invalid cached integrity metadata")
                    verify_file(state_path, expected_size=size, sha256=sha256)
                except (OSError, KeyError, TypeError, ValueError, IntegrityError) as exc:
                    raise PredefinedVoiceIntegrityError(
                        f"Cached predefined Pocket voice {name!r} failed integrity validation"
                    ) from exc
                return state_path, sha256

            with tempfile.TemporaryDirectory(prefix="onnxvoice-pocket-hf-") as download_dir:
                try:
                    downloaded = download_huggingface_file(
                        source,
                        local_dir=Path(download_dir),
                        offline=self._offline,
                    )
                except OptionalDependencyError:
                    raise
                except Exception as exc:
                    self._raise_predefined_voice_download_error(
                        exc, name=name, bundle_id=bundle_id, repository=source.repository
                    )
                if not downloaded.is_file():
                    raise PredefinedVoiceIntegrityError(
                        f"Downloaded state for predefined Pocket voice {name!r} is missing"
                    )
                cache_dir.mkdir(parents=True, exist_ok=True)
                fd, temporary_name = tempfile.mkstemp(prefix=f".{name}.", dir=cache_dir)
                os.close(fd)
                temporary_state = Path(temporary_name)
                try:
                    shutil.copyfile(downloaded, temporary_state)
                    size = temporary_state.stat().st_size
                    if size <= 0:
                        raise PredefinedVoiceIntegrityError(
                            f"Downloaded state for predefined Pocket voice {name!r} is empty"
                        )
                    sha256 = digest_file(temporary_state)
                    os.replace(temporary_state, state_path)
                    record = {"key": cache_key, "size": size, "sha256": sha256}
                    fd, temporary_name = tempfile.mkstemp(prefix=f".{name}.", dir=cache_dir)
                    os.close(fd)
                    temporary_metadata = Path(temporary_name)
                    try:
                        temporary_metadata.write_text(
                            json.dumps(record, sort_keys=True) + "\n", encoding="utf-8"
                        )
                        os.replace(temporary_metadata, metadata_path)
                    finally:
                        temporary_metadata.unlink(missing_ok=True)
                finally:
                    temporary_state.unlink(missing_ok=True)
                return state_path, sha256

    @staticmethod
    def _raise_predefined_voice_download_error(
        exc: Exception, *, name: str, bundle_id: str, repository: str
    ) -> NoReturn:
        error_name = type(exc).__name__
        status = getattr(getattr(exc, "response", None), "status_code", None)
        if isinstance(exc, OfflineError) or error_name in {
            "LocalEntryNotFoundError",
            "OfflineModeIsEnabled",
        }:
            raise OfflineError(
                f"Predefined Pocket voice {name!r} is not available in the local cache "
                "and offline mode is enabled."
            ) from exc
        if (
            isinstance(exc, AssetAccessError)
            or error_name
            in {
                "PredefinedVoiceAccessError",
                "AssetAccessError",
                "AssetAuthenticationError",
                "AssetPermissionError",
                "GatedRepoError",
            }
            or status in {401, 403}
        ):
            raise PredefinedVoiceAccessError(
                f"Predefined Pocket voice {name!r} requires access to "
                f"{repository}. Accept or request access on the repository "
                "page, authenticate with 'hf auth login' or HF_TOKEN, verify with "
                "'hf auth whoami', and retry."
            ) from exc
        if (
            isinstance(exc, AssetNotFoundError)
            or error_name
            in {
                "AssetNotFoundError",
                "EntryNotFoundError",
                "RemoteEntryNotFoundError",
                "RevisionNotFoundError",
                "RepositoryNotFoundError",
            }
            or status == 404
        ):
            raise PredefinedVoiceNotFoundError(
                f"Predefined Pocket voice {name!r} is not present for bundle "
                f"{bundle_id!r} at the pinned upstream revision."
            ) from exc
        if isinstance(exc, AssetDownloadError):
            raise exc
        raise AssetDownloadError(
            f"Could not download predefined Pocket voice {name!r} for bundle {bundle_id!r}."
        ) from exc

    @staticmethod
    def _predefined_model_state(state_path: Path, *, name: str) -> dict[str, dict[str, np.ndarray]]:
        try:
            tensors = _load_safetensors(state_path)
        except OptionalDependencyError:
            raise
        except Exception as exc:
            raise PredefinedVoiceIntegrityError(
                f"Could not load predefined Pocket voice state {name!r}"
            ) from exc
        if not isinstance(tensors, Mapping) or not tensors:
            raise PredefinedVoiceIntegrityError(
                f"Predefined Pocket voice state {name!r} contains no model tensors"
            )
        model_state: dict[str, dict[str, np.ndarray]] = {}
        for tensor_path, tensor in tensors.items():
            if not isinstance(tensor_path, str) or "/" not in tensor_path:
                raise PredefinedVoiceIntegrityError(
                    f"Predefined Pocket voice state {name!r} has an invalid tensor key"
                )
            module, key = tensor_path.split("/", 1)
            if not module or not key:
                raise PredefinedVoiceIntegrityError(
                    f"Predefined Pocket voice state {name!r} has an invalid tensor key"
                )
            model_state.setdefault(module, {})[key] = np.asarray(tensor)
        return model_state

    def prepare_predefined_voice(self, name: str) -> PocketVoiceState:
        metadata = self._ensure_validated()
        available = self._predefined_voice_names(metadata)
        if not isinstance(name, str) or name not in available:
            names = ", ".join(available) or "none"
            bundle_name = metadata.get("bundle_name") or self.installation.id
            raise RuntimeContractError(
                f"Unknown predefined Pocket voice {name!r} for bundle {bundle_name!r}. "
                f"Available voices: {names}."
            )
        bundle_name = str(metadata.get("bundle_name") or self.installation.id)
        try:
            validate_safe_component(bundle_name, field_name="bundle id")
        except ValueError as exc:
            raise RuntimeContractError(str(exc)) from exc
        source = self._predefined_voice_source(metadata, name, bundle_name)
        cache_key = (
            "pocket",
            "predefined_voice_state",
            source.repository,
            source.revision,
            bundle_name,
            name,
            source.path,
        )
        cached = self._predefined_voice_states.get(cache_key)
        if cached is not None:
            return cached
        state_path, sha256 = self._cached_predefined_voice_asset(name, bundle_name, source)
        model_state = self._predefined_model_state(state_path, name=name)
        flow_state = _flow_state_from_model_state(model_state, self._flow_specs, name=name)
        sample_rate = int(metadata["sample_rate"])
        voice_state = PocketVoiceState(
            flow_state=flow_state,
            sample_rate=sample_rate,
            metadata={
                "sample_rate": sample_rate,
                "bundle_id": bundle_name,
                "voice_name": name,
                "kind": "predefined_flow_state",
                "model_repo": source.repository,
                "model_revision": source.revision,
                "asset_path": source.path,
                "sha256": sha256,
            },
        )
        self._predefined_voice_states[cache_key] = voice_state
        return voice_state

    def prepare_voice(self, audio: np.ndarray, *, sample_rate: int) -> PocketVoiceState:
        metadata = self._ensure_validated()
        expected_rate = metadata.get("sample_rate")
        if sample_rate != expected_rate:
            raise RuntimeContractError(
                f"Pocket reference audio sample rate must be {expected_rate}, got {sample_rate}"
            )
        audio_array = np.asarray(audio, dtype=np.float32)
        if audio_array.ndim == 1:
            audio_array = audio_array.reshape(1, 1, -1)
        elif audio_array.ndim == 3 and audio_array.shape[0] == 1 and audio_array.shape[1] == 1:
            pass
        else:
            raise RuntimeContractError(
                f"Pocket reference audio must have shape [1, 1, samples], got {audio_array.shape}"
            )
        if audio_array.shape[-1] == 0 or not np.isfinite(audio_array).all():
            raise RuntimeContractError("Pocket reference audio must be non-empty and finite")
        encoder = self._get_session("mimi_encoder")
        try:
            outputs = encoder.run({"audio": audio_array})
            named = (
                self._named_outputs(encoder, outputs)
                if self._names(encoder, "output_names")
                else {"encoded": np.asarray(outputs[0])}
            )
            output_name = self._find_output(
                named,
                ("embeddings", "voice_embeddings", "encoded"),
                label="Mimi encoder embeddings",
            )
            embeddings = self._normalize_embeddings(
                named[output_name], label="Mimi encoder embeddings"
            )
            if metadata.get("insert_bos_before_voice"):
                bos = self._normalize_embeddings(
                    self._load_bos_conditioning(), label="BOS conditioning"
                )
                embeddings = np.concatenate((bos, embeddings), axis=1)
        except RuntimeContractError:
            raise
        except Exception as exc:
            raise RuntimeContractError(f"Mimi encoder failed: {exc}") from exc
        return PocketVoiceState(
            embeddings=np.array(embeddings, dtype=np.float32, copy=True),
            sample_rate=sample_rate,
            metadata={
                "sample_rate": sample_rate,
                "encoder_output": output_name,
                "frames": embeddings.shape[1],
            },
            values={output_name: np.array(embeddings, dtype=np.float32, copy=True)},
        )

    def _encode_text(self, session: OnnxSession, token_ids: Sequence[int]) -> np.ndarray:
        tokens = np.asarray([list(token_ids)], dtype=np.int64)
        if tokens.shape[1] == 0:
            raise RuntimeContractError("Pocket token_ids must not be empty")
        outputs = session.run({"token_ids": tokens})
        named = (
            self._named_outputs(session, outputs)
            if self._names(session, "output_names")
            else {"embeddings": outputs[0]}
        )
        output_name = self._find_output(
            named,
            ("embeddings", "text_embeddings", "conditioning"),
            label="text conditioner embeddings",
        )
        return self._normalize_embeddings(named[output_name], label="Text conditioner embeddings")

    @staticmethod
    def _outputs(session: OnnxSession, values: Sequence[Any]) -> dict[str, np.ndarray]:
        if PocketAdapter._names(session, "output_names"):
            return PocketAdapter._named_outputs(session, values)
        return {f"output_{index}": np.asarray(value) for index, value in enumerate(values)}

    def _condition_embeddings(
        self,
        session: OnnxSession,
        state: dict[str, np.ndarray],
        embeddings: np.ndarray,
        latent_dim: int,
    ) -> None:
        empty_sequence = np.empty((1, 0, latent_dim), dtype=np.float32)
        inputs = {"sequence": empty_sequence, "text_embeddings": embeddings, **state}
        outputs = self._outputs(session, session.run(inputs))
        if self._flow_specs:
            update_state_from_named_outputs(state, outputs, self._flow_specs)

    def _condition_prefix(
        self,
        session: OnnxSession,
        state: dict[str, np.ndarray],
        voice_embeddings: np.ndarray,
        text_embeddings: np.ndarray,
        latent_dim: int,
    ) -> None:
        self._condition_embeddings(session, state, voice_embeddings, latent_dim)
        self._condition_embeddings(session, state, text_embeddings, latent_dim)

    def _generate_latents(
        self,
        main: OnnxSession,
        flow: OnnxSession,
        state: dict[str, np.ndarray],
        *,
        temperature: float,
        lsd_steps: int,
        max_frames: int,
        frames_after_eos: int | None,
        latent_dim: int,
        conditioning_dim: int,
    ) -> tuple[np.ndarray, bool, int]:
        if lsd_steps < 1:
            raise RuntimeContractError("lsd_steps must be at least 1")
        frames: list[np.ndarray] = []
        previous_latent: np.ndarray | None = None
        eos_detected = False
        post_eos = 0
        flow_input_ranks = self._input_ranks(flow)
        empty_text = np.empty((1, 0, conditioning_dim), dtype=np.float32)
        for _ in range(max_frames):
            sequence = (
                previous_latent
                if previous_latent is not None
                else np.full((1, 1, latent_dim), np.nan, dtype=np.float32)
            )
            outputs = self._outputs(
                main, main.run({"sequence": sequence, "text_embeddings": empty_text, **state})
            )
            if self._flow_specs:
                update_state_from_named_outputs(state, outputs, self._flow_specs)
            eos_name = next(
                (name for name in ("eos", "eos_logit", "eos_probability") if name in outputs), None
            )
            eos_now = eos_name is not None and bool(np.any(outputs[eos_name] > 0.5))
            non_state = {
                name: value
                for name, value in outputs.items()
                if name not in {spec.output_name for spec in self._flow_specs} and name != eos_name
            }
            conditioning_name = self._find_output(
                non_state,
                ("conditioning", "cond", "c", "latent"),
                label="Flow-LM main conditioning",
            )
            conditioning = self._fit_rank(
                non_state[conditioning_name], flow_input_ranks.get("c", 3)
            )
            x = np.random.standard_normal((1, latent_dim)).astype(np.float32) * np.float32(
                temperature
            )
            x = self._fit_rank(x, flow_input_ranks.get("x", 3))
            for step in range(lsd_steps):
                t = np.float32(1.0 - step / lsd_steps)
                s = np.float32(1.0 - (step + 1) / lsd_steps)
                flow_outputs = self._outputs(
                    flow,
                    flow.run(
                        {
                            "c": conditioning,
                            "s": self._fit_rank(
                                np.asarray([s], dtype=np.float32),
                                flow_input_ranks.get("s", 1),
                            ),
                            "t": self._fit_rank(
                                np.asarray([t], dtype=np.float32),
                                flow_input_ranks.get("t", 1),
                            ),
                            "x": x,
                        }
                    ),
                )
                velocity_name = self._find_output(
                    flow_outputs,
                    ("velocity", "flow", "v", "spectral", "output"),
                    label="flow velocity",
                )
                x = x + (t - s) * np.asarray(flow_outputs[velocity_name], dtype=np.float32)
            latent = self._fit_rank(x, 3)
            frames.append(latent)
            previous_latent = latent
            if eos_now:
                eos_detected = True
                post_eos += 1
                if frames_after_eos is None or post_eos >= frames_after_eos:
                    break
        if not frames:
            raise RuntimeContractError("Pocket Flow-LM generated no latent frames")
        return np.concatenate(frames, axis=1), eos_detected, len(frames)

    def _decode_latents(
        self,
        decoder: OnnxSession,
        latents: np.ndarray,
        state: dict[str, np.ndarray],
        chunk_frames: int,
    ) -> np.ndarray:
        parts: list[np.ndarray] = []
        for start in range(0, latents.shape[1], chunk_frames):
            chunk = latents[:, start : start + chunk_frames, ...]
            outputs = self._outputs(decoder, decoder.run({"latent": chunk, **state}))
            if self._mimi_specs:
                update_state_from_named_outputs(state, outputs, self._mimi_specs)
            non_state = {
                name: value
                for name, value in outputs.items()
                if name not in {spec.output_name for spec in self._mimi_specs}
            }
            audio_name = self._find_output(
                non_state, ("audio", "waveform", "output"), label="Mimi decoder audio"
            )
            parts.append(np.asarray(non_state[audio_name], dtype=np.float32))
        return np.concatenate(parts, axis=-1)

    def infer(
        self,
        token_ids: Sequence[int],
        *,
        voice_state: PocketVoiceState | None = None,
        temperature: float = 0.7,
        lsd_steps: int = 1,
        max_frames: int | None = None,
        frames_after_eos: int | None = None,
        **kwargs: Any,
    ) -> InferenceResult:
        metadata = self._ensure_validated()
        if temperature < 0:
            raise RuntimeContractError("temperature must be non-negative")
        if frames_after_eos is not None and frames_after_eos < 0:
            raise RuntimeContractError("frames_after_eos must be non-negative")
        frame_limit = (
            max_frames
            if max_frames is not None
            else int(metadata.get("max_frames", self.DEFAULT_MAX_FRAMES))
        )
        if frame_limit <= 0:
            raise RuntimeContractError("max_frames must be a positive frame count")
        text_conditioner = self._get_session("text_conditioner")
        main = self._get_session("flow_lm_main")
        flow = self._get_session("flow_lm_flow")
        decoder = self._get_session("mimi_decoder")
        text_embeddings = self._encode_text(text_conditioner, token_ids)
        mimi_state = initialize_state(self._mimi_specs)
        if voice_state is None:
            flow_state = initialize_state(self._flow_specs)
            voice_embeddings = self._normalize_embeddings(
                self._load_bos_conditioning(), label="BOS conditioning"
            )
            self._condition_prefix(
                main, flow_state, voice_embeddings, text_embeddings, int(metadata["latent_dim"])
            )
        else:
            if voice_state.sample_rate != metadata.get("sample_rate"):
                raise RuntimeContractError(
                    "Pocket voice state sample rate does not match the bundle"
                )
            if voice_state.flow_state is not None:
                flow_state = {
                    name: np.array(value, copy=True)
                    for name, value in voice_state.flow_state.items()
                }
                self._condition_embeddings(
                    main, flow_state, text_embeddings, int(metadata["latent_dim"])
                )
            else:
                if voice_state.embeddings is None:
                    raise RuntimeContractError("Pocket voice state has no conditioning data")
                flow_state = initialize_state(self._flow_specs)
                voice_embeddings = self._normalize_embeddings(
                    voice_state.embeddings, label="Pocket voice embeddings"
                )
                self._condition_prefix(
                    main,
                    flow_state,
                    voice_embeddings,
                    text_embeddings,
                    int(metadata["latent_dim"]),
                )
        latents, eos_detected, frame_count = self._generate_latents(
            main,
            flow,
            flow_state,
            temperature=temperature,
            lsd_steps=lsd_steps,
            max_frames=frame_limit,
            frames_after_eos=frames_after_eos,
            latent_dim=int(metadata["latent_dim"]),
            conditioning_dim=int(metadata["conditioning_dim"]),
        )
        chunk_value = metadata.get("decoder_chunk_frames")
        chunk_frames = int(chunk_value if chunk_value is not None else latents.shape[1])
        if chunk_frames <= 0:
            raise RuntimeContractError("decoder_chunk_frames must be positive")
        audio = self._decode_latents(decoder, latents, mimi_state, chunk_frames)
        audio = np.squeeze(audio)
        if audio.ndim != 1:
            if audio.ndim == 2 and 1 in audio.shape:
                audio = audio.reshape(-1)
            else:
                raise RuntimeContractError("Pocket audio output must be one-dimensional")
        audio = np.asarray(audio, dtype=np.float32)
        if not np.isfinite(audio).all():
            raise RuntimeContractError("Pocket decoder produced non-finite audio")
        return InferenceResult(
            audio=audio,
            sample_rate=int(metadata["sample_rate"]),
            metadata={
                "system": "pocket",
                "temperature": temperature,
                "lsd_steps": lsd_steps,
                "frames_generated": frame_count,
                "eos_detected": eos_detected,
            },
        )

    def _initialize_state(
        self,
        manifest: Mapping[str, Any],
        *,
        batch_size: int = 1,
    ) -> dict[str, np.ndarray]:
        """Compatibility helper for legacy callers; v2 runtime uses typed manifests."""
        if isinstance(manifest, Sequence) and not isinstance(manifest, (str, bytes, Mapping)):
            return initialize_state(parse_state_manifest(manifest, name="state"))
        state: dict[str, np.ndarray] = {}
        for name, spec in manifest.items():
            if isinstance(spec, Mapping):
                shape = tuple(spec.get("shape", (1,)))
                if shape and shape[0] is None:
                    shape = (batch_size, *shape[1:])
                dtype = spec.get("dtype", "float32")
            elif isinstance(spec, (list, tuple)):
                shape = tuple(spec)
                dtype = "float32"
            else:
                continue
            state[name] = np.zeros(shape, dtype=dtype)
        return state

    def _owned_sessions(self) -> tuple[Any, ...]:
        return tuple(self._sessions.values())

    def close(self) -> None:
        for session in tuple(self._sessions.values()):
            session.close()
        self._sessions.clear()
        self._bundle_metadata = None
        self._bos_conditioning = None
        self._flow_specs = ()
        self._mimi_specs = ()
        self._validated_contracts.clear()
        self._validated = False
        self._predefined_voice_states.clear()
