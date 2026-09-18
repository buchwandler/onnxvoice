"""Pocket ONNX bundle adapter for multi-session voice synthesis."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..errors import RuntimeContractError
from ..runtime import OnnxSession
from ..types import InferenceResult, Installation
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


@dataclass(frozen=True, slots=True)
class PocketVoiceState:
    """Reusable voice state for Pocket inference."""

    values: Mapping[str, np.ndarray]
    metadata: Mapping[str, Any] = field(default_factory=dict)


class PocketAdapter(SystemAdapter):
    """Pocket ONNX bundle adapter with multi-session management."""

    system = "pocket"

    def __init__(self, installation: Installation, **kwargs: Any) -> None:
        super().__init__(installation, **kwargs)
        self._sessions: dict[str, OnnxSession] = {}
        self._bundle_metadata: dict[str, Any] | None = None
        self._bos_conditioning: np.ndarray | None = None
        self._validated = False

    def _validate_bundle_metadata(self) -> dict[str, Any]:
        """Validate and return bundle metadata."""
        if self._bundle_metadata is not None:
            return self._bundle_metadata

        # Try to load bundle metadata from artifact
        try:
            metadata_artifact = self.installation.artifact("bundle_metadata")
            metadata_path = metadata_artifact.path
        except KeyError as err:
            # Fall back to installation metadata for local opening
            runtime_metadata = self.installation.metadata.get("runtime") or {}
            if not runtime_metadata:
                raise RuntimeContractError(
                    "Pocket bundle metadata not found. "
                    "Provide bundle_metadata artifact or runtime metadata."
                ) from err
            self._bundle_metadata = dict(runtime_metadata)
            return self._bundle_metadata

        try:
            raw = metadata_path.read_text(encoding="utf-8")
            bundle_data = json.loads(raw)
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeContractError(
                f"Could not read Pocket bundle metadata {metadata_path}: {exc}"
            ) from exc

        if not isinstance(bundle_data, dict):
            raise RuntimeContractError("Pocket bundle metadata must be a JSON object")

        # Merge with installation metadata
        merged = {**bundle_data, **self.installation.metadata}
        self._bundle_metadata = merged
        return self._bundle_metadata

    def _ensure_validated(self) -> dict[str, Any]:
        """Ensure bundle metadata is validated and return it."""
        if self._validated:
            assert self._bundle_metadata is not None
            return self._bundle_metadata

        metadata = self._validate_bundle_metadata()

        # Check required fields
        missing = []
        for required_field in POCKET_REQUIRED_METADATA_FIELDS:
            if required_field not in metadata:
                missing.append(required_field)

        if missing:
            raise RuntimeContractError(
                f"Pocket bundle metadata is missing required fields: {', '.join(sorted(missing))}"
            )

        self._validated = True
        return metadata

    def _get_session(self, role: str) -> OnnxSession:
        """Get or create a lazy session for the given role."""
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
        return session

    def _load_bos_conditioning(self) -> np.ndarray:
        """Load BOS conditioning array."""
        if self._bos_conditioning is not None:
            return self._bos_conditioning

        try:
            artifact = self.installation.artifact("bos_conditioning")
            self._bos_conditioning = np.load(artifact.path)
        except (KeyError, FileNotFoundError) as exc:
            raise RuntimeContractError(f"Could not load BOS conditioning: {exc}") from exc

        return self._bos_conditioning

    def prepare_voice(
        self,
        audio: np.ndarray,
        *,
        sample_rate: int,
    ) -> PocketVoiceState:
        """Prepare voice state from reference audio waveform.

        Uses the Mimi encoder to encode the audio into a reusable voice state.
        """
        self._ensure_validated()

        # Get or create Mimi encoder session (lazy)
        encoder_session = self._get_session("mimi_encoder")

        # Prepare audio input
        audio_array = np.asarray(audio, dtype=np.float32)
        if audio_array.ndim == 1:
            audio_array = audio_array.reshape(1, 1, -1)  # [batch, channels, samples]

        # Run encoder
        try:
            outputs = encoder_session.run({"audio": audio_array})
        except Exception as exc:
            raise RuntimeContractError(f"Mimi encoder failed: {exc}") from exc

        # Build voice state from encoder outputs
        output_names = encoder_session.output_names
        values: dict[str, np.ndarray] = {}
        for name, value in zip(output_names, outputs, strict=True):
            values[name] = np.asarray(value)

        return PocketVoiceState(
            values=values,
            metadata={
                "sample_rate": sample_rate,
                "encoder_outputs": output_names,
            },
        )

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
        """Run Pocket inference with token IDs and optional voice state."""
        metadata = self._ensure_validated()

        # Get required sessions
        text_conditioner = self._get_session("text_conditioner")
        flow_lm_main = self._get_session("flow_lm_main")
        flow_lm_flow = self._get_session("flow_lm_flow")
        mimi_decoder = self._get_session("mimi_decoder")

        # Load BOS conditioning
        bos_conditioning = self._load_bos_conditioning()

        # Prepare token input
        token_array = np.asarray([token_ids], dtype=np.int64)

        # Step 1: Text conditioning
        try:
            conditioner_outputs = text_conditioner.run({"tokens": token_array})
        except Exception as exc:
            raise RuntimeContractError(f"Text conditioner failed: {exc}") from exc

        conditioner_names = text_conditioner.output_names
        conditioning = {
            name: np.asarray(value)
            for name, value in zip(conditioner_names, conditioner_outputs, strict=True)
        }

        # Step 2: Prepare Flow-LM inputs
        sample_rate = metadata.get("sample_rate", 24000)
        samples_per_frame = metadata.get("samples_per_frame", 1)
        latent_dim = metadata.get("latent_dim", 1)
        conditioning_dim = metadata.get("conditioning_dim", 1)

        # Initialize state arrays from manifests
        flow_lm_state = self._initialize_state(
            metadata.get("flow_lm_state_manifest") or {},
            batch_size=1,
        )
        mimi_state = self._initialize_state(
            metadata.get("mimi_state_manifest") or {},
            batch_size=1,
        )

        # Add voice state if provided
        if voice_state is not None:
            for key, value in voice_state.values.items():
                if key in flow_lm_state:
                    flow_lm_state[key] = value
                elif key in mimi_state:
                    mimi_state[key] = value

        # Add BOS conditioning if required
        if metadata.get("insert_bos_before_voice", False):
            bos_key = metadata.get("bos_before_voice_file", "bos")
            if bos_key in flow_lm_state:
                flow_lm_state[bos_key] = bos_conditioning

        # Step 3: Autoregressive Flow-LM loop
        max_token_per_chunk = metadata.get("max_token_per_chunk", 200)
        if max_frames is None:
            max_frames = max_token_per_chunk * samples_per_frame

        frames_collected: list[np.ndarray] = []
        frames_count = 0
        eos_detected = False

        # Prepare Flow-LM inputs
        flow_inputs: dict[str, np.ndarray] = {
            "conditioning": conditioning.get("conditioning", np.zeros((1, 1, conditioning_dim))),
            "temperature": np.asarray([temperature], dtype=np.float32),
        }
        flow_inputs.update(flow_lm_state)

        while frames_count < max_frames:
            # Run Flow-LM main
            try:
                flow_outputs = flow_lm_main.run(flow_inputs)
            except Exception as exc:
                raise RuntimeContractError(f"Flow-LM main failed: {exc}") from exc

            flow_output_names = flow_lm_main.output_names
            flow_results = {
                name: np.asarray(value)
                for name, value in zip(flow_output_names, flow_outputs, strict=True)
            }

            # Check for EOS
            if "eos" in flow_results and np.any(flow_results["eos"] > 0.5):
                eos_detected = True
                break

            # Extract latent frame
            latent = flow_results["latent"] if "latent" in flow_results else flow_outputs[0]

            frames_collected.append(latent)
            frames_count += 1

            # Update state for next iteration
            for key in flow_lm_state:
                if key in flow_results:
                    flow_lm_state[key] = flow_results[key]

            flow_inputs = {
                "conditioning": conditioning.get(
                    "conditioning", np.zeros((1, 1, conditioning_dim))
                ),
                "temperature": np.asarray([temperature], dtype=np.float32),
            }
            flow_inputs.update(flow_lm_state)

            # Apply frames_after_eos limit
            if eos_detected and frames_after_eos is not None and frames_count >= frames_after_eos:
                break

        # Step 4: LSD (Latent-to-Spectral Decoding) with flow matching
        if frames_collected:
            latent_sequence = np.concatenate(frames_collected, axis=1)

            # Run flow matching for LSD steps
            flow_matching_inputs = {
                "latent": latent_sequence,
                "steps": np.asarray([lsd_steps], dtype=np.int64),
            }
            flow_matching_inputs.update(mimi_state)

            try:
                decoded_outputs = flow_lm_flow.run(flow_matching_inputs)
            except Exception as exc:
                raise RuntimeContractError(f"Flow-LM flow (LSD) failed: {exc}") from exc

            decoded_names = flow_lm_flow.output_names
            decoded = {
                name: np.asarray(value)
                for name, value in zip(decoded_names, decoded_outputs, strict=True)
            }

            # Extract spectral features
            spectral = decoded.get("spectral", decoded_outputs[0])
        else:
            spectral = np.zeros((1, 1, latent_dim), dtype=np.float32)

        # Step 5: Mimi decoder
        decoder_inputs = {"spectral": spectral}
        decoder_inputs.update(mimi_state)

        try:
            audio_outputs = mimi_decoder.run(decoder_inputs)
        except Exception as exc:
            raise RuntimeContractError(f"Mimi decoder failed: {exc}") from exc

        audio = np.asarray(audio_outputs[0])

        # Reshape audio to mono float32
        audio = np.squeeze(audio)
        if audio.ndim != 1:
            if audio.ndim == 2 and 1 in audio.shape:
                audio = audio.reshape(-1)
            else:
                raise RuntimeContractError("Pocket audio output must be one-dimensional")

        audio = audio.astype(np.float32, copy=False)

        return InferenceResult(
            audio=audio,
            sample_rate=sample_rate,
            metadata={
                "system": "pocket",
                "temperature": temperature,
                "lsd_steps": lsd_steps,
                "frames_generated": frames_count,
                "eos_detected": eos_detected,
            },
        )

    def _initialize_state(
        self,
        manifest: Mapping[str, Any],
        *,
        batch_size: int = 1,
    ) -> dict[str, np.ndarray]:
        """Initialize state arrays from a manifest."""
        state: dict[str, np.ndarray] = {}
        for name, spec in manifest.items():
            if isinstance(spec, dict):
                shape = tuple(spec.get("shape", (1,)))
                dtype = spec.get("dtype", "float32")
                # Replace batch dimension
                if shape and shape[0] is None:
                    shape = (batch_size, *shape[1:])
                state[name] = np.zeros(shape, dtype=dtype)
            elif isinstance(spec, (list, tuple)):
                state[name] = np.zeros(spec, dtype=np.float32)
        return state

    def _owned_sessions(self) -> tuple[Any, ...]:
        """Return all lazily-created sessions owned by this adapter."""
        return tuple(self._sessions.values())

    def close(self) -> None:
        """Close every owned session. Repeated calls are safe."""
        for session in self._sessions.values():
            session.close()
        self._sessions.clear()
        self._bundle_metadata = None
        self._bos_conditioning = None
        self._validated = False
