"""Static, machine-readable inference contracts for registered ONNXVoice engines.

Catalog- and installation-specific facts are resolved by the service layer added
alongside these descriptors. This module deliberately imports no ONNX Runtime
code and never opens model artifacts.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from types import MappingProxyType
from typing import Any

from .errors import OnnxVoiceError


@dataclass(frozen=True, slots=True)
class OptionDescriptor:
    """One API input, inference control, or selection metadata field."""

    name: str
    type: str
    group: str
    description: str
    required: bool = False
    default: Any = None
    default_status: str = "not_applicable"
    minimum: int | float | None = None
    maximum: int | float | None = None
    minimum_inclusive: bool | None = None
    maximum_inclusive: bool | None = None
    choices: tuple[Any, ...] = ()
    unit: str | None = None
    supported: bool | None = True
    status: str = "supported"
    availability: str | None = None
    constraint_status: str = "enforced"
    source: str = "adapter_contract"
    notes: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly copy with tuple-valued fields as lists."""
        result = asdict(self)
        result["choices"] = list(self.choices)
        result["notes"] = list(self.notes)
        return result


@dataclass(frozen=True, slots=True)
class LimitDescriptor:
    """One input, output, reference-audio, or runtime limit."""

    name: str
    value: Any
    status: str
    unit: str
    source: str
    description: str
    constraint_status: str = "unknown"
    notes: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        if isinstance(self.value, tuple):
            result["value"] = list(self.value)
        result["notes"] = list(self.notes)
        return result


@dataclass(frozen=True, slots=True)
class CapabilityDescriptor:
    """A target-sensitive capability; ``None`` means evidence is insufficient."""

    name: str
    supported: bool | None
    status: str
    mode: str | None = None
    requires: tuple[str, ...] = ()
    requires_reference_transcript: bool | None = None
    reason: str | None = None
    constraints: tuple[str, ...] = ()
    source: str = "adapter_contract"

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["requires"] = list(self.requires)
        result["constraints"] = list(self.constraints)
        return result


@dataclass(frozen=True, slots=True)
class EngineContract:
    """Immutable static contract for one engine adapter."""

    engine: str
    inputs: tuple[OptionDescriptor, ...] = ()
    parameters: tuple[OptionDescriptor, ...] = ()
    limits: tuple[LimitDescriptor, ...] = ()
    capabilities: tuple[CapabilityDescriptor, ...] = ()
    selection_metadata: tuple[OptionDescriptor, ...] = ()
    contract_status: str = "available"
    contract_reason: str | None = None
    notes: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """Return a deterministic JSON-friendly static contract."""
        return {
            "engine": self.engine,
            "contract_status": self.contract_status,
            "contract_reason": self.contract_reason,
            "inputs": [item.to_dict() for item in self.inputs],
            "parameters": [item.to_dict() for item in self.parameters],
            "limits": [item.to_dict() for item in self.limits],
            "capabilities": {
                item.name: item.to_dict()
                for item in sorted(self.capabilities, key=lambda x: x.name)
            },
            "selection_metadata": [item.to_dict() for item in self.selection_metadata],
            "notes": list(self.notes),
        }


def _input(
    name: str,
    type_name: str,
    description: str,
    *,
    required: bool = True,
    group: str = "required_inference_inputs",
    availability: str | None = None,
    status: str = "supported",
    supported: bool | None = True,
    notes: tuple[str, ...] = (),
) -> OptionDescriptor:
    return OptionDescriptor(
        name=name,
        type=type_name,
        group=group,
        description=description,
        required=required,
        default_status="not_applicable" if required else "known",
        availability=availability,
        status=status,
        supported=supported,
        notes=notes,
    )


def _control(
    name: str,
    type_name: str,
    description: str,
    *,
    default: Any = None,
    default_status: str = "known",
    minimum: int | float | None = None,
    maximum: int | float | None = None,
    minimum_inclusive: bool | None = None,
    maximum_inclusive: bool | None = None,
    choices: tuple[Any, ...] = (),
    unit: str | None = None,
    availability: str | None = None,
    supported: bool | None = True,
    status: str = "supported",
    constraint_status: str = "enforced",
    notes: tuple[str, ...] = (),
) -> OptionDescriptor:
    return OptionDescriptor(
        name=name,
        type=type_name,
        group="optional_inference_controls",
        description=description,
        required=False,
        default=default,
        default_status=default_status,
        minimum=minimum,
        maximum=maximum,
        minimum_inclusive=minimum_inclusive,
        maximum_inclusive=maximum_inclusive,
        choices=choices,
        unit=unit,
        availability=availability,
        supported=supported,
        status=status,
        constraint_status=constraint_status,
        notes=notes,
    )


def _limit(
    name: str,
    value: Any,
    status: str,
    unit: str,
    source: str,
    description: str,
    *,
    constraint_status: str = "unknown",
    notes: tuple[str, ...] = (),
) -> LimitDescriptor:
    return LimitDescriptor(
        name=name,
        value=value,
        status=status,
        unit=unit,
        source=source,
        description=description,
        constraint_status=constraint_status,
        notes=notes,
    )


def _capability(
    name: str,
    supported: bool | None,
    status: str,
    *,
    mode: str | None = None,
    requires: tuple[str, ...] = (),
    requires_reference_transcript: bool | None = None,
    reason: str | None = None,
    constraints: tuple[str, ...] = (),
    source: str = "adapter_contract",
) -> CapabilityDescriptor:
    return CapabilityDescriptor(
        name=name,
        supported=supported,
        status=status,
        mode=mode,
        requires=requires,
        requires_reference_transcript=requires_reference_transcript,
        reason=reason,
        constraints=constraints,
        source=source,
    )


_TOKEN_IDS = _input("token_ids", "integer[]", "Caller-prepared model-ready token IDs.")
_NO_CLONING = _capability(
    "voice_cloning",
    False,
    "not_supported",
    reason="This adapter does not condition synthesis on reference audio or a voice state prepared from it.",
)

# Contract details here follow public adapter validation. Catalog-dependent
# values (for example target layouts and model manifest limits) are resolved
# separately and take precedence over these static fallbacks.
ENGINE_CONTRACTS: Mapping[str, EngineContract] = MappingProxyType(
    {
        "chatterbox": EngineContract(
            engine="chatterbox",
            inputs=(
                _TOKEN_IDS,
                _input(
                    "voice_state",
                    "ChatterboxVoiceState",
                    "Prepared reference-audio conditioning state.",
                ),
            ),
            parameters=(
                _control(
                    "max_new_tokens",
                    "integer",
                    "Maximum generated speech-token count.",
                    default=1024,
                    minimum=1,
                    minimum_inclusive=True,
                    availability="May be overridden by target runtime metadata.",
                ),
                _control(
                    "repetition_penalty",
                    "number",
                    "Greedy decoding repetition penalty.",
                    default=1.2,
                    minimum=0,
                    minimum_inclusive=False,
                ),
            ),
            limits=(
                _limit(
                    "max_new_tokens",
                    None,
                    "unknown",
                    "tokens",
                    "catalog_metadata_or_runtime_fallback",
                    "Effective default and any cap may be target-runtime metadata dependent.",
                    constraint_status="model_declared",
                ),
            ),
            capabilities=(
                _capability(
                    "voice_cloning",
                    True,
                    "supported",
                    mode="reference_audio",
                    requires=("reference_audio", "voice_state"),
                    requires_reference_transcript=False,
                    reason="prepare_voice encodes reference audio into reusable ChatterboxVoiceState.",
                ),
            ),
            notes=("The adapter requires prepared voice state for inference.",),
        ),
        "inflect": EngineContract(
            engine="inflect",
            inputs=(
                _input(
                    "token_ids",
                    "integer[]",
                    "Non-empty caller-prepared model-ready integer IDs.",
                    notes=("The adapter validates a non-empty integer sequence.",),
                ),
            ),
            parameters=(
                _control(
                    "speed",
                    "number",
                    "Duration scales as 1 / speed.",
                    default=1.0,
                    minimum=0.5,
                    maximum=2.0,
                    minimum_inclusive=True,
                    maximum_inclusive=True,
                ),
                _control(
                    "variation",
                    "number",
                    "Decoder noise scale.",
                    default=0.667,
                    minimum=0.0,
                    maximum=1.0,
                    minimum_inclusive=True,
                    maximum_inclusive=True,
                ),
                _control(
                    "seed",
                    "integer",
                    "Non-negative NumPy random-generator seed.",
                    default=0,
                    minimum=0,
                    minimum_inclusive=True,
                ),
            ),
            limits=(
                _limit(
                    "max_input_tokens",
                    None,
                    "not_declared",
                    "tokens",
                    "adapter_contract",
                    "No authoritative hard token maximum is declared.",
                    constraint_status="not_enforced_by_adapter",
                ),
            ),
            capabilities=(_NO_CLONING,),
            selection_metadata=(
                _input(
                    "voice",
                    "string",
                    "Fixed adapter voice.",
                    required=False,
                    group="selection_metadata",
                    status="fixed",
                    availability="default",
                    notes=("Do not infer a token limit from the fixed voice.",),
                ),
            ),
        ),
        "kitten": EngineContract(
            engine="kitten",
            inputs=(
                _input("style", "tensor", "Required model-ready style tensor."),
                _TOKEN_IDS,
            ),
            parameters=(
                _control(
                    "speed",
                    "number",
                    "Finite, strictly positive speed multiplier; no adapter upper bound is declared.",
                    default=1.0,
                    minimum=0.0,
                    minimum_inclusive=False,
                ),
            ),
            limits=(
                _limit(
                    "max_input_tokens",
                    None,
                    "unknown",
                    "tokens",
                    "onnx_graph_or_manifest",
                    "Only report a fixed graph sequence dimension when adapter enforcement makes it a real limit.",
                    constraint_status="unknown",
                ),
            ),
            capabilities=(_NO_CLONING,),
        ),
        "kitten2": EngineContract(
            engine="kitten2",
            inputs=(_input("text", "string", "Text passed to the high-level infer_text API."),),
            parameters=(
                _control(
                    "voice",
                    "string",
                    "Named voice in the installed bundle.",
                    default_status="unknown",
                ),
                _control(
                    "preset",
                    "string",
                    "Named decoder preset in the installed bundle.",
                    default_status="unknown",
                ),
                _control(
                    "seed",
                    "integer",
                    "Random seed; null delegates to the library RNG behavior.",
                    default=None,
                    default_status="known",
                ),
                _control(
                    "max_new_tokens",
                    "integer",
                    "Maximum generated language-model tokens per chunk.",
                    default=512,
                    minimum=1,
                    minimum_inclusive=True,
                ),
                _control(
                    "use_emotion",
                    "boolean",
                    "Enable expression/emotion conditioning; null follows text markup.",
                    default=None,
                    default_status="known",
                ),
                _control(
                    "use_reference_prompt",
                    "boolean",
                    "Use the selected named voice's reference prompt; null follows bundle configuration.",
                    default=None,
                    default_status="known",
                ),
            ),
            limits=(
                _limit(
                    "max_text_len",
                    None,
                    "not_declared",
                    "characters",
                    "adapter_contract",
                    "Kitten2 chunks input text using bundle settings and declares no hard overall text length.",
                    constraint_status="not_enforced_by_adapter",
                ),
            ),
            capabilities=(_NO_CLONING,),
            selection_metadata=(
                _input(
                    "voices",
                    "string[]",
                    "Named voices declared by the installed bundle.",
                    required=False,
                    group="selection_metadata",
                    status="target_dependent",
                    availability="installed manifest",
                ),
                _input(
                    "presets",
                    "string[]",
                    "Decode presets declared by the installed bundle.",
                    required=False,
                    group="selection_metadata",
                    status="target_dependent",
                    availability="installed manifest",
                ),
            ),
            notes=(
                "Kitten2 exposes infer_text; its adapter infer method intentionally rejects token-level calls.",
            ),
        ),
        "kokoro": EngineContract(
            engine="kokoro",
            inputs=(
                _input(
                    "style",
                    "tensor",
                    "Required model-ready style tensor.",
                    required=False,
                    availability="ordinary single-model or split layout",
                ),
                _TOKEN_IDS,
                _input(
                    "reference",
                    "KokoroReferenceState",
                    "Prepared reference voice state.",
                    required=False,
                    group="capability_specific_inputs",
                    availability="cloning-onnx-v1 layout only",
                    status="conditional",
                ),
            ),
            parameters=(
                _control(
                    "speed",
                    "number",
                    "Ordinary speed control; the cloning layout supports only exactly 1.0.",
                    default=1.0,
                    constraint_status="not_enforced_by_adapter",
                    availability="split runtime; graph-dependent for ordinary single-model; cloning layout accepts only 1.0",
                    notes=("Do not infer an upper or lower bound for ordinary layouts.",),
                ),
                _control(
                    "seed",
                    "integer",
                    "Seed used by split and cloning runtimes; ordinary single-model output is not claimed to change.",
                    default=1234,
                    availability="split and cloning layouts; ordinary single-model output does not use the seed",
                ),
            ),
            limits=(
                _limit(
                    "max_input_tokens",
                    None,
                    "unknown",
                    "tokens",
                    "runtime_manifest_or_graph_contract",
                    "Selected layout determines whether the limit counts caller tokens or model-ready tokens.",
                    constraint_status="model_declared",
                ),
                _limit(
                    "max_content_tokens",
                    508,
                    "conditional",
                    "tokens",
                    "adapter_contract",
                    "Cloning default of 510 includes two adapter-added boundary tokens; split counts caller tokens directly.",
                    constraint_status="enforced",
                    notes=(
                        "The value is applicable only to cloning layout when its effective max_tokens is 510.",
                    ),
                ),
                _limit(
                    "min_reference_seconds",
                    3,
                    "conditional",
                    "seconds",
                    "adapter_contract",
                    "Minimum cloning reference-audio duration.",
                    constraint_status="enforced",
                ),
                _limit(
                    "max_reference_seconds",
                    30,
                    "conditional",
                    "seconds",
                    "adapter_contract",
                    "Maximum cloning reference-audio duration at 24 kHz.",
                    constraint_status="enforced",
                ),
                _limit(
                    "reference_sample_rates",
                    (24000, 16000),
                    "conditional",
                    "Hz",
                    "adapter_contract",
                    "Reference preparation requires matching mono 24 kHz and 16 kHz audio inputs.",
                    constraint_status="enforced",
                ),
            ),
            capabilities=(
                _capability(
                    "voice_cloning",
                    None,
                    "conditional",
                    mode="reference_audio",
                    requires=("reference_audio", "reference_transcript_token_ids", "reference"),
                    requires_reference_transcript=True,
                    reason="Supported only when the selected target distribution uses cloning-onnx-v1.",
                ),
                _capability(
                    "reference_voice_enrollment",
                    None,
                    "conditional",
                    mode="inno_enrollment",
                    requires=("reference_audio",),
                    reason="Optional Inno voice enrollment is separate from cloning-onnx-v1.",
                ),
            ),
        ),
        "piper": EngineContract(
            engine="piper",
            inputs=(_TOKEN_IDS,),
            parameters=(
                _control(
                    "speaker_id",
                    "integer",
                    "Numeric speaker selector; this is not reference-audio cloning.",
                    default=None,
                    availability="required only for graphs with a sid input; rejected by graphs without sid",
                    constraint_status="model_declared",
                ),
                _control(
                    "noise_scale",
                    "number",
                    "Noise scale; no adapter bounds are declared.",
                    default=0.667,
                    constraint_status="not_enforced_by_adapter",
                ),
                _control(
                    "length_scale",
                    "number",
                    "Length scale; no adapter bounds are declared.",
                    default=1.0,
                    constraint_status="not_enforced_by_adapter",
                ),
                _control(
                    "noise_w",
                    "number",
                    "Noise width; no adapter bounds are declared.",
                    default=0.8,
                    constraint_status="not_enforced_by_adapter",
                ),
            ),
            limits=(
                _limit(
                    "max_input_tokens",
                    None,
                    "not_declared",
                    "tokens",
                    "adapter_contract",
                    "The adapter does not claim a hard maximum phoneme/token count.",
                    constraint_status="not_enforced_by_adapter",
                ),
            ),
            capabilities=(
                _capability(
                    "voice_cloning",
                    False,
                    "not_supported",
                    reason="speaker_id selects a graph speaker and does not condition on reference audio.",
                ),
            ),
        ),
        "pocket": EngineContract(
            engine="pocket",
            inputs=(
                _TOKEN_IDS,
                _input(
                    "voice_state",
                    "PocketVoiceState",
                    "Optional prepared voice state; omission uses bundle default/BOS conditioning.",
                    required=False,
                    group="capability_specific_inputs",
                ),
            ),
            parameters=(
                _control(
                    "temperature",
                    "number",
                    "Sampling temperature.",
                    default=0.7,
                    minimum=0.0,
                    minimum_inclusive=True,
                ),
                _control(
                    "lsd_steps",
                    "integer",
                    "Langevin dynamics sampling steps.",
                    default=1,
                    minimum=1,
                    minimum_inclusive=True,
                ),
                _control(
                    "max_frames",
                    "integer",
                    "Maximum generated frames, not text tokens or samples.",
                    default=None,
                    default_status="known",
                    minimum=1,
                    minimum_inclusive=True,
                    unit="frames",
                    notes=(
                        "Effective fallback is 200 frames unless bundle metadata overrides it.",
                    ),
                ),
                _control(
                    "frames_after_eos",
                    "integer",
                    "Number of frames appended after EOS.",
                    default=None,
                    default_status="known",
                    minimum=0,
                    minimum_inclusive=True,
                    unit="frames",
                ),
            ),
            limits=(
                _limit(
                    "max_frames",
                    200,
                    "known",
                    "frames",
                    "adapter_contract",
                    "Adapter fallback; bundle metadata may declare the effective default.",
                    constraint_status="enforced",
                ),
                _limit(
                    "max_input_tokens",
                    None,
                    "not_declared",
                    "tokens",
                    "adapter_contract",
                    "The adapter does not impose a hard text-token limit; catalog chunk metadata is producer guidance only.",
                    constraint_status="not_enforced_by_adapter",
                ),
            ),
            capabilities=(
                _capability(
                    "voice_cloning",
                    None,
                    "conditional",
                    mode="reference_audio",
                    requires=("reference_audio", "voice_state"),
                    reason="Use the selected target's normalized supports_voice_cloning metadata.",
                    source="catalog_metadata",
                ),
            ),
        ),
        "phonon": EngineContract(
            engine="phonon",
            inputs=(_TOKEN_IDS,),
            parameters=(
                _control(
                    "voice",
                    "string",
                    "Bundled projected voice prompt; no reference-audio cloning is supported.",
                    default_status="unknown",
                    availability="Installed or catalog voice metadata.",
                ),
                _control(
                    "duration_delta",
                    "number",
                    "Text-conditioner duration offset; default is bundle metadata dependent.",
                    default_status="unknown",
                ),
                _control(
                    "temperature",
                    "number",
                    "Flow noise temperature; noise scale is sqrt(temperature).",
                    default_status="unknown",
                    minimum=0.0,
                    minimum_inclusive=True,
                ),
                _control(
                    "seed",
                    "integer",
                    "Optional non-negative seed for reproducible per-utterance flow noise.",
                    default=None,
                    minimum=0,
                    minimum_inclusive=True,
                ),
                _control(
                    "eos_threshold",
                    "number",
                    "EOS-logit threshold; the default is bundle metadata dependent.",
                    default_status="unknown",
                ),
                _control(
                    "frames_after_eos",
                    "integer",
                    "Decoded frames to append after detecting EOS.",
                    default_status="unknown",
                    minimum=0,
                    minimum_inclusive=True,
                    unit="frames",
                ),
                _control(
                    "min_frames_before_eos",
                    "integer",
                    "Earliest frame index at which EOS may terminate generation.",
                    default_status="unknown",
                    minimum=0,
                    minimum_inclusive=True,
                    unit="frames",
                ),
                _control(
                    "max_frames",
                    "integer",
                    "Mandatory hard cap on generated frames.",
                    default_status="unknown",
                    minimum=1,
                    minimum_inclusive=True,
                    unit="frames",
                ),
            ),
            limits=(
                _limit(
                    "max_frames",
                    None,
                    "unknown",
                    "frames",
                    "bundle_generation_metadata",
                    "The selected bundle may declare a hard frame cap; otherwise callers must provide one.",
                    constraint_status="model_declared",
                ),
                _limit(
                    "max_input_tokens",
                    None,
                    "not_declared",
                    "tokens",
                    "adapter_contract",
                    "ONNXVoice accepts caller-prepared IDs and does not tokenize or chunk text.",
                    constraint_status="not_enforced_by_adapter",
                ),
            ),
            capabilities=(
                _capability(
                    "voice_cloning",
                    False,
                    "not_supported",
                    reason="Phonon selects bundled projected voice prompts; reference-audio enrollment is not implemented.",
                ),
                _capability(
                    "streaming",
                    True,
                    "supported",
                    mode="frame_streaming",
                    reason="The adapter yields decoded PCM frames from the four-graph runtime.",
                ),
                _capability(
                    "named_voices",
                    True,
                    "supported",
                    mode="bundled_voice_prompt",
                    reason="The selected bundle declares its available voice prompts.",
                ),
            ),
            selection_metadata=(
                _input(
                    "voices",
                    "string[]",
                    "Voice names declared by the selected bundle.",
                    required=False,
                    group="selection_metadata",
                    status="target_dependent",
                    availability="catalog or installed config",
                ),
                _input(
                    "default_voice",
                    "string",
                    "Bundle-declared default voice, if any.",
                    required=False,
                    group="selection_metadata",
                    status="target_dependent",
                    availability="catalog or installed config",
                ),
            ),
            notes=(
                "The runtime accepts model-ready token IDs and returns streaming PCM; tokenization and long-text chunking belong to a frontend.",
                "Generation defaults are resolved from the selected bundle; controls without defaults remain required.",
            ),
        ),
        "qwen3": EngineContract(
            engine="qwen3",
            inputs=(
                _TOKEN_IDS,
                _input(
                    "voice_state",
                    "Qwen3VoiceState",
                    "Prepared reference voice state; required for Base targets.",
                    required=False,
                    group="capability_specific_inputs",
                    availability="model_type == base",
                    status="conditional",
                ),
                _input(
                    "instruction_token_ids",
                    "integer[]",
                    "Voice-description instruction tokens; required for VoiceDesign targets.",
                    required=False,
                    group="capability_specific_inputs",
                    availability="model_type == voice_design",
                    status="conditional",
                ),
            ),
            parameters=(
                _control(
                    "language",
                    "string",
                    "Non-empty codec language label; valid choices are model-config dependent.",
                    default="english",
                    constraint_status="model_declared",
                ),
                _control(
                    "temperature",
                    "number",
                    "Sampling temperature.",
                    default=0.9,
                    minimum=0.0,
                    maximum=5.0,
                    minimum_inclusive=False,
                    maximum_inclusive=True,
                ),
                _control(
                    "top_k",
                    "integer",
                    "Top-k sampling; zero disables top-k filtering.",
                    default=50,
                    minimum=0,
                    minimum_inclusive=True,
                ),
                _control(
                    "repetition_penalty",
                    "number",
                    "Finite positive repetition penalty.",
                    default=1.05,
                    minimum=0.0,
                    minimum_inclusive=False,
                ),
                _control(
                    "max_new_tokens",
                    "integer",
                    "Generated token limit.",
                    default=2048,
                    minimum=1,
                    maximum=8192,
                    minimum_inclusive=True,
                    maximum_inclusive=True,
                ),
                _control(
                    "seed",
                    "integer",
                    "Optional reproducibility seed.",
                    default=None,
                    minimum=0,
                    maximum=4294967295,
                    minimum_inclusive=True,
                    maximum_inclusive=True,
                ),
            ),
            limits=(
                _limit(
                    "max_reference_samples",
                    240000,
                    "conditional",
                    "samples",
                    "adapter_contract",
                    "Qwen3 Base reference audio maximum.",
                    constraint_status="enforced",
                ),
                _limit(
                    "min_reference_samples",
                    1024,
                    "conditional",
                    "samples",
                    "adapter_contract",
                    "Qwen3 Base reference audio minimum.",
                    constraint_status="enforced",
                ),
                _limit(
                    "max_reference_seconds",
                    10,
                    "conditional",
                    "seconds",
                    "adapter_contract",
                    "Qwen3 Base reference audio maximum at 24 kHz.",
                    constraint_status="enforced",
                ),
                _limit(
                    "reference_sample_rate",
                    24000,
                    "conditional",
                    "Hz",
                    "adapter_contract",
                    "Qwen3 Base reference audio must be at 24 kHz.",
                    constraint_status="enforced",
                ),
                _limit(
                    "max_input_tokens",
                    None,
                    "unknown",
                    "tokens",
                    "installed_manifest",
                    "Model context capacity is not a raw text-token limit; wrapper/chat-template tokens also consume context.",
                    constraint_status="model_declared",
                ),
            ),
            capabilities=(
                _capability(
                    "voice_cloning",
                    None,
                    "conditional",
                    mode="reference_audio",
                    requires=("reference_audio", "reference_transcript_token_ids", "voice_state"),
                    requires_reference_transcript=True,
                    reason="Supported for Base targets, not VoiceDesign targets.",
                ),
                _capability(
                    "voice_design",
                    None,
                    "conditional",
                    mode="instruction_tokens",
                    requires=("instruction_token_ids",),
                    reason="Supported for VoiceDesign targets only.",
                ),
            ),
        ),
        "sopro": EngineContract(
            engine="sopro",
            inputs=(
                _TOKEN_IDS,
                _input(
                    "voice_state", "SoproVoiceState", "Required prepared reference voice state."
                ),
                _input(
                    "prompt_tokens",
                    "integer[]",
                    "Optional one-dimensional semantic prompt-token override.",
                    required=False,
                    group="capability_specific_inputs",
                    availability="bounded by manifest generation.prompt_tokens and semantic vocabulary",
                    status="conditional",
                ),
            ),
            parameters=(
                _control(
                    "language",
                    "string",
                    "Language selector; empty/auto selects automatic language mode.",
                    default="auto",
                    choices=("auto", "en", "pt", "fr", "de"),
                ),
                _control(
                    "temperature",
                    "number",
                    "Sampling temperature; effective default comes from manifest.",
                    default_status="unknown",
                    minimum=0.0,
                    minimum_inclusive=True,
                ),
                _control(
                    "top_p",
                    "number",
                    "Nucleus sampling threshold.",
                    default_status="unknown",
                    minimum=0.0,
                    maximum=1.0,
                    minimum_inclusive=True,
                    maximum_inclusive=True,
                ),
                _control(
                    "top_k",
                    "integer",
                    "Top-k filtering; zero disables top-k.",
                    default_status="unknown",
                    minimum=0,
                    minimum_inclusive=True,
                ),
                _control(
                    "max_seconds",
                    "number",
                    "Maximum generated audio duration.",
                    default_status="unknown",
                    minimum=0.0,
                    minimum_inclusive=False,
                    unit="seconds",
                    availability="bounded by manifest generation.max_seconds",
                ),
                _control(
                    "min_seconds",
                    "number",
                    "Minimum generated audio duration.",
                    default_status="unknown",
                    minimum=0.0,
                    minimum_inclusive=True,
                    unit="seconds",
                    availability="must not exceed effective max_seconds",
                ),
                _control(
                    "seed",
                    "integer",
                    "Random seed.",
                    default=1,
                    minimum=0,
                    maximum=4294967295,
                    minimum_inclusive=True,
                    maximum_inclusive=True,
                ),
                _control(
                    "steps",
                    "integer",
                    "Fixed manifest-declared inference step count.",
                    default_status="unknown",
                    minimum=1,
                    minimum_inclusive=True,
                    availability="caller may only provide the manifest-declared value",
                    constraint_status="model_declared",
                ),
            ),
            limits=(
                _limit(
                    "max_text_len",
                    None,
                    "unknown",
                    "tokens",
                    "installed_manifest",
                    "Input text-token bound from model.max_text_len.",
                    constraint_status="model_declared",
                ),
                _limit(
                    "max_seconds",
                    None,
                    "unknown",
                    "seconds",
                    "installed_manifest",
                    "Output-duration limit from generation.max_seconds.",
                    constraint_status="model_declared",
                ),
                _limit(
                    "max_prompt_tokens",
                    None,
                    "unknown",
                    "tokens",
                    "installed_manifest",
                    "Reference prompt limit from generation.prompt_tokens.",
                    constraint_status="model_declared",
                ),
                _limit(
                    "min_reference_seconds",
                    None,
                    "unknown",
                    "seconds",
                    "installed_manifest",
                    "Reference-audio requirements are declared by the installed manifest.",
                    constraint_status="model_declared",
                ),
                _limit(
                    "reference_seconds",
                    None,
                    "unknown",
                    "seconds",
                    "installed_manifest",
                    "Reference conditioning duration from generation.ref_seconds; the adapter fallback is 10 seconds.",
                    constraint_status="model_declared",
                ),
                _limit(
                    "reference_sample_rate",
                    24000,
                    "known",
                    "Hz",
                    "adapter_contract",
                    "Sopro native Python reference preparation resamples to its 24 kHz profile.",
                    constraint_status="enforced",
                ),
            ),
            capabilities=(
                _capability(
                    "voice_cloning",
                    True,
                    "supported",
                    mode="reference_audio",
                    requires=("reference_audio", "voice_state"),
                    requires_reference_transcript=False,
                    reason="Reference audio is prepared into a reusable SoproVoiceState.",
                    constraints=(
                        "Native Python inference requires profile wasm-uint8 and CPUExecutionProvider.",
                    ),
                ),
            ),
        ),
        "supertonic": EngineContract(
            engine="supertonic",
            inputs=(
                _TOKEN_IDS,
                _input("text_mask", "tensor", "Required model-ready text attention mask."),
                _input("style_ttl", "tensor", "Required model-ready TTL style tensor."),
                _input("style_dp", "tensor", "Required model-ready DP style tensor."),
            ),
            parameters=(
                _control(
                    "steps",
                    "integer",
                    "Denoising steps.",
                    default=5,
                    minimum=1,
                    maximum=100,
                    minimum_inclusive=True,
                    maximum_inclusive=True,
                ),
                _control(
                    "speed",
                    "number",
                    "Output speed multiplier.",
                    default=1.05,
                    minimum=0.7,
                    maximum=2.0,
                    minimum_inclusive=True,
                    maximum_inclusive=True,
                ),
                _control(
                    "seed",
                    "integer",
                    "Optional random seed.",
                    default=None,
                    minimum=0,
                    maximum=4294967295,
                    minimum_inclusive=True,
                    maximum_inclusive=True,
                ),
            ),
            limits=(
                _limit(
                    "max_input_tokens",
                    None,
                    "not_declared",
                    "tokens",
                    "adapter_contract",
                    "No general hard token limit is declared.",
                    constraint_status="not_enforced_by_adapter",
                ),
            ),
            capabilities=(_NO_CLONING,),
        ),
    }
)


def get_engine_contract(engine: str) -> EngineContract:
    """Return a registered static contract or an explicit unavailable contract.

    This function intentionally accepts third-party adapter names rather than
    raising, allowing discovery clients to explain that contract data is absent.
    """
    key = engine.casefold()
    contract = ENGINE_CONTRACTS.get(key)
    if contract is not None:
        return contract
    return EngineContract(
        engine=key,
        contract_status="unavailable",
        contract_reason="No engine-options contract is registered for this adapter.",
        notes=("Third-party adapters may provide their own descriptor registry.",),
    )


def contract_coverage(registered: tuple[str, ...] | None = None) -> dict[str, Any]:
    """Report built-in registry coverage without importing runtime/model code."""
    if registered is None:
        from .systems import registered_systems

        registered = registered_systems()
    registered_names = tuple(sorted({name.casefold() for name in registered}))
    missing = [name for name in registered_names if name not in ENGINE_CONTRACTS]
    extra = [name for name in ENGINE_CONTRACTS if name not in registered_names]
    return {"registered": list(registered_names), "missing": missing, "extra": sorted(extra)}


def _jsonable(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    return value


__all__ = [
    "CapabilityDescriptor",
    "ENGINE_CONTRACTS",
    "EngineContract",
    "LimitDescriptor",
    "OptionDescriptor",
    "contract_coverage",
    "get_engine_contract",
    "EngineOptionsError",
    "list_engine_targets",
    "resolve_engine_options",
]


class EngineOptionsError(ValueError):
    """Actionable target, selector, or engine-options resolution error."""


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _as_list(value: Any) -> list[Any]:
    if isinstance(value, (list, tuple)):
        return list(value)
    return []


def _catalog_metadata(item: Any) -> Mapping[str, Any]:
    return _mapping(getattr(item, "metadata", None))


def _runtime_metadata(metadata: Mapping[str, Any]) -> Mapping[str, Any]:
    return _mapping(metadata.get("runtime"))


def _metadata_choices(metadata: Mapping[str, Any], key: str) -> tuple[str, ...]:
    raw = metadata.get(key)
    if isinstance(raw, Mapping):
        return tuple(sorted(str(choice) for choice in raw))
    return tuple(sorted(str(choice) for choice in raw)) if isinstance(raw, (list, tuple)) else ()


def _item_choices(item: Any, key: str) -> tuple[str, ...]:
    metadata = _catalog_metadata(item)
    raw = getattr(item, key, None) if key == "distribution_choices" else metadata.get(key)
    if isinstance(raw, Mapping):
        return tuple(sorted(str(choice) for choice in raw))
    return tuple(sorted(str(choice) for choice in raw)) if isinstance(raw, (list, tuple)) else ()


def _find_item(items: list[Any], target: str) -> Any | None:
    matches = [item for item in items if item.id == target or target in item.aliases]
    if len(matches) > 1:
        raise EngineOptionsError(
            f"Target alias {target!r} is ambiguous: {', '.join(sorted(i.id for i in matches))}"
        )
    return matches[0] if matches else None


def _catalog_items(manager: Any, engine: str, refresh: bool) -> tuple[list[Any] | None, str | None]:
    from .errors import CatalogError, OfflineError

    try:
        return manager.catalog.list(engine, refresh=refresh), None
    except (OfflineError, CatalogError, OSError) as exc:
        return None, str(exc)


def _read_installed_json(installation: Any) -> tuple[dict[str, Mapping[str, Any]], list[str]]:
    """Read only small JSON manifest/config artifacts; never touch model files."""
    import json

    found: dict[str, Mapping[str, Any]] = {}
    errors: list[str] = []
    allowed_roles = {"manifest", "config", "bundle_metadata", "metadata"}
    for artifact in getattr(installation, "artifacts", ()):
        role = getattr(artifact, "role", None)
        path = getattr(artifact, "path", None)
        if (
            role not in allowed_roles
            or path is None
            or getattr(path, "suffix", "").casefold() != ".json"
        ):
            continue
        try:
            if path.stat().st_size > 2_000_000:
                errors.append(f"{path.name}: JSON metadata exceeds the 2 MB inspection limit")
                continue
            parsed = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, ValueError) as exc:
            errors.append(f"{getattr(path, 'name', role)}: {exc}")
            continue
        if isinstance(parsed, Mapping):
            found[str(role)] = parsed
    return found, errors


def _installed_for(manager: Any, item: Any | None, engine: str, target: str) -> Any | None:
    try:
        installations = manager.installed(engine)
    except (OnnxVoiceError, OSError, ValueError):
        return None
    wanted_id = getattr(item, "id", target)
    wanted_metadata = _catalog_metadata(item)
    wanted_quality = wanted_metadata.get("selected_quality") or wanted_metadata.get("quality")
    wanted_distribution = wanted_metadata.get("selected_distribution") or wanted_metadata.get(
        "distribution_id"
    )
    candidates = [installation for installation in installations if installation.id == wanted_id]
    for installation in candidates:
        metadata = _mapping(getattr(installation, "metadata", None))
        installed_quality = metadata.get("selected_quality") or metadata.get("quality")
        installed_distribution = metadata.get("selected_distribution") or metadata.get(
            "distribution_id"
        )
        if wanted_quality is not None and installed_quality != wanted_quality:
            continue
        if wanted_distribution is not None and installed_distribution != wanted_distribution:
            continue
        return installation
    return None


def _manifest_roots(manifests: Mapping[str, Mapping[str, Any]]) -> tuple[Mapping[str, Any], ...]:
    roots: list[Mapping[str, Any]] = []
    for role in ("manifest", "config", "bundle_metadata", "metadata"):
        raw = manifests.get(role)
        if not isinstance(raw, Mapping):
            continue
        nested = raw.get("manifest")
        if isinstance(nested, Mapping):
            roots.append(nested)
        roots.append(raw)
    return tuple(roots)


def _lookup(roots: tuple[Mapping[str, Any], ...], *paths: tuple[str, ...]) -> Any:
    for root in roots:
        for path in paths:
            value: Any = root
            for key in path:
                if not isinstance(value, Mapping) or key not in value:
                    value = None
                    break
                value = value[key]
            if value is not None:
                return value
    return None


def _install_value(
    result: dict[str, Any],
    name: str,
    value: Any,
    *,
    source: str,
    status: str = "known",
    unit: str | None = None,
    description: str | None = None,
    constraint_status: str = "model_declared",
) -> None:
    for limit in result["limits"]:
        if limit["name"] == name:
            limit.update(
                value=value,
                status=status,
                source=source,
                constraint_status=constraint_status,
            )
            if unit is not None:
                limit["unit"] = unit
            if description is not None:
                limit["description"] = description
            return
    result["limits"].append(
        {
            "name": name,
            "value": value,
            "status": status,
            "unit": unit or "",
            "source": source,
            "description": description or "Resolved target-specific limit.",
            "constraint_status": constraint_status,
            "notes": [],
        }
    )


def _set_option(result: dict[str, Any], name: str, **values: Any) -> None:
    for option in (*result["inputs"], *result["parameters"]):
        if option["name"] == name:
            option.update(values)
            return


def _fact_source(metadata: Mapping[str, Any], key: str) -> str:
    """Return the highest-priority declared metadata source for one fact."""
    catalog = _mapping(metadata.get("_catalog_metadata"))
    installed = _mapping(metadata.get("_installed_metadata"))
    for root, source in ((catalog, "catalog_metadata"), (installed, "installed_manifest")):
        runtime = _runtime_metadata(root)
        if (key in root and root[key] is not None) or (key in runtime and runtime[key] is not None):
            return source
    return "installed_manifest" if metadata.get("_manifest_loaded") else "unknown"


def _target_mapping(metadata: Mapping[str, Any], key: str) -> tuple[Mapping[str, Any] | None, str]:
    """Find a target metadata mapping, preferring catalog facts over installs."""
    catalog = _mapping(metadata.get("_catalog_metadata"))
    installed = _mapping(metadata.get("_installed_metadata"))
    for root, source in ((catalog, "catalog_metadata"), (installed, "installed_manifest")):
        direct = root.get(key)
        if isinstance(direct, Mapping) and direct:
            return direct, source
        nested = _runtime_metadata(root).get(key)
        if isinstance(nested, Mapping) and nested:
            return nested, source
    return None, "unknown"


def _target_value_source(metadata: Mapping[str, Any], section: str, field: str) -> str:
    """Find the source of a field nested in target generation/model metadata."""
    catalog = _mapping(metadata.get("_catalog_metadata"))
    installed = _mapping(metadata.get("_installed_metadata"))
    for root, source in ((catalog, "catalog_metadata"), (installed, "installed_manifest")):
        for parent in (root, _runtime_metadata(root)):
            values = parent.get(section)
            if isinstance(values, Mapping) and field in values and values[field] is not None:
                return source
    return "installed_manifest" if metadata.get("_manifest_loaded") else "unknown"


def _explicit_capability_source(metadata: Mapping[str, Any], name: str) -> str:
    catalog = _mapping(metadata.get("_catalog_metadata"))
    installed = _mapping(metadata.get("_installed_metadata"))
    if _capability_override(catalog, name) is not None:
        return "catalog_metadata"
    if _capability_override(installed, name) is not None:
        return "installed_manifest"
    return "installed_manifest" if metadata.get("_manifest_loaded") else "unknown"


def _capability_override(metadata: Mapping[str, Any], name: str) -> Mapping[str, Any] | bool | None:
    runtime = _runtime_metadata(metadata)
    for root in (metadata, runtime):
        caps = root.get("capabilities")
        if isinstance(caps, Mapping) and name in caps:
            raw = caps[name]
            if isinstance(raw, (Mapping, bool)):
                return raw
        raw = root.get(name)
        if isinstance(raw, (Mapping, bool)):
            return raw
    raw = metadata.get(f"supports_{name}")
    if isinstance(raw, bool):
        return raw
    return None


def _capability_dict(contract: EngineContract) -> dict[str, dict[str, Any]]:
    return {
        item.name: item.to_dict() for item in sorted(contract.capabilities, key=lambda x: x.name)
    }


def _apply_explicit_capability(
    result: dict[str, Any], metadata: Mapping[str, Any], name: str
) -> bool:
    explicit = _capability_override(metadata, name)
    if explicit is None:
        return False
    if isinstance(explicit, bool):
        existing = result["capabilities"].get(name, {"name": name})
        existing.update(
            supported=explicit,
            status="supported" if explicit else "not_supported",
            source=_explicit_capability_source(metadata, name),
        )
    else:
        existing = dict(result["capabilities"].get(name, {"name": name}))
        existing.update(explicit)
        existing.setdefault("source", _explicit_capability_source(metadata, name))
        if "supported" in explicit and "status" not in explicit:
            existing["status"] = "supported" if explicit["supported"] else "not_supported"
        result["capabilities"][name] = existing
        return True
    result["capabilities"][name] = existing
    return True


def _apply_target_rules(
    result: dict[str, Any],
    engine: str,
    metadata: Mapping[str, Any],
    manifests: Mapping[str, Mapping[str, Any]],
    *,
    manifest_errors: list[str],
) -> None:
    runtime = _runtime_metadata(metadata)
    roots = _manifest_roots(manifests)
    if manifests:
        result["metadata_sources"].append("installed_manifest")
    if manifest_errors:
        result["metadata_warnings"] = sorted(manifest_errors)

    for capability_name in tuple(result["capabilities"]):
        _apply_explicit_capability(result, metadata, capability_name)

    layout = runtime.get("layout")
    if not isinstance(layout, str):
        layout = _lookup(roots, ("layout",), ("runtime", "layout"))
    if not isinstance(layout, str):
        layout = None
    model_type = runtime.get("model_type") or metadata.get("model_type")
    if model_type not in {"base", "voice_design"}:
        model_type = _lookup(roots, ("model_type",), ("config", "model_type"))

    if engine == "kokoro":
        cloning = layout == "cloning-onnx-v1"
        if _capability_override(metadata, "voice_cloning") is None:
            result["capabilities"]["voice_cloning"] = {
                "name": "voice_cloning",
                "supported": True if cloning else False if layout else None,
                "status": "supported" if cloning else "not_supported" if layout else "unknown",
                "mode": "reference_audio" if cloning else None,
                "requires": ["reference_audio", "reference_transcript_token_ids", "reference"]
                if cloning
                else [],
                "requires_reference_transcript": True if cloning else None,
                "reason": (
                    "Selected distribution uses cloning-onnx-v1."
                    if cloning
                    else "Selected layout does not use reference-audio conditioning."
                    if layout
                    else "Target runtime layout is unavailable."
                ),
                "constraints": [],
                "source": _fact_source(metadata, "layout") if layout else "unknown",
            }
        if _capability_override(metadata, "reference_voice_enrollment") is None:
            enrollers = runtime.get("voice_enrollers")
            has_enroller = isinstance(enrollers, (list, tuple)) and bool(enrollers)
            result["capabilities"]["reference_voice_enrollment"] = {
                "name": "reference_voice_enrollment",
                "supported": has_enroller if isinstance(enrollers, (list, tuple)) else None,
                "status": "supported"
                if has_enroller
                else "not_supported"
                if isinstance(enrollers, (list, tuple))
                else "unknown",
                "mode": "inno_enrollment" if has_enroller else None,
                "requires": ["reference_audio"] if has_enroller else [],
                "requires_reference_transcript": None,
                "reason": "Optional Inno enrollment is distinct from the full cloning layout.",
                "constraints": [],
                "source": _fact_source(metadata, "voice_enrollers")
                if isinstance(enrollers, (list, tuple))
                else "unknown",
            }
        if cloning:
            configured = runtime.get("max_tokens")
            manifest_value = _lookup(roots, ("max_tokens",), ("runtime", "max_tokens"))
            max_tokens = (
                configured if type(configured) is int and configured > 0 else manifest_value
            )
            source = (
                _fact_source(metadata, "max_tokens")
                if type(configured) is int and configured > 0
                else "installed_manifest"
            )
            if type(max_tokens) is not int or max_tokens <= 0:
                max_tokens, source = 510, "adapter_contract"
            _install_value(
                result,
                "max_input_tokens",
                max_tokens,
                source=source,
                description="Cloning model-ready sequence limit, including two adapter-added boundary tokens.",
                constraint_status="enforced",
            )
            _install_value(
                result,
                "max_content_tokens",
                max(0, max_tokens - 2),
                source=source,
                description="Caller-provided content tokens after two adapter-added boundary tokens.",
                constraint_status="enforced",
            )
            _set_option(result, "reference", required=True, status="required", supported=True)
            _set_option(result, "style", required=False, status="not_supported", supported=False)
            _set_option(
                result,
                "speed",
                supported=False,
                status="unsupported_except_value",
                choices=[1.0],
                availability="cloning-onnx-v1 only accepts speed=1.0",
                constraint_status="enforced",
            )
            _set_option(result, "seed", availability="cloning-onnx-v1 layout")
        elif layout in {"split", "split-onnx-v1", "multi"}:
            configured = runtime.get("max_tokens")
            manifest_value = _lookup(roots, ("max_tokens",), ("runtime", "max_tokens"))
            max_tokens = (
                configured if type(configured) is int and configured > 0 else manifest_value
            )
            source = (
                _fact_source(metadata, "max_tokens")
                if type(configured) is int and configured > 0
                else "installed_manifest"
            )
            if type(max_tokens) is not int or max_tokens <= 0:
                max_tokens, source = 510, "adapter_contract"
            _install_value(
                result,
                "max_input_tokens",
                max_tokens,
                source=source,
                description="Split layout compares caller token count directly with this limit.",
                constraint_status="enforced",
            )
            _install_value(
                result,
                "max_content_tokens",
                None,
                source=source,
                status="not_applicable",
                description="Split layout does not add cloning boundary tokens.",
                constraint_status="not_applicable",
            )
            _set_option(result, "reference", status="not_supported", supported=False)
            _set_option(result, "style", required=True, status="required", supported=True)
            _set_option(result, "seed", availability="split layout harmonic source")
        else:
            _install_value(
                result,
                "max_content_tokens",
                None,
                source="onnx_graph",
                status="unknown",
                description="Single-model content limit depends on a fixed graph sequence dimension, which discovery does not inspect.",
                constraint_status="unknown",
            )
            _set_option(
                result,
                "speed",
                status="conditional",
                supported=None,
                availability="Depends on whether the selected graph exposes a speed input.",
            )
            if layout:
                _set_option(result, "style", required=True, status="required", supported=True)
                _set_option(result, "reference", status="not_supported", supported=False)
                _set_option(
                    result,
                    "seed",
                    supported=False,
                    status="no_effect",
                    availability="ordinary single-model runtime does not apply the seed",
                )
        if not layout:
            _set_option(
                result,
                "seed",
                supported=None,
                status="conditional",
                availability="selected target layout is unknown",
            )
        result["layout"] = layout

    elif engine == "qwen3":
        cloning = model_type == "base"
        design = model_type == "voice_design"
        _set_option(
            result,
            "voice_state",
            required=cloning,
            status="required" if cloning else "not_supported" if design else "conditional",
            supported=cloning if model_type else None,
        )
        _set_option(
            result,
            "instruction_token_ids",
            required=design,
            status="required" if design else "not_supported" if cloning else "conditional",
            supported=design if model_type else None,
        )
        if _capability_override(metadata, "voice_cloning") is None:
            result["capabilities"]["voice_cloning"] = {
                "name": "voice_cloning",
                "supported": cloning if model_type else None,
                "status": "supported" if cloning else "not_supported" if design else "unknown",
                "mode": "reference_audio" if cloning else None,
                "requires": ["reference_audio", "reference_transcript_token_ids", "voice_state"]
                if cloning
                else [],
                "requires_reference_transcript": True if cloning else None,
                "reason": "Qwen3 Base requires reference audio and transcript tokens."
                if cloning
                else "VoiceDesign uses text instructions, not reference-audio cloning."
                if design
                else "Target model_type is unavailable.",
                "constraints": [],
                "source": _fact_source(metadata, "model_type") if model_type else "unknown",
            }
        if _capability_override(metadata, "voice_design") is None:
            result["capabilities"]["voice_design"] = {
                "name": "voice_design",
                "supported": design if model_type else None,
                "status": "supported" if design else "not_supported" if cloning else "unknown",
                "mode": "instruction_tokens" if design else None,
                "requires": ["instruction_token_ids"] if design else [],
                "requires_reference_transcript": None,
                "reason": "VoiceDesign target accepts instruction tokens."
                if design
                else "Only Qwen3 VoiceDesign targets support text-directed voice design."
                if cloning
                else "Target model_type is unavailable.",
                "constraints": [],
                "source": _fact_source(metadata, "model_type") if model_type else "unknown",
            }
        catalog_facts = _mapping(metadata.get("_catalog_metadata"))
        context_capacity = _lookup(
            (catalog_facts,),
            ("max_position_embeddings",),
            ("text_config", "max_position_embeddings"),
            ("model", "max_position_embeddings"),
        )
        context_source = "catalog_metadata"
        if type(context_capacity) is not int or context_capacity <= 0:
            context_capacity = _lookup(
                roots,
                ("max_position_embeddings",),
                ("text_config", "max_position_embeddings"),
                ("model", "max_position_embeddings"),
            )
            context_source = "installed_manifest"
        if type(context_capacity) is int and context_capacity > 0:
            _install_value(
                result,
                "model_context_capacity",
                context_capacity,
                source=context_source,
                unit="model tokens",
                description="Model context capacity includes chat-template and wrapper tokens; it is not a raw text-token limit.",
                constraint_status="model_declared",
            )
        result["layout"] = layout

    elif engine == "pocket":
        explicit = _capability_override(metadata, "voice_cloning")
        declared = metadata.get("supports_voice_cloning")
        if not isinstance(declared, bool):
            manifest_declared = _lookup(
                roots, ("supports_voice_cloning",), ("capabilities", "voice_cloning")
            )
            if isinstance(manifest_declared, bool):
                declared = manifest_declared
        if explicit is None:
            supported = declared if isinstance(declared, bool) else None
            result["capabilities"]["voice_cloning"] = {
                "name": "voice_cloning",
                "supported": supported,
                "status": "supported"
                if supported is True
                else "not_supported"
                if supported is False
                else "unknown",
                "mode": "reference_audio" if supported is True else None,
                "requires": ["reference_audio", "voice_state"] if supported is True else [],
                "requires_reference_transcript": False if supported is True else None,
                "reason": "Target metadata declares reference voice-state conditioning."
                if supported is True
                else "Target metadata explicitly disables reference-audio cloning."
                if supported is False
                else "Target metadata does not declare supports_voice_cloning.",
                "constraints": [],
                "source": _explicit_capability_source(metadata, "voice_cloning")
                if isinstance(declared, bool)
                else "unknown",
            }
        frame_limit = metadata.get("max_frames")
        if type(frame_limit) is not int or frame_limit <= 0:
            frame_limit = _lookup(roots, ("max_frames",), ("generation", "max_frames"))
        if type(frame_limit) is int and frame_limit > 0:
            _install_value(
                result,
                "max_frames",
                frame_limit,
                source=_fact_source(metadata, "max_frames")
                if _fact_source(metadata, "max_frames") != "unknown"
                else "installed_manifest",
                unit="frames",
                description="Maximum generated output frames; not an input-token limit.",
                constraint_status="enforced",
            )
        chunk_hint = metadata.get("max_token_per_chunk")
        if chunk_hint is not None:
            result["selection_metadata"].append(
                {
                    "name": "max_token_per_chunk",
                    "value": chunk_hint,
                    "status": "guidance",
                    "source": _fact_source(metadata, "max_token_per_chunk")
                    if _fact_source(metadata, "max_token_per_chunk") != "unknown"
                    else "installed_manifest",
                    "description": "Producer chunking guidance only; not an adapter-enforced hard limit.",
                }
            )
        result["layout"] = layout

    elif engine == "phonon":
        generation_mapping, generation_source = _target_mapping(metadata, "generation")
        generation = dict(_mapping(_lookup(roots, ("generation",), ("onnx", "generation"))))
        if generation_mapping:
            generation.update(generation_mapping)
        elif generation:
            generation_source = "installed_manifest"
        for option_name, field_name in (
            ("duration_delta", "duration_delta_default"),
            ("temperature", "temperature_default"),
            ("eos_threshold", "eos_threshold_default"),
            ("frames_after_eos", "frames_after_eos"),
            ("min_frames_before_eos", "min_frames_before_eos"),
            ("max_frames", "max_frames"),
        ):
            value = generation.get(field_name)
            source = _target_value_source(metadata, "generation", field_name)
            if source == "unknown":
                source = generation_source
            if value is None:
                _set_option(
                    result,
                    option_name,
                    default=None,
                    default_status="unknown",
                    required=True,
                    status="required",
                    source=source,
                    availability="No default is declared; supply this control for inference.",
                )
            else:
                _set_option(
                    result,
                    option_name,
                    default=value,
                    default_status="known",
                    required=False,
                    status="supported",
                    source=source,
                )
            if option_name == "max_frames" and type(value) is int and value > 0:
                _install_value(
                    result,
                    "max_frames",
                    value,
                    source=source,
                    unit="frames",
                    description="Selected bundle's enforced maximum output frame count.",
                    constraint_status="enforced",
                )

        voices = metadata.get("voices")
        if not isinstance(voices, (list, tuple)) or not voices:
            configured_voices = _lookup(roots, ("voices",))
            if isinstance(configured_voices, Mapping):
                voices = tuple(configured_voices)
            elif isinstance(configured_voices, (list, tuple)):
                voice_entries = [
                    entry.get("name") if isinstance(entry, Mapping) else entry
                    for entry in configured_voices
                ]
                voices = tuple(name for name in voice_entries if isinstance(name, str) and name)
        voice_names = (
            sorted(str(name) for name in voices) if isinstance(voices, (list, tuple)) else []
        )
        default_voice = metadata.get("default_voice")
        if not isinstance(default_voice, str) or not default_voice:
            configured_default = _lookup(roots, ("default_voice",), ("voices", "default"))
            default_voice = configured_default if isinstance(configured_default, str) else None
        if voice_names:
            voice_source = _fact_source(metadata, "voices")
            if voice_source == "unknown":
                voice_source = (
                    "installed_manifest"
                    if metadata.get("_manifest_loaded")
                    else "catalog_metadata"
                    if isinstance(metadata.get("_catalog_metadata"), Mapping)
                    else "unknown"
                )
            _set_option(
                result,
                "voice",
                choices=voice_names,
                default=default_voice,
                default_status="known" if default_voice in voice_names else "unknown",
                required=default_voice not in voice_names,
                status="supported" if default_voice in voice_names else "required",
                source=voice_source,
                availability="Declared bundled voice prompts.",
            )
        result["layout"] = "phonon-four-graph-streaming-v1"
        result["field_status"]["layout"] = "known"

    elif engine == "sopro":
        generation_mapping, generation_source = _target_mapping(metadata, "generation")
        generation = dict(_mapping(_lookup(roots, ("generation",))))
        if generation_mapping:
            generation.update(generation_mapping)
        elif not generation:
            generation_source = "adapter_contract"
        elif generation_source == "unknown":
            generation_source = "installed_manifest"
        model_mapping, model_source = _target_mapping(metadata, "model")
        model = dict(_mapping(_lookup(roots, ("model",))))
        if model_mapping:
            model.update(model_mapping)
        elif not model:
            model_source = "adapter_contract"
        elif model_source == "unknown":
            model_source = "installed_manifest"
        for option_name in (
            "temperature",
            "top_p",
            "top_k",
            "max_seconds",
            "min_seconds",
            "seed",
            "steps",
        ):
            value = generation.get(option_name)
            option_source = _target_value_source(metadata, "generation", option_name)
            if option_source == "unknown":
                option_source = generation_source
            if value is None and option_name == "steps":
                value = _lookup(roots, ("steps",))
                option_source = "installed_manifest" if value is not None else option_source
            if value is not None:
                _set_option(
                    result, option_name, default=value, default_status="known", source=option_source
                )
        for name, value, unit, description in (
            (
                "max_text_len",
                model.get("max_text_len"),
                "tokens",
                "Hard input token limit declared by model.max_text_len.",
            ),
            (
                "max_seconds",
                generation.get("max_seconds"),
                "seconds",
                "Output-duration limit declared by generation.max_seconds.",
            ),
            (
                "max_prompt_tokens",
                generation.get("prompt_tokens"),
                "tokens",
                "Reference prompt-token limit declared by generation.prompt_tokens.",
            ),
        ):
            if value is not None:
                section = "model" if name == "max_text_len" else "generation"
                field_name = "prompt_tokens" if name == "max_prompt_tokens" else name
                limit_source = _target_value_source(metadata, section, field_name)
                if limit_source == "unknown":
                    limit_source = model_source if section == "model" else generation_source
                _install_value(
                    result,
                    name,
                    value,
                    source=limit_source,
                    unit=unit,
                    description=description,
                    constraint_status="model_declared",
                )
        reference_seconds = generation.get("ref_seconds", 10.0)
        if (
            isinstance(reference_seconds, (int, float))
            and not isinstance(reference_seconds, bool)
            and math.isfinite(float(reference_seconds))
            and reference_seconds > 0
        ):
            reference_source = _target_value_source(metadata, "generation", "ref_seconds")
            if reference_source == "unknown":
                reference_source = (
                    generation_source if "ref_seconds" in generation else "adapter_contract"
                )
            _install_value(
                result,
                "reference_seconds",
                float(reference_seconds),
                source=reference_source,
                unit="seconds",
                description="Effective reference conditioning duration; callers may override it with another positive finite duration.",
                constraint_status="enforced",
            )
        result["layout"] = layout

    elif engine == "chatterbox":
        generation_mapping, generation_source = _target_mapping(metadata, "generation")
        generation = dict(_mapping(_lookup(roots, ("generation",), ("runtime", "generation"))))
        if generation_mapping:
            generation.update(generation_mapping)
        elif generation:
            generation_source = "installed_manifest"
        else:
            generation_source = "unknown"
        for option_name in ("max_new_tokens", "repetition_penalty"):
            value = generation.get(option_name)
            source = _target_value_source(metadata, "generation", option_name)
            if source == "unknown":
                source = generation_source
            if value is None:
                value = runtime.get(f"default_{option_name}")
                source = _fact_source(metadata, f"default_{option_name}")
            if value is not None:
                _set_option(
                    result, option_name, default=value, default_status="known", source=source
                )
        result["layout"] = layout

    elif engine == "kitten2":
        voice_value = metadata.get("voices") or _lookup(
            roots, ("voices",), ("voice_names",), ("available_voices",)
        )
        if isinstance(voice_value, (Mapping, list, tuple)):
            voices = sorted(str(name) for name in voice_value)
        else:
            voices = []
        preset_value = (
            metadata.get("decode_presets")
            or metadata.get("presets")
            or _lookup(roots, ("decode_presets",), ("presets",), ("generation", "presets"))
        )
        if isinstance(preset_value, (Mapping, list, tuple)):
            presets = sorted(str(name) for name in preset_value)
        else:
            presets = []
        default_voice = metadata.get("default_voice") or _lookup(
            roots, ("default_voice",), ("voices", "default")
        )
        default_preset = metadata.get("default_preset") or _lookup(
            roots, ("default_preset",), ("presets", "default")
        )
        selection_source = _fact_source(metadata, "voices")
        if selection_source == "unknown":
            selection_source = _fact_source(metadata, "default_voice")
        if selection_source == "unknown" and roots:
            selection_source = "installed_manifest"
        result["selection_metadata"].extend(
            [
                {
                    "name": "voices",
                    "value": voices,
                    "status": "known" if voices else "unknown",
                    "source": selection_source if voices else "unknown",
                },
                {
                    "name": "default_voice",
                    "value": default_voice,
                    "status": "known" if default_voice else "unknown",
                    "source": selection_source if default_voice else "unknown",
                },
                {
                    "name": "presets",
                    "value": presets,
                    "status": "known" if presets else "unknown",
                    "source": selection_source if presets else "unknown",
                },
                {
                    "name": "default_preset",
                    "value": default_preset,
                    "status": "known" if default_preset else "unknown",
                    "source": selection_source if default_preset else "unknown",
                },
            ]
        )
        if default_voice is not None:
            _set_option(
                result,
                "voice",
                default=default_voice,
                default_status="known",
                source=selection_source,
            )
        if default_preset is not None:
            _set_option(
                result,
                "preset",
                default=default_preset,
                default_status="known",
                source=selection_source,
            )
        use_reference_prompt = metadata.get("use_reference_prompt")
        if use_reference_prompt is None:
            use_reference_prompt = _lookup(
                roots, ("use_reference_prompt",), ("generation", "use_reference_prompt")
            )
        if isinstance(use_reference_prompt, bool):
            _set_option(
                result,
                "use_reference_prompt",
                default=use_reference_prompt,
                default_status="known",
                source="catalog_metadata"
                if metadata.get("use_reference_prompt") is not None
                else "installed_manifest",
            )
        result["layout"] = layout
    else:
        result["layout"] = layout

    result["metadata_sources"] = sorted(set(result["metadata_sources"]))
    result["limits"].sort(key=lambda value: value["name"])
    result["inputs"].sort(key=lambda value: (value["group"], value["name"]))
    result["parameters"].sort(key=lambda value: value["name"])
    result["selection_metadata"].sort(key=lambda value: value["name"])


def _variant_response(
    manager: Any,
    contract: EngineContract,
    item: Any | None,
    engine: str,
    target: str,
    *,
    quality: str | None,
    distribution: str | None,
) -> dict[str, Any]:
    catalog_metadata = dict(_catalog_metadata(item)) if item is not None else {}
    installation = _installed_for(manager, item, engine, target)
    manifests: dict[str, Mapping[str, Any]] = {}
    manifest_errors: list[str] = []
    installed_metadata: Mapping[str, Any] = {}
    if installation is not None:
        installed_metadata = _mapping(getattr(installation, "metadata", None))
        manifests, manifest_errors = _read_installed_json(installation)
    metadata = dict(installed_metadata)
    if item is not None:
        if getattr(item, "default_voice", None) is not None:
            catalog_metadata.setdefault("default_voice", item.default_voice)
        if getattr(item, "voices", None):
            catalog_metadata.setdefault("voices", list(item.voices))
    metadata.update(catalog_metadata)
    merged_runtime = dict(_runtime_metadata(installed_metadata))
    merged_runtime.update(_runtime_metadata(catalog_metadata))
    if merged_runtime:
        metadata["runtime"] = merged_runtime
    runtime = _runtime_metadata(metadata)
    metadata["_catalog_metadata"] = catalog_metadata
    metadata["_installed_metadata"] = dict(installed_metadata)
    metadata["_manifest_loaded"] = bool(manifests)
    manifest_roots = _manifest_roots(manifests)
    resolved_layout = runtime.get("layout")
    if not isinstance(resolved_layout, str):
        resolved_layout = _lookup(manifest_roots, ("layout",), ("runtime", "layout"))
    manifest_sample_rate = _lookup(
        manifest_roots,
        ("sample_rate",),
        ("runtime", "sample_rate"),
        ("audio", "sample_rate"),
        ("model", "sample_rate"),
    )
    sample_rate = (
        getattr(item, "sample_rate", None)
        or getattr(installation, "sample_rate", None)
        or runtime.get("sample_rate")
        or manifest_sample_rate
    )
    selected_quality = quality or metadata.get("selected_quality") or metadata.get("quality")
    selected_distribution = (
        distribution or metadata.get("selected_distribution") or metadata.get("distribution_id")
    )
    target_id = getattr(item, "id", None) or getattr(installation, "id", None) or target
    artifact_source = (
        getattr(item, "artifacts", ())
        if item is not None
        else getattr(installation, "artifacts", ())
    ) or ()
    quality_available = bool(
        metadata.get("profiles") or metadata.get("quality") or metadata.get("selected_quality")
    ) or any(getattr(artifact, "quality", None) for artifact in artifact_source)
    canonical_ref = getattr(item, "ref", None) or f"{engine}:{target_id}"
    caps = _capability_dict(contract)
    inputs = [entry.to_dict() for entry in contract.inputs]
    params = [entry.to_dict() for entry in contract.parameters]
    result: dict[str, Any] = {
        "schema_version": 1,
        "engine": engine,
        "target": target_id,
        "ref": canonical_ref,
        "quality": selected_quality,
        "distribution": selected_distribution,
        "layout": resolved_layout,
        "sample_rate": sample_rate,
        "capabilities": caps,
        "inputs": inputs,
        "parameters": params,
        "limits": [limit.to_dict() for limit in contract.limits],
        "selection_metadata": [],
        "metadata_sources": ["adapter_contract"]
        + (
            ["catalog_metadata"]
            if item is not None
            else ["installed_manifest"]
            if installation is not None
            else []
        ),
        "field_status": {
            "target": "known",
            "ref": "known",
            "quality": "known"
            if selected_quality is not None
            else "not_selected"
            if quality_available
            else "not_applicable",
            "distribution": "known"
            if selected_distribution is not None
            else "not_selected"
            if metadata.get("distribution_choices")
            else "not_applicable",
            "layout": "known" if resolved_layout is not None else "unknown",
            "sample_rate": "known" if sample_rate is not None else "unknown",
        },
        "contract_status": contract.contract_status,
        "contract_reason": contract.contract_reason,
    }
    if engine == "phonon":
        license_info = metadata.get("license")
        if not isinstance(license_info, Mapping):
            license_info = _lookup(manifest_roots, ("license",))
        if isinstance(license_info, Mapping):
            result["license"] = dict(license_info)
    voices = (
        getattr(item, "voices", None)
        or getattr(installation, "voices", None)
        or metadata.get("voices")
        or ()
    )
    default_voice = (
        getattr(item, "default_voice", None)
        or getattr(installation, "default_voice", None)
        or metadata.get("default_voice")
        or ("default" if engine == "inflect" else None)
    )
    if engine == "phonon":
        if not voices:
            configured_voices = _lookup(manifest_roots, ("voices",))
            if isinstance(configured_voices, (list, tuple)):
                names = [
                    entry.get("name") if isinstance(entry, Mapping) else entry
                    for entry in configured_voices
                ]
                voices = tuple(name for name in names if isinstance(name, str) and name)
        if not default_voice:
            configured_default = _lookup(manifest_roots, ("default_voice",), ("voices", "default"))
            if isinstance(configured_default, str) and configured_default:
                default_voice = configured_default
    choices = (
        _item_choices(item, "distribution_choices")
        if item is not None
        else _metadata_choices(metadata, "distribution_choices")
    )
    profiles = (
        _item_choices(item, "profiles")
        if item is not None
        else _metadata_choices(metadata, "profiles")
    )
    qualities = tuple(
        sorted(
            set(profiles)
            | {
                str(artifact.quality)
                for artifact in artifact_source
                if getattr(artifact, "quality", None)
            }
        )
    )
    language_values = (
        metadata.get("languages") or metadata.get("language_codes") or metadata.get("language")
    )
    if isinstance(language_values, str):
        languages = [language_values]
    elif isinstance(language_values, (list, tuple)):
        languages = sorted(str(value) for value in language_values)
    else:
        languages = []
    qualities_source = _fact_source(metadata, "profiles")
    if qualities and qualities_source == "unknown":
        qualities_source = "catalog_metadata" if item is not None else "installed_manifest"
    languages_source = _fact_source(metadata, "languages")
    if languages_source == "unknown":
        languages_source = _fact_source(metadata, "language_codes")
    if languages_source == "unknown" and languages:
        languages_source = (
            "catalog_metadata"
            if item is not None
            else "installed_manifest"
            if installation is not None
            else "unknown"
        )
    distributions_source = _fact_source(metadata, "distribution_choices")
    if choices and distributions_source == "unknown":
        distributions_source = "catalog_metadata" if item is not None else "installed_manifest"
    voices_source = _fact_source(metadata, "voices")
    if voices and voices_source == "unknown":
        voices_source = "catalog_metadata" if item is not None else "installed_manifest"
    default_voice_source = _fact_source(metadata, "default_voice")
    if engine == "inflect" and default_voice:
        default_voice_source = "adapter_contract"
    elif default_voice and default_voice_source == "unknown":
        default_voice_source = "catalog_metadata" if item is not None else "installed_manifest"
    result["selection_metadata"] = [
        {
            "name": "qualities",
            "value": list(qualities),
            "status": "known" if qualities else "unknown",
            "source": qualities_source if qualities else "unknown",
        },
        {
            "name": "languages",
            "value": languages,
            "status": "known" if languages else "unknown",
            "source": languages_source if languages else "unknown",
        },
        {
            "name": "distributions",
            "value": list(choices),
            "status": "known" if choices else "not_applicable",
            "source": distributions_source if choices else "adapter_contract",
        },
        {
            "name": "voices",
            "value": sorted(str(v) for v in voices),
            "status": "known" if voices else "unknown",
            "source": voices_source if voices else "unknown",
        },
        {
            "name": "default_voice",
            "value": default_voice,
            "status": "known" if default_voice else "unknown",
            "source": default_voice_source if default_voice else "unknown",
        },
    ]
    _apply_target_rules(result, engine, metadata, manifests, manifest_errors=manifest_errors)
    return result


def _generic_response(
    contract: EngineContract,
    engine: str,
    items: list[Any] | None,
    unavailable_reason: str | None,
) -> dict[str, Any]:
    targets = []
    if items is not None:
        for item in sorted(items, key=lambda candidate: candidate.id):
            metadata = _catalog_metadata(item)
            runtime = _runtime_metadata(metadata)
            language_value = (
                metadata.get("languages")
                or metadata.get("language_codes")
                or metadata.get("language")
            )
            if isinstance(language_value, str):
                languages = [language_value]
            elif isinstance(language_value, (list, tuple)):
                languages = sorted(str(value) for value in language_value)
            else:
                languages = []
            targets.append(
                {
                    "target": item.id,
                    "ref": item.ref,
                    "aliases": sorted(str(alias) for alias in getattr(item, "aliases", ())),
                    "quality": metadata.get("selected_quality") or metadata.get("quality"),
                    "qualities": list(_item_choices(item, "profiles")),
                    "distributions": list(_item_choices(item, "distribution_choices")),
                    "layout": runtime.get("layout"),
                    "sample_rate": getattr(item, "sample_rate", None)
                    or metadata.get("sample_rate"),
                    "languages": languages,
                    "voices": sorted(
                        str(voice)
                        for voice in (getattr(item, "voices", None) or metadata.get("voices") or ())
                    ),
                    "default_voice": getattr(item, "default_voice", None)
                    or metadata.get("default_voice"),
                    "license": dict(metadata["license"])
                    if isinstance(metadata.get("license"), Mapping)
                    else None,
                    "field_status": {
                        "target": "known",
                        "ref": "known",
                        "quality": "known"
                        if metadata.get("selected_quality") or metadata.get("quality")
                        else "not_selected"
                        if _item_choices(item, "profiles")
                        else "not_applicable",
                        "distributions": "known"
                        if _item_choices(item, "distribution_choices")
                        else "not_applicable",
                        "layout": "known" if runtime.get("layout") is not None else "unknown",
                        "sample_rate": "known"
                        if getattr(item, "sample_rate", None) is not None
                        or metadata.get("sample_rate") is not None
                        else "unknown",
                        "languages": "known" if languages else "unknown",
                        "default_voice": "known"
                        if getattr(item, "default_voice", None) or metadata.get("default_voice")
                        else "unknown",
                    },
                }
            )
    return {
        "schema_version": 1,
        "engine": engine,
        "target": None,
        "ref": None,
        "quality": None,
        "distribution": None,
        "layout": None,
        "sample_rate": None,
        "capabilities": _capability_dict(contract),
        "inputs": [entry.to_dict() for entry in contract.inputs],
        "parameters": [entry.to_dict() for entry in contract.parameters],
        "limits": [entry.to_dict() for entry in contract.limits],
        "selection_metadata": list(contract.to_dict()["selection_metadata"]),
        "known_targets": targets,
        "field_status": {
            "target": "not_selected" if items is not None else "unknown",
            "ref": "not_selected" if items is not None else "unknown",
            "quality": "not_selected"
            if any(target["qualities"] for target in targets)
            else "not_applicable"
            if items is not None
            else "unknown",
            "distribution": "not_selected"
            if any(target["distributions"] for target in targets)
            else "not_applicable"
            if items is not None
            else "unknown",
            "layout": "not_selected"
            if targets
            else "not_applicable"
            if items is not None
            else "unknown",
            "sample_rate": "unknown",
        },
        "target_discovery": {
            "status": "available" if items is not None else "unavailable",
            "reason": unavailable_reason,
        },
        "metadata_sources": ["adapter_contract", "catalog_metadata"]
        if items is not None
        else ["adapter_contract"],
        "contract_status": contract.contract_status,
        "contract_reason": contract.contract_reason,
    }


def resolve_engine_options(
    manager: Any,
    engine: str,
    *,
    target: str | None = None,
    quality: str | None = None,
    distribution: str | None = None,
    refresh: bool = False,
) -> dict[str, Any]:
    """Resolve an engine contract and optional catalog target without loading models.

    ``manager`` is an :class:`OnnxVoice` instance. Catalog reads may use its
    configured network/cache policy; installed JSON manifests are read only
    from already-installed lightweight metadata artifacts.
    """
    normalized_engine = engine.casefold()
    contract = get_engine_contract(normalized_engine)
    if target is None:
        if quality is not None or distribution is not None:
            raise EngineOptionsError("--quality and --distribution require --target")
        items, unavailable = _catalog_items(manager, normalized_engine, refresh)
        return _generic_response(contract, normalized_engine, items, unavailable)

    items, unavailable = _catalog_items(manager, normalized_engine, refresh)
    item = _find_item(items, target) if items is not None else None
    if (
        item is None
        and items is not None
        and _installed_for(manager, None, normalized_engine, target) is None
    ):
        valid = ", ".join(sorted(candidate.id for candidate in items)) or "none"
        raise EngineOptionsError(
            f"Unknown target {target!r} for engine {normalized_engine}; valid targets: {valid}"
        )
    if item is None:
        installation = _installed_for(manager, None, normalized_engine, target)
        if installation is None:
            reason = (
                f"Catalog target discovery is unavailable ({unavailable})"
                if unavailable
                else "Target is not present in the catalog or installed assets."
            )
            raise EngineOptionsError(f"Cannot resolve {normalized_engine}:{target}: {reason}")
        installation_metadata = _mapping(installation.metadata)
        installed_distribution = installation_metadata.get(
            "selected_distribution"
        ) or installation_metadata.get("distribution_id")
        installed_quality = installation_metadata.get(
            "selected_quality"
        ) or installation_metadata.get("quality")
        if distribution is not None and installed_distribution != distribution:
            raise EngineOptionsError(
                f"Installed target {normalized_engine}:{target} does not match distribution {distribution!r}"
            )
        if quality is not None and installed_quality != quality:
            raise EngineOptionsError(
                f"Installed target {normalized_engine}:{target} does not match quality {quality!r}"
            )
        return _variant_response(
            manager,
            contract,
            None,
            normalized_engine,
            target,
            quality=quality,
            distribution=distribution,
        )

    choices = _item_choices(item, "distribution_choices")
    if not choices:
        declared = _catalog_metadata(item).get("distribution_id")
        if isinstance(declared, str):
            choices = (declared,)
    if distribution is not None:
        if choices and distribution not in choices:
            raise EngineOptionsError(
                f"Unknown distribution {distribution!r} for {item.ref}; valid choices: {', '.join(choices)}"
            )
        if not choices:
            raise EngineOptionsError(f"Target {item.ref} does not declare distribution choices")
        selected_distributions: tuple[str | None, ...] = (distribution,)
    elif normalized_engine == "kokoro" and len(choices) > 1:
        selected_distributions = tuple(choices)
    else:
        selected = _catalog_metadata(item).get("distribution_id")
        selected_distributions = (selected if isinstance(selected, str) else None,)

    responses: list[dict[str, Any]] = []
    for selected_distribution in selected_distributions:
        try:
            selected_item = manager.catalog.resolve(
                item.ref,
                refresh=refresh,
                quality=quality,
                distribution=selected_distribution,
            )
        except (OnnxVoiceError, ValueError, OSError) as exc:
            raise EngineOptionsError(str(exc)) from exc
        responses.append(
            _variant_response(
                manager,
                contract,
                selected_item,
                normalized_engine,
                target,
                quality=quality,
                distribution=selected_distribution,
            )
        )
    if len(responses) == 1:
        return responses[0]
    result = _generic_response(contract, normalized_engine, [item], None)
    result.update(
        target=item.id,
        ref=item.ref,
        quality=quality,
        variants=responses,
        distribution=None,
        layout=None,
        sample_rate=responses[0]["sample_rate"]
        if len({response["sample_rate"] for response in responses}) == 1
        else None,
        field_status={
            "target": "known",
            "ref": "known",
            "quality": "known"
            if quality is not None
            else "not_selected"
            if _item_choices(item, "profiles")
            else "not_applicable",
            "distribution": "multiple_variants",
            "layout": "multiple_variants",
            "sample_rate": "known"
            if responses[0]["sample_rate"] is not None
            and len({response["sample_rate"] for response in responses}) == 1
            else "unknown",
        },
        selection_metadata=result["selection_metadata"]
        + [
            {
                "name": "distribution_variants",
                "value": [response["distribution"] for response in responses],
                "status": "known",
                "source": "catalog_metadata",
            }
        ],
        target_discovery={"status": "available", "reason": None},
    )
    return result


def list_engine_targets(
    manager: Any,
    *,
    engine: str | None = None,
    capability: str | None = None,
    refresh: bool = False,
) -> dict[str, Any]:
    """List target-level capability rows; unavailable catalogs remain explicit."""
    from .systems import registered_systems

    if capability not in {None, "voice-cloning"}:
        raise EngineOptionsError(f"Unknown capability {capability!r}; valid choices: voice-cloning")
    engines = (engine.casefold(),) if engine else registered_systems()
    rows: list[dict[str, Any]] = []
    unavailable_engines: list[dict[str, str | None]] = []
    for name in engines:
        contract = get_engine_contract(name)
        items, unavailable = _catalog_items(manager, name, refresh)
        if items is None:
            generic = _generic_response(contract, name, None, unavailable)
            unavailable_engines.append({"engine": name, "reason": unavailable})
            clone = generic["capabilities"].get("voice_cloning", {})
            if capability is None:
                rows.append(
                    {
                        "engine": name,
                        "target": None,
                        "ref": None,
                        "quality": None,
                        "profile": None,
                        "distribution": None,
                        "layout": None,
                        "voice_cloning": clone.get("status", "unknown"),
                        "reason": unavailable,
                        "target_discovery": "unavailable",
                        "contract_status": contract.contract_status,
                    }
                )
            continue
        for item in sorted(items, key=lambda candidate: candidate.id):
            distributions = _item_choices(item, "distribution_choices")
            selected: tuple[str | None, ...]
            if name == "kokoro" and len(distributions) > 1:
                selected = distributions
            else:
                declared = _catalog_metadata(item).get("distribution_id")
                selected = (declared,) if isinstance(declared, str) else (None,)
            for dist in selected:
                try:
                    selected_item = manager.catalog.resolve(
                        item.ref, refresh=refresh, distribution=dist
                    )
                except (OnnxVoiceError, ValueError, OSError):
                    selected_item = item
                variant = _variant_response(
                    manager, contract, selected_item, name, item.id, quality=None, distribution=dist
                )
                clone = variant["capabilities"].get("voice_cloning", {})
                if capability == "voice-cloning" and clone.get("supported") is not True:
                    continue
                rows.append(
                    {
                        "engine": name,
                        "target": item.id,
                        "ref": item.ref,
                        "quality": variant.get("quality"),
                        "profile": variant.get("quality"),
                        "quality_choices": list(_item_choices(item, "profiles")),
                        "distribution": variant.get("distribution"),
                        "layout": variant.get("layout"),
                        "sample_rate": variant.get("sample_rate"),
                        "license_spdx": variant.get("license", {}).get("spdx")
                        if isinstance(variant.get("license"), Mapping)
                        else None,
                        "commercial_use": variant.get("license", {}).get("commercial_use")
                        if isinstance(variant.get("license"), Mapping)
                        else None,
                        "license": variant.get("license"),
                        "voice_cloning": clone.get("status", "unknown"),
                        "cloning_mode": clone.get("mode"),
                        "reason": clone.get("reason"),
                        "contract_status": variant["contract_status"],
                        "target_discovery": "available",
                    }
                )
    rows.sort(
        key=lambda row: (
            row["engine"],
            row["target"] or "",
            row["distribution"] or "",
            row["profile"] or "",
        )
    )
    return {
        "schema_version": 1,
        "capability_filter": capability,
        "target_discovery": "partial" if unavailable_engines else "available",
        "engines": sorted(engines),
        "catalog_unavailable": unavailable_engines,
        "rows": rows,
    }
