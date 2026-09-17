from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from ..errors import RuntimeContractError
from ..runtime import OnnxSession
from ..types import AudioResult
from .base import SystemAdapter


class KokoroAdapter(SystemAdapter):
    system = "kokoro"

    @property
    def session(self) -> OnnxSession:
        if self._session is None:
            model = self.installation.artifact("model").path
            self._session = OnnxSession(
                model,
                providers=self.providers,
                provider_options=self.provider_options,
            )
        return self._session

    def _voice_style(self, voice: str, token_count: int) -> np.ndarray:
        try:
            voices_path = self.installation.artifact("voices").path
        except KeyError as exc:
            raise RuntimeContractError("Kokoro installation has no voices archive") from exc
        with np.load(voices_path, allow_pickle=False) as archive:
            if voice not in archive.files:
                raise RuntimeContractError(
                    f"Unknown Kokoro voice {voice!r}. Available: {', '.join(archive.files[:12])}"
                )
            style = np.asarray(archive[voice], dtype=np.float32)
        if style.ndim == 1:
            return style[None, :]
        index = min(max(token_count - 1, 0), style.shape[0] - 1)
        selected = style[index]
        if selected.ndim == 1:
            selected = selected[None, :]
        return np.asarray(selected, dtype=np.float32)

    def infer(
        self,
        tokens: Sequence[int],
        *,
        voice: str | None = None,
        speed: float = 1.0,
        **_: Any,
    ) -> AudioResult:
        token_values = list(tokens)
        voice = voice or self.installation.default_voice
        if not voice:
            raise RuntimeContractError("A Kokoro voice must be provided")
        padded = np.asarray([[0, *token_values, 0]], dtype=np.int64)
        style = self._voice_style(voice, len(token_values))
        names = set(self.session.input_names)
        token_name = "input_ids" if "input_ids" in names else "tokens"
        style_name = "ref_s" if "ref_s" in names else "style"
        inputs = {
            token_name: padded,
            style_name: style,
            "speed": np.asarray([speed], dtype=np.float32),
        }
        missing = set(inputs) - names
        if missing:
            raise RuntimeContractError(
                f"Kokoro model is missing expected inputs: {', '.join(sorted(missing))}"
            )
        outputs = self.session.run(inputs)
        audio = np.squeeze(np.asarray(outputs[0]).T).astype(np.float32, copy=False)
        metadata: dict[str, Any] = {"system": "kokoro", "voice": voice, "speed": speed}
        if len(outputs) > 1:
            metadata["aux_outputs"] = len(outputs) - 1
        return AudioResult(
            audio=audio,
            sample_rate=int(self.installation.sample_rate or 24000),
            metadata=metadata,
        )
