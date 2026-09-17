from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

import numpy as np

from ..errors import RuntimeContractError
from ..runtime import OnnxSession
from ..types import AudioResult
from .base import SystemAdapter


class PiperAdapter(SystemAdapter):
    system = "piper"

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

    def _config(self) -> dict[str, Any]:
        try:
            path = self.installation.artifact("config").path
        except KeyError:
            return {}
        return json.loads(path.read_text(encoding="utf-8"))

    def infer(
        self,
        tokens: Sequence[int],
        *,
        speaker: int | None = None,
        noise_scale: float = 0.667,
        length_scale: float = 1.0,
        noise_w: float = 0.8,
        **_: Any,
    ) -> AudioResult:
        ids = np.asarray([list(tokens)], dtype=np.int64)
        inputs: dict[str, np.ndarray] = {
            "input": ids,
            "input_lengths": np.asarray([ids.shape[1]], dtype=np.int64),
            "scales": np.asarray([noise_scale, length_scale, noise_w], dtype=np.float32),
        }
        if "sid" in self.session.input_names:
            if speaker is None:
                speaker = 0
            inputs["sid"] = np.asarray([speaker], dtype=np.int64)
        missing = set(inputs) - set(self.session.input_names)
        if missing:
            raise RuntimeContractError(
                f"Piper model is missing expected inputs: {', '.join(sorted(missing))}"
            )
        outputs = self.session.run(inputs)
        audio = np.squeeze(np.asarray(outputs[0])).astype(np.float32, copy=False)
        config = self._config()
        sample_rate = int(
            self.installation.sample_rate
            or (config.get("audio") or {}).get("sample_rate")
            or config.get("sample_rate")
            or 22050
        )
        return AudioResult(
            audio=audio,
            sample_rate=sample_rate,
            metadata={"system": "piper", "speaker": speaker},
        )
