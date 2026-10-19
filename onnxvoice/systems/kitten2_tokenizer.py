"""Direct execution of the KittenTTS 2 bundled tokenizer."""

from __future__ import annotations

from pathlib import Path

from ..errors import OptionalDependencyError, RuntimeContractError


class Kitten2Tokenizer:
    def __init__(self, tokenizer_json: str | Path) -> None:
        path = Path(tokenizer_json)
        if not path.is_file():
            raise RuntimeContractError(
                f"Invalid Kitten2 bundle: required tokenizer artifact is missing: {path}"
            )
        try:
            from tokenizers import Tokenizer
        except ImportError as exc:
            raise OptionalDependencyError(
                "Kitten2 text synthesis requires onnxvoice[kitten2]."
            ) from exc
        self._tokenizer = Tokenizer.from_file(str(path))

    def encode(self, text: str) -> list[int]:
        return list(self._tokenizer.encode(text, add_special_tokens=False).ids)
