from __future__ import annotations

import hashlib
from pathlib import Path

from .errors import IntegrityError


def digest_file(path: Path, algorithm: str = "sha256", chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_file(
    path: Path,
    *,
    expected_size: int | None = None,
    sha256: str | None = None,
    md5: str | None = None,
) -> None:
    if expected_size is not None and path.stat().st_size != expected_size:
        raise IntegrityError(
            f"Size mismatch for {path.name}: expected {expected_size}, got {path.stat().st_size}"
        )
    if sha256 is not None:
        actual = digest_file(path, "sha256")
        if actual.lower() != sha256.lower():
            raise IntegrityError(f"SHA-256 mismatch for {path.name}")
    if md5 is not None:
        actual = digest_file(path, "md5")
        if actual.lower() != md5.lower():
            raise IntegrityError(f"MD5 mismatch for {path.name}")
