from __future__ import annotations

import sys
import time
from collections import deque
from collections.abc import Callable
from typing import TYPE_CHECKING, TextIO

if TYPE_CHECKING:
    from .types import AssetProgress


class InstallProgress:
    """Render shared asset-install events to stderr."""

    def __init__(
        self,
        ref: str,
        *,
        stream: TextIO | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.ref = ref
        self.stream = stream if stream is not None else sys.stderr
        self._clock = clock
        self._tty = self.stream.isatty()
        self._closed = False
        self._last_render = self._clock()
        self._last_tty_line_length = 0
        self._total_bytes: int | None = None
        self._completed_bytes = 0
        self._current_artifact: str | None = None
        self._current_total: int | None = None
        self._current_received = 0
        self._last_file_completed = 0
        self._file_count = 0
        self._file_count_total: int | None = None
        self._finalized_artifacts: set[str] = set()
        self._network_bytes = 0
        self._rate_samples: deque[tuple[float, int]] = deque()
        self._status = "Installing"
        self._write_line(f"Resolving {ref}")

    def __call__(self, event: AssetProgress) -> None:
        if self._closed:
            return

        if event.phase == "install_started":
            self._total_bytes = self._known_size(event.total)
            self._status = "Installing"
            self._render(force=True)
        elif event.phase == "download_started":
            self._current_artifact = event.artifact
            self._current_total = self._known_size(event.total)
            self._current_received = 0
            self._last_file_completed = 0
            self._status = "Downloading"
            self._render(force=True)
        elif event.phase == "download_progress":
            self._record_download_progress(event)
            self._render()
        elif event.phase == "download_completed":
            self._finish_download(event)
            self._status = "Installing"
            self._render(force=True)
        elif event.phase == "artifact_cached":
            self._record_cached_artifact(event)
            self._status = "Installing"
            self._render(force=True)
        elif event.phase == "artifact_installed":
            if type(event.completed) is int and event.completed >= 0:
                self._file_count = max(self._file_count, event.completed)
            self._file_count_total = self._known_size(event.total)
            self._status = "Installing"
            self._render(force=self._tty)
        elif event.phase == "install_completed":
            self._current_artifact = None
            self._current_total = None
            self._current_received = 0
            self._status = "Installed"
            self._render(force=True)
        elif event.phase == "install_failed":
            self._status = "Install failed"
            self._render(force=True)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._tty and self._last_tty_line_length:
            self.stream.write("\n")
            self.stream.flush()

    def _record_download_progress(self, event: AssetProgress) -> None:
        if self._current_artifact != event.artifact:
            self._current_artifact = event.artifact
            self._current_total = self._known_size(event.total)
            self._current_received = 0
            self._last_file_completed = 0
        elif self._current_total is None:
            self._current_total = self._known_size(event.total)

        completed = self._known_size(event.completed)
        if completed is None:
            return
        new_completed = max(self._last_file_completed, completed)
        delta = new_completed - self._last_file_completed
        self._last_file_completed = new_completed
        self._current_received += delta
        if delta:
            now = self._clock()
            self._network_bytes += delta
            self._rate_samples.append((now, self._network_bytes))
            while len(self._rate_samples) > 2 and now - self._rate_samples[1][0] > 20:
                self._rate_samples.popleft()

    def _finish_download(self, event: AssetProgress) -> None:
        artifact = event.artifact or self._current_artifact
        if artifact is None or artifact in self._finalized_artifacts:
            return

        received = self._current_received if self._current_artifact == artifact else 0
        reported = self._known_size(event.completed)
        if reported is not None:
            received = max(received, reported)
        self._completed_bytes += received
        self._finalized_artifacts.add(artifact)
        if self._current_artifact == artifact:
            self._current_artifact = None
            self._current_total = None
            self._current_received = 0
            self._last_file_completed = 0

    def _record_cached_artifact(self, event: AssetProgress) -> None:
        artifact = event.artifact
        if artifact is None or artifact in self._finalized_artifacts:
            return
        total = self._known_size(event.total)
        if total is not None:
            self._completed_bytes += total
        self._finalized_artifacts.add(artifact)

    def _render(self, *, force: bool = False) -> None:
        now = self._clock()
        interval = 0.12 if self._tty else 5.0
        if not force and now - self._last_render < interval:
            return
        self._last_render = now

        downloaded = self._completed_bytes + self._current_received
        if self._total_bytes is None:
            byte_progress = f"{self._format_bytes(downloaded)} downloaded (total unknown)"
        elif self._total_bytes > 0:
            percent = min(100, downloaded * 100 // self._total_bytes)
            if self._tty:
                width = 20
                filled = percent * width // 100
                bar = "#" * filled + "-" * (width - filled)
                byte_progress = (
                    f"[{bar}] {percent}% {self._format_bytes(downloaded)} / "
                    f"{self._format_bytes(self._total_bytes)}"
                )
            else:
                byte_progress = (
                    f"{percent}% {self._format_bytes(downloaded)} / "
                    f"{self._format_bytes(self._total_bytes)}"
                )
        else:
            byte_progress = f"{self._format_bytes(downloaded)} / 0 B"

        details = [byte_progress]
        if self._current_artifact is not None:
            file_progress = self._format_bytes(self._current_received)
            if self._current_total is not None:
                file_progress += f" / {self._format_bytes(self._current_total)}"
            details.append(f"file {file_progress}")
            details.append(f"current {self._current_artifact}")

        if self._file_count_total is None:
            details.append(f"files {self._file_count}")
        else:
            details.append(f"files {self._file_count}/{self._file_count_total}")

        rate = self._network_rate()
        if rate is not None:
            details.append(f"{self._format_rate(rate)}")
            if self._total_bytes is not None:
                remaining = max(0, self._total_bytes - downloaded)
                details.append(f"ETA {self._format_duration(remaining / rate)}")

        line = f"{self._status} {self.ref}: " + ", ".join(details)
        if self._tty:
            padding = " " * max(0, self._last_tty_line_length - len(line))
            self.stream.write("\r" + line + padding)
            self._last_tty_line_length = len(line)
        else:
            self.stream.write(line + "\n")
        self.stream.flush()

    def _network_rate(self) -> float | None:
        if len(self._rate_samples) < 2:
            return None
        first_time, first_bytes = self._rate_samples[0]
        last_time, last_bytes = self._rate_samples[-1]
        elapsed = last_time - first_time
        if elapsed <= 0 or last_bytes <= first_bytes:
            return None
        return (last_bytes - first_bytes) / elapsed

    def _write_line(self, line: str) -> None:
        self.stream.write(line + "\n")
        self.stream.flush()

    @staticmethod
    def _known_size(value: int | None) -> int | None:
        if type(value) is int and value >= 0:
            return value
        return None

    @staticmethod
    def _format_bytes(value: int) -> str:
        if value < 1024:
            return f"{value} B"
        amount = float(value)
        for unit in ("KiB", "MiB", "GiB", "TiB"):
            amount /= 1024
            if amount < 1024 or unit == "TiB":
                return f"{amount:.1f} {unit}"
        raise AssertionError("unreachable")

    @classmethod
    def _format_rate(cls, value: float) -> str:
        if value < 1024:
            return f"{value:.0f} B/s"
        return f"{cls._format_bytes(int(value))}/s"

    @staticmethod
    def _format_duration(seconds: float) -> str:
        if seconds < 60:
            return f"{seconds:.0f}s"
        if seconds < 3600:
            return f"{seconds / 60:.0f}m"
        return f"{seconds / 3600:.1f}h"
