from __future__ import annotations

from io import StringIO

from onnxvoice._install_progress import InstallProgress
from onnxvoice.types import AssetProgress


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


class TTYStream(StringIO):
    def isatty(self) -> bool:
        return True


def _event(
    phase: str,
    *,
    artifact: str | None = None,
    completed: int | None = None,
    total: int | None = None,
) -> AssetProgress:
    return AssetProgress(
        phase,
        "qwen3:base-0.6b",
        artifact=artifact,
        completed=completed,
        total=total,
    )


def test_redirected_output_is_sparse_plain_and_indeterminate():
    stream = StringIO()
    clock = FakeClock()
    progress = InstallProgress("qwen3:base-0.6b", stream=stream, clock=clock)
    assert stream.getvalue() == "Resolving qwen3:base-0.6b\n"

    progress(_event("install_started"))
    progress(_event("download_started", artifact="model.onnx"))
    clock.advance(1)
    progress(_event("download_progress", artifact="model.onnx", completed=5))
    clock.advance(5)
    progress(_event("download_progress", artifact="model.onnx", completed=10))

    output = stream.getvalue()
    assert "total unknown" in output
    assert "%" not in output
    assert "current model.onnx" in output
    assert "10 B downloaded" in output
    assert "\r" not in output
    assert "\x1b" not in output
    assert len(output.splitlines()) == 4


def test_tty_renders_throttled_bar_and_close_is_idempotent():
    stream = TTYStream()
    clock = FakeClock()
    progress = InstallProgress("qwen3:base-0.6b", stream=stream, clock=clock)

    progress(_event("install_started", total=100))
    progress(_event("download_started", artifact="model.onnx", total=100))
    before_progress = stream.getvalue()
    clock.advance(0.05)
    progress(_event("download_progress", artifact="model.onnx", completed=25, total=100))
    assert stream.getvalue() == before_progress

    clock.advance(0.1)
    progress(_event("download_progress", artifact="model.onnx", completed=25, total=100))
    output = stream.getvalue()
    assert "[#####---------------] 25%" in output
    assert "current model.onnx" in output
    assert "\r" in output
    assert "\x1b" not in output

    progress.close()
    progress.close()
    assert stream.getvalue().endswith("\n")
    assert not stream.getvalue().endswith("\n\n")


def test_cached_and_downloaded_bytes_are_counted_once_without_cache_rate():
    stream = TTYStream()
    clock = FakeClock()
    progress = InstallProgress("qwen3:base-0.6b", stream=stream, clock=clock)

    progress(_event("install_started", total=100))
    progress(_event("artifact_cached", artifact="config.json", total=40))
    progress(_event("download_started", artifact="model.onnx", total=60))
    clock.advance(1)
    progress(_event("download_progress", artifact="model.onnx", completed=20, total=60))
    clock.advance(1)
    progress(_event("download_progress", artifact="model.onnx", completed=20, total=60))
    clock.advance(1)
    progress(_event("download_progress", artifact="model.onnx", completed=50, total=60))

    assert progress._network_bytes == 50
    assert "15 B/s" in stream.getvalue()

    completed = _event("download_completed", artifact="model.onnx", completed=50, total=60)
    progress(completed)
    progress(completed)
    progress(_event("artifact_installed", completed=1, total=2))
    progress(_event("artifact_installed", completed=2, total=2))
    progress(_event("install_completed", total=100))

    final_line = stream.getvalue().split("\r")[-1]
    assert "90 B / 100 B" in final_line
    assert "files 2/2" in final_line
    assert "current model.onnx" not in final_line


def test_cached_artifacts_do_not_create_network_rate_samples():
    stream = TTYStream()
    clock = FakeClock()
    progress = InstallProgress("qwen3:base-0.6b", stream=stream, clock=clock)

    progress(_event("install_started", total=10))
    progress(_event("artifact_cached", artifact="cached.bin", total=5))
    assert progress._network_rate() is None
    assert progress._network_bytes == 0


def test_install_cli_keeps_stdout_machine_readable_and_passes_progress(
    monkeypatch, capsys, tmp_path
):
    from types import SimpleNamespace

    import onnxvoice.cli as cli

    installed_path = tmp_path / "installed-model"
    calls = {}

    class Manager:
        def install(self, ref, **kwargs):
            calls["ref"] = ref
            calls.update(kwargs)
            progress = kwargs["progress"]
            assert isinstance(progress, InstallProgress)
            progress(_event("install_started", total=10))
            progress(_event("install_completed", total=10))
            return SimpleNamespace(path=installed_path)

    monkeypatch.setattr(cli, "_manager", lambda _args: Manager())
    result = cli.main(
        [
            "install",
            "qwen3:base-0.6b",
            "--quality",
            "int4",
            "--distribution",
            "standard",
            "--refresh",
            "--force",
        ]
    )

    captured = capsys.readouterr()
    assert result == 0
    assert captured.out == f"{installed_path}\n"
    assert "Resolving qwen3:base-0.6b" in captured.err
    assert calls["ref"] == "qwen3:base-0.6b"
    assert calls["quality"] == "int4"
    assert calls["distribution"] == "standard"
    assert calls["refresh"] is True
    assert calls["force"] is True


def test_install_cli_no_progress_passes_none_and_writes_no_progress_stderr(
    monkeypatch, capsys, tmp_path
):
    from types import SimpleNamespace

    import onnxvoice.cli as cli

    installed_path = tmp_path / "installed-model"
    calls = {}

    class Manager:
        def install(self, ref, **kwargs):
            calls["progress"] = kwargs["progress"]
            return SimpleNamespace(path=installed_path)

    monkeypatch.setattr(cli, "_manager", lambda _args: Manager())
    result = cli.main(["install", "qwen3:base-0.6b", "--no-progress"])

    captured = capsys.readouterr()
    assert result == 0
    assert captured.out == f"{installed_path}\n"
    assert captured.err == ""
    assert calls["progress"] is None


def test_install_cli_closes_progress_after_failure(monkeypatch, capsys):
    import onnxvoice.cli as cli

    displays = []

    class Display:
        def __init__(self, ref):
            self.ref = ref
            self.close_count = 0
            displays.append(self)

        def close(self):
            self.close_count += 1

    class Manager:
        def install(self, _ref, **_kwargs):
            raise RuntimeError("install failed")

    monkeypatch.setattr(cli, "InstallProgress", Display)
    monkeypatch.setattr(cli, "_manager", lambda _args: Manager())
    result = cli.main(["install", "qwen3:base-0.6b"])

    captured = capsys.readouterr()
    assert result == 2
    assert captured.out == ""
    assert captured.err == "onnxvoice: install failed\n"
    assert len(displays) == 1
    assert displays[0].close_count == 1


def test_install_cli_ctrl_c_returns_130_without_traceback(monkeypatch, capsys):
    import onnxvoice.cli as cli

    displays = []

    class Display:
        def __init__(self, ref):
            self.ref = ref
            self.close_count = 0
            displays.append(self)

        def close(self):
            self.close_count += 1

    class Manager:
        def install(self, _ref, **_kwargs):
            raise KeyboardInterrupt

    monkeypatch.setattr(cli, "InstallProgress", Display)
    monkeypatch.setattr(cli, "_manager", lambda _args: Manager())
    result = cli.main(["install", "qwen3:base-0.6b"])

    captured = capsys.readouterr()
    assert result == 130
    assert captured.out == ""
    assert captured.err == "Cancelled.\n"
    assert "Traceback" not in captured.err
    assert len(displays) == 1
    assert displays[0].close_count == 1
