"""The worker bridge must hand the configured model id to the worker script
as argv, so one script can serve several models (moondream2 on GPU boxes,
blip-base on CPU-only ones)."""

import importlib.util
import io
from pathlib import Path

from image_search.processors import subprocess_bridge


def test_bridge_passes_model_id_as_argv(monkeypatch, tmp_path):
    captured = {}

    class FakeProc:
        def __init__(self, argv, **kwargs):
            captured["argv"] = argv
            self.stdin = io.StringIO()
            self.stdout = io.StringIO("READY\n")

    monkeypatch.setattr(subprocess_bridge.subprocess, "Popen", FakeProc)
    monkeypatch.setenv("CONDA_EXE", "/opt/conda/bin/conda")
    worker = tmp_path / "worker.py"
    worker.write_text("print('READY')\n")

    class Bridged(subprocess_bridge.SubprocessBridgeProcessor):
        kind = "caption"
        worker_script = worker
        conda_env = "some-env"

    proc = Bridged("blip-base")
    proc.load()
    # CONDA_EXE, not bare "conda": Windows' activated conda is a batch
    # function CreateProcess can't exec.
    assert captured["argv"][0] == "/opt/conda/bin/conda"
    assert captured["argv"][-1] == "blip-base"
    assert captured["argv"][-2] == str(worker)


def test_bridge_missing_conda_raises_clear_error(monkeypatch, tmp_path):
    monkeypatch.delenv("CONDA_EXE", raising=False)
    monkeypatch.setattr(subprocess_bridge.shutil, "which", lambda _: None)
    worker = tmp_path / "worker.py"
    worker.write_text("print('READY')\n")

    class Bridged(subprocess_bridge.SubprocessBridgeProcessor):
        kind = "ocr"
        worker_script = worker
        conda_env = "some-env"

    import pytest

    with pytest.raises(RuntimeError, match="CONDA_EXE"):
        Bridged("m").load()


def test_close_tolerates_dead_worker(monkeypatch, tmp_path):
    """A worker that already died leaves a broken pipe; close() runs at phase
    boundaries mid-ingest and must clean up without raising."""

    class DeadStdin:
        def close(self):
            raise BrokenPipeError(32, "Broken pipe")

    class FakeProc:
        def __init__(self, argv, **kwargs):
            self.stdin = DeadStdin()
            self.stdout = io.StringIO("READY\n")

        def terminate(self):
            pass

        def wait(self, timeout=None):
            return 0

    monkeypatch.setattr(subprocess_bridge.subprocess, "Popen", FakeProc)
    monkeypatch.setenv("CONDA_EXE", "/opt/conda/bin/conda")
    worker = tmp_path / "worker.py"
    worker.write_text("print('READY')\n")

    class Bridged(subprocess_bridge.SubprocessBridgeProcessor):
        kind = "ocr"
        worker_script = worker
        conda_env = "some-env"

    proc = Bridged("m")
    proc.load()
    proc.close()  # must not raise
    assert proc._proc is None


def _fake_proc_class(captured, ready=True, alive=True):
    class FakeProc:
        def __init__(self, argv, **kwargs):
            captured.setdefault("launches", 0)
            captured["launches"] += 1
            self.stdin = io.StringIO()
            self.stdout = io.StringIO("READY\n" if ready else "")

        def poll(self):
            return None if alive else 1

        def terminate(self):
            pass

        def kill(self):
            pass

        def wait(self, timeout=None):
            return 0

    return FakeProc


def test_dead_worker_relaunches_on_next_call(monkeypatch, tmp_path):
    """One worker crash costs one file: the next load() detects the corpse
    and relaunches instead of writing into a broken pipe forever."""
    captured = {}
    monkeypatch.setattr(
        subprocess_bridge.subprocess, "Popen", _fake_proc_class(captured, alive=False)
    )
    monkeypatch.setenv("CONDA_EXE", "/opt/conda/bin/conda")
    worker = tmp_path / "worker.py"
    worker.write_text("print('READY')\n")

    class Bridged(subprocess_bridge.SubprocessBridgeProcessor):
        kind = "ocr"
        worker_script = worker
        conda_env = "some-env"

    proc = Bridged("m")
    proc.load()
    assert captured["launches"] == 1
    proc.load()  # poll() says dead -> close + relaunch
    assert captured["launches"] == 2


def test_failed_starts_stop_retrying_after_cap(monkeypatch, tmp_path):
    """A worker that never READYs (bad env) fails fast after a few attempts
    instead of relaunch-storming through a huge run."""
    import pytest

    captured = {}
    monkeypatch.setattr(
        subprocess_bridge.subprocess, "Popen", _fake_proc_class(captured, ready=False)
    )
    monkeypatch.setenv("CONDA_EXE", "/opt/conda/bin/conda")
    worker = tmp_path / "worker.py"
    worker.write_text("print('nope')\n")

    class Bridged(subprocess_bridge.SubprocessBridgeProcessor):
        kind = "caption"
        worker_script = worker
        conda_env = "missing-env"

    proc = Bridged("m")
    for _ in range(subprocess_bridge.SubprocessBridgeProcessor.MAX_START_FAILURES):
        with pytest.raises(RuntimeError, match="expected READY"):
            proc.load()
    assert captured["launches"] == proc.MAX_START_FAILURES
    with pytest.raises(RuntimeError, match="not retrying"):
        proc.load()
    assert captured["launches"] == proc.MAX_START_FAILURES  # no further spawns


def test_caption_worker_knows_both_models():
    script = (
        Path(__file__).resolve().parents[1] / "scripts" / "caption_worker.py"
    )
    spec = importlib.util.spec_from_file_location("caption_worker", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert set(module.LOADERS) == {"moondream2", "blip-base"}
