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


def test_caption_worker_knows_both_models():
    script = (
        Path(__file__).resolve().parents[1] / "scripts" / "caption_worker.py"
    )
    spec = importlib.util.spec_from_file_location("caption_worker", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert set(module.LOADERS) == {"moondream2", "blip-base"}
