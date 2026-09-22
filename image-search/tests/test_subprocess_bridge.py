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
    worker = tmp_path / "worker.py"
    worker.write_text("print('READY')\n")

    class Bridged(subprocess_bridge.SubprocessBridgeProcessor):
        kind = "caption"
        worker_script = worker
        conda_env = "some-env"

    proc = Bridged("blip-base")
    proc.load()
    assert captured["argv"][-1] == "blip-base"
    assert captured["argv"][-2] == str(worker)


def test_caption_worker_knows_both_models():
    script = (
        Path(__file__).resolve().parents[1] / "scripts" / "caption_worker.py"
    )
    spec = importlib.util.spec_from_file_location("caption_worker", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert set(module.LOADERS) == {"moondream2", "blip-base"}
