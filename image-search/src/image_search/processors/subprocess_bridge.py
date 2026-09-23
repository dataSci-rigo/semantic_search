from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


class SubprocessBridgeProcessor:
    """Base for processors that run their model in a separate conda env as a
    persistent subprocess, communicating over stdin/stdout (one image path
    in, one JSON line out: `{"text": ...}` or `{"error": ...}`).

    Used when a model's dependencies conflict with what `sem_search_gpu`
    needs (cuDNN version, `transformers` version, etc. — see
    docs/gpu-setup.md for the specific conflicts driving each subclass).
    Subclasses set `worker_script` and `conda_env`, and translate the raw
    text response into their own Record type in `process()`.
    """

    worker_script: Path
    conda_env: str
    # A worker that can't start won't start on retry #4 either; failing fast
    # after this many attempts keeps a misconfigured env (bad
    # IMAGE_SEARCH_<KIND>_ENV, missing deps) from relaunch-storming through
    # a hundred-thousand-file run.
    MAX_START_FAILURES = 3

    def __init__(self, model_id: str) -> None:
        self.model_id = model_id
        self._proc: subprocess.Popen | None = None
        self._start_failures = 0

    def env_name(self) -> str:
        """Conda env to run the worker in. `IMAGE_SEARCH_<KIND>_ENV` overrides
        the class default, so a machine that keeps the deps somewhere else
        (e.g. a CPU-only box with no cuDNN conflict to work around) can point
        at its own env without editing code."""
        return os.environ.get(f"IMAGE_SEARCH_{self.kind.upper()}_ENV", self.conda_env)

    def load(self) -> None:
        if self._proc is not None:
            if self._proc.poll() is None:
                return
            # The worker died since the last call; clean up and relaunch.
            self.close()
        if self._start_failures >= self.MAX_START_FAILURES:
            raise RuntimeError(
                f"Worker for kind={self.kind!r} failed to start "
                f"{self._start_failures} times (env={self.env_name()!r}); "
                "not retrying — check the env name and its dependencies"
            )
        if not self.worker_script.exists():
            raise RuntimeError(f"Worker script not found at {self.worker_script}")

        # CONDA_EXE (set by conda activation on every OS) rather than bare
        # "conda": on Windows the activated "conda" is a batch file/function
        # that CreateProcess can't exec, so Popen(["conda", ...]) raises
        # FileNotFoundError there.
        conda_exe = os.environ.get("CONDA_EXE") or shutil.which("conda")
        if conda_exe is None:
            raise RuntimeError(
                "conda executable not found: activate a conda environment "
                "(so CONDA_EXE is set) or put conda on PATH"
            )
        # The model id rides along as argv so one worker script can serve
        # several models (e.g. moondream2 on GPU boxes, blip-base on CPU).
        # Workers that only know one model ignore it.
        self._proc = subprocess.Popen(
            [conda_exe, "run", "-n", self.env_name(), "--no-capture-output",
             "python", str(self.worker_script), self.model_id],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=sys.stderr,
            text=True,
            bufsize=1,
        )
        ready_line = self._proc.stdout.readline()
        if ready_line.strip() != "READY":
            self._proc.kill()
            self._proc = None  # a corpse here would poison every later call
            self._start_failures += 1
            raise RuntimeError(
                f"Worker failed to start (env={self.env_name()!r}): "
                f"expected READY, got {ready_line!r}"
            )
        self._start_failures = 0

    def _call(self, path) -> str:
        """Send one image path to the worker, return its "text" response.
        A worker that died is cleaned up so the NEXT call relaunches it —
        one crash must cost one file, not the rest of the run."""
        self.load()
        assert self._proc is not None and self._proc.stdin is not None
        assert self._proc.stdout is not None

        try:
            self._proc.stdin.write(str(path) + "\n")
            self._proc.stdin.flush()
            response_line = self._proc.stdout.readline()
        except OSError as exc:
            self.close()
            raise RuntimeError(
                f"Worker ({self.env_name()}) pipe broke on {path}: {exc} "
                "(worker will be relaunched on the next call)"
            ) from exc
        if not response_line:
            self.close()
            raise RuntimeError(
                f"Worker process ({self.env_name()}) exited unexpectedly "
                "(worker will be relaunched on the next call)"
            )

        response = json.loads(response_line)
        if "error" in response:
            raise RuntimeError(f"Worker error ({self.env_name()}) on {path}: {response['error']}")
        return response["text"]

    def close(self) -> None:
        # Must never raise: it runs at phase boundaries mid-ingest, and a
        # worker that already died (crash, OOM kill) leaves a broken pipe
        # behind — that's exactly when cleanup matters most.
        if self._proc is not None:
            try:
                self._proc.stdin.close()
            except OSError:
                pass
            self._proc.terminate()
            try:
                self._proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self._proc.kill()
                self._proc.wait()
            self._proc = None
