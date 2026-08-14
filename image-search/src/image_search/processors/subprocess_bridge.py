from __future__ import annotations

import json
import os
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

    def __init__(self, model_id: str) -> None:
        self.model_id = model_id
        self._proc: subprocess.Popen | None = None

    def env_name(self) -> str:
        """Conda env to run the worker in. `IMAGE_SEARCH_<KIND>_ENV` overrides
        the class default, so a machine that keeps the deps somewhere else
        (e.g. a CPU-only box with no cuDNN conflict to work around) can point
        at its own env without editing code."""
        return os.environ.get(f"IMAGE_SEARCH_{self.kind.upper()}_ENV", self.conda_env)

    def load(self) -> None:
        if self._proc is not None:
            return
        if not self.worker_script.exists():
            raise RuntimeError(f"Worker script not found at {self.worker_script}")

        self._proc = subprocess.Popen(
            ["conda", "run", "-n", self.env_name(), "--no-capture-output",
             "python", str(self.worker_script)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=sys.stderr,
            text=True,
            bufsize=1,
        )
        ready_line = self._proc.stdout.readline()
        if ready_line.strip() != "READY":
            self._proc.kill()
            raise RuntimeError(
                f"Worker failed to start (env={self.env_name()!r}): "
                f"expected READY, got {ready_line!r}"
            )

    def _call(self, path) -> str:
        """Send one image path to the worker, return its "text" response."""
        self.load()
        assert self._proc is not None and self._proc.stdin is not None
        assert self._proc.stdout is not None

        self._proc.stdin.write(str(path) + "\n")
        self._proc.stdin.flush()
        response_line = self._proc.stdout.readline()
        if not response_line:
            raise RuntimeError(
                f"Worker process ({self.conda_env}) exited unexpectedly"
            )

        response = json.loads(response_line)
        if "error" in response:
            raise RuntimeError(f"Worker error ({self.conda_env}) on {path}: {response['error']}")
        return response["text"]

    def close(self) -> None:
        if self._proc is not None:
            self._proc.stdin.close()
            self._proc.terminate()
            try:
                self._proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self._proc.kill()
                self._proc.wait()
            self._proc = None
