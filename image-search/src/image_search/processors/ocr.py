from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from image_search.processors.base import LoadedImage, OcrRecord, Record

WORKER_SCRIPT = Path(__file__).resolve().parents[3] / "scripts" / "ocr_worker.py"
WORKER_CONDA_ENV = "sem_search_ocr"


class RapidOcrProcessor:
    """OCR via RapidOCR, run in a separate conda env (`sem_search_ocr`) as a
    persistent subprocess. RapidOCR's GPU execution provider needs cuDNN 8,
    which conflicts with the cuDNN 9 this project's main env (`sem_search_gpu`)
    needs for torch/sentence-transformers — so OCR runs out-of-process rather
    than sharing a Python environment. See docs/gpu-setup.md."""

    kind = "ocr"

    def __init__(self, model_id: str = "rapidocr") -> None:
        self.model_id = model_id
        self._proc: subprocess.Popen | None = None

    def load(self) -> None:
        if self._proc is not None:
            return
        if not WORKER_SCRIPT.exists():
            raise RuntimeError(f"OCR worker script not found at {WORKER_SCRIPT}")

        self._proc = subprocess.Popen(
            ["conda", "run", "-n", WORKER_CONDA_ENV, "--no-capture-output",
             "python", str(WORKER_SCRIPT)],
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
                f"OCR worker failed to start (env={WORKER_CONDA_ENV!r}): "
                f"expected READY, got {ready_line!r}"
            )

    def process(self, img: LoadedImage) -> list[Record]:
        self.load()
        assert self._proc is not None and self._proc.stdin is not None
        assert self._proc.stdout is not None

        self._proc.stdin.write(str(img.path) + "\n")
        self._proc.stdin.flush()
        response_line = self._proc.stdout.readline()
        if not response_line:
            raise RuntimeError("OCR worker process exited unexpectedly")

        response = json.loads(response_line)
        if "error" in response:
            raise RuntimeError(f"OCR worker error on {img.path}: {response['error']}")
        return [OcrRecord(text=response["text"])]

    def close(self) -> None:
        if self._proc is not None:
            self._proc.stdin.close()
            self._proc.terminate()
            self._proc = None
