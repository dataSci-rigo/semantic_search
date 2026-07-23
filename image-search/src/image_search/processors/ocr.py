from __future__ import annotations

from pathlib import Path

from image_search.processors.base import LoadedImage, OcrRecord, Record
from image_search.processors.subprocess_bridge import SubprocessBridgeProcessor


class RapidOcrProcessor(SubprocessBridgeProcessor):
    """OCR via RapidOCR, run in a separate conda env (`sem_search_ocr`) as a
    persistent subprocess. RapidOCR's GPU execution provider needs cuDNN 8,
    which conflicts with the cuDNN 9 this project's main env (`sem_search_gpu`)
    needs for torch/sentence-transformers — so OCR runs out-of-process rather
    than sharing a Python environment. See docs/gpu-setup.md."""

    kind = "ocr"
    worker_script = Path(__file__).resolve().parents[3] / "scripts" / "ocr_worker.py"
    conda_env = "sem_search_ocr"

    def __init__(self, model_id: str = "rapidocr") -> None:
        super().__init__(model_id)

    def process(self, img: LoadedImage) -> list[Record]:
        return [OcrRecord(text=self._call(img.path))]
