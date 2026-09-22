from __future__ import annotations

from pathlib import Path

from image_search.processors.base import CaptionRecord, LoadedImage, Record
from image_search.processors.subprocess_bridge import SubprocessBridgeProcessor


class MoondreamCaptionProcessor(SubprocessBridgeProcessor):
    """Captioning via a persistent worker subprocess in a separate conda env
    (`sem_search_caption`; override with IMAGE_SEARCH_CAPTION_ENV). The model
    id is passed to the worker, which knows "moondream2" (GPU default; its
    trust_remote_code class doesn't load under the `transformers` version
    `sem_search_gpu` needs — see docs/gpu-setup.md) and "blip-base"
    (Salesforce BLIP, plain transformers, runs on CPU-only machines)."""

    kind = "caption"
    worker_script = Path(__file__).resolve().parents[3] / "scripts" / "caption_worker.py"
    conda_env = "sem_search_caption"

    def __init__(self, model_id: str = "moondream2") -> None:
        super().__init__(model_id)

    def process(self, img: LoadedImage) -> list[Record]:
        return [CaptionRecord(text=self._call(img.path))]
