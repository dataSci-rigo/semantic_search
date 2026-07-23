from __future__ import annotations

from pathlib import Path

from image_search.processors.base import CaptionRecord, LoadedImage, Record
from image_search.processors.subprocess_bridge import SubprocessBridgeProcessor


class MoondreamCaptionProcessor(SubprocessBridgeProcessor):
    """Captioning via Moondream2, run in a separate conda env
    (`sem_search_caption`) as a persistent subprocess. Moondream2's
    trust_remote_code model class doesn't load under the `transformers`
    version `sem_search_gpu` needs for native SigLIP2/sentence-transformers
    support — so captioning runs out-of-process. See docs/gpu-setup.md."""

    kind = "caption"
    worker_script = Path(__file__).resolve().parents[3] / "scripts" / "caption_worker.py"
    conda_env = "sem_search_caption"

    def __init__(self, model_id: str = "moondream2") -> None:
        super().__init__(model_id)

    def process(self, img: LoadedImage) -> list[Record]:
        return [CaptionRecord(text=self._call(img.path))]
