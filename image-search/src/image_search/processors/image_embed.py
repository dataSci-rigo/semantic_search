from __future__ import annotations

from image_search.processors.base import ImageEmbedRecord, LoadedImage, Record

# HuggingFace repo ids, keyed by the short names used in config.
MODEL_REPOS = {
    "siglip2-base": "google/siglip2-base-patch16-224",
}


class SiglipImageEmbedProcessor:
    kind = "image_embed"

    def __init__(self, model_id: str = "siglip2-base") -> None:
        self.model_id = model_id
        self._model = None
        self._processor = None
        self._device = None

    def load(self) -> None:
        if self._model is not None:
            return
        import torch

        # This GPU (Pascal, sm_61) hits "unable to find an engine" from cuDNN's
        # conv2d algorithm search — a real cuDNN9/Pascal incompatibility, not a
        # perf tradeoff. Disabling cuDNN falls back to torch's native conv
        # kernels, which do work. See docs/gpu-setup.md. Harmless for
        # text_embed's attention/matmul-only path in the same process.
        torch.backends.cudnn.enabled = False

        from transformers import AutoModel, AutoProcessor

        repo = MODEL_REPOS.get(self.model_id, self.model_id)
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._processor = AutoProcessor.from_pretrained(repo)
        self._model = AutoModel.from_pretrained(repo).to(self._device).eval()

    def embed(self, path) -> list[float]:
        """Embed an image file (used for both indexing and query-by-image)."""
        self.load()
        import torch
        from PIL import Image

        img = Image.open(path).convert("RGB")
        inputs = self._processor(images=img, return_tensors="pt").to(self._device)
        with torch.no_grad():
            out = self._model.get_image_features(**inputs)
        feats = out.pooler_output
        feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats[0].cpu().tolist()

    def process(self, img: LoadedImage) -> list[Record]:
        return [ImageEmbedRecord(model=self.model_id, vector=self.embed(img.path))]
