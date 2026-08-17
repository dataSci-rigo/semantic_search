from __future__ import annotations

import os

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
        # Reduced-precision weights for RAM-constrained machines (e.g.
        # "bfloat16" halves the fp32 footprint; needed on <4GB WSL hosts).
        dtype_name = os.environ.get("IMAGE_SEARCH_TORCH_DTYPE")
        self._dtype = getattr(torch, dtype_name) if dtype_name else None
        kwargs = {"dtype": self._dtype} if self._dtype else {}
        self._processor = AutoProcessor.from_pretrained(repo)
        self._model = AutoModel.from_pretrained(repo, **kwargs).to(self._device).eval()

    @staticmethod
    def _pooled(out):
        """transformers <5 returns an output object with pooler_output; 5.x
        returns the pooled tensor directly. Applies to both towers."""
        import torch

        return out if isinstance(out, torch.Tensor) else out.pooler_output

    def embed(self, path) -> list[float]:
        """Embed an image file (used for both indexing and query-by-image)."""
        self.load()
        import torch
        from PIL import Image

        img = Image.open(path).convert("RGB")
        inputs = self._processor(images=img, return_tensors="pt").to(self._device)
        if self._dtype is not None:
            inputs = inputs.to(self._dtype)  # casts floating tensors only
        with torch.no_grad():
            out = self._model.get_image_features(**inputs)
        feats = self._pooled(out)
        feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats[0].float().cpu().tolist()

    def embed_text(self, texts: list[str]) -> list[list[float]]:
        """Embed text with the same model's text tower, into the same space as
        embed(). Used for zero-shot tagging (processors/tagger.py); these
        vectors are only ever compared against this model's image vectors."""
        self.load()
        import torch

        inputs = self._processor(text=texts, return_tensors="pt", padding=True).to(self._device)
        with torch.no_grad():
            out = self._model.get_text_features(**inputs)
        feats = self._pooled(out)
        feats = feats / feats.norm(dim=-1, keepdim=True)
        return [row.float().cpu().tolist() for row in feats]

    def process(self, img: LoadedImage) -> list[Record]:
        return [ImageEmbedRecord(model=self.model_id, vector=self.embed(img.path))]
