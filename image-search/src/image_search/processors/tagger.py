"""Zero-shot image-type tagger (spec's `clip-zs` tag source).

The image_embed model is a joint text-image model, so classifying an image
costs one dot product against a handful of label prompts embedded once at
startup — and no extra vision forward pass at all when the tagger reuses the
vector image_embed already computed for the same image.

Its output serves two purposes: content-based routing (a meme in a mixed
folder gets OCR; a photo gets captioned instead) and search facets
("unemployment graphs" -> filter to charts, rank on "unemployment").
"""

from __future__ import annotations

from image_search.processors.base import LoadedImage, Record, TagRecord

SOURCE = "clip-zs"

# Short tag -> prompt sentence. Prompts (not bare words) score noticeably
# better with CLIP-family text towers.
LABEL_PROMPTS: dict[str, str] = {
    "meme": "a meme or comic with caption text",
    "chart": "a chart, graph or data plot",
    "screenshot": "a screenshot of a computer app or website",
    "document": "a scanned document or page of text",
    "photo": "a photograph of a place or person",
    "art": "digital art or a drawing",
}


class ClipZeroShotTagger:
    """Scores each image against LABEL_PROMPTS using the folder's image_embed
    model. Emits one TagRecord per label; ranking (not an absolute cosine
    threshold) is what downstream routing and filtering use, because cosine
    magnitudes are not comparable across images."""

    kind = "tagger"

    def __init__(self, model_id: str, embedder=None) -> None:
        self.model_id = model_id
        # Shared with the image_embed processor via the registry cache, so the
        # weights are loaded once for both.
        self._embedder = embedder
        self._label_vectors: list[list[float]] | None = None

    def load(self) -> None:
        if self._embedder is None:
            from image_search.processors.image_embed import SiglipImageEmbedProcessor

            self._embedder = SiglipImageEmbedProcessor(self.model_id)
        self._embedder.load()
        if self._label_vectors is None:
            self._label_vectors = self._embedder.embed_text(list(LABEL_PROMPTS.values()))

    def process(self, img: LoadedImage) -> list[Record]:
        self.load()
        vector = img.image_vector
        if vector is None:
            # Folder has image_embed off (or it ran after us) — embed here.
            vector = self._embedder.embed(img.path)

        scores = [
            (tag, sum(a * b for a, b in zip(vector, label_vec)))
            for tag, label_vec in zip(LABEL_PROMPTS, self._label_vectors)
        ]
        return [TagRecord(tag=tag, score=score, source=SOURCE) for tag, score in scores]
