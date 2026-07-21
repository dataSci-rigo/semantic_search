from __future__ import annotations

from image_search.processors.base import LoadedImage, Record, TextEmbedRecord

# sentence-transformers model repo ids, keyed by the short names used in config.
MODEL_REPOS = {
    "bge-small-en-v1.5": "BAAI/bge-small-en-v1.5",
    "all-MiniLM-L6-v2": "sentence-transformers/all-MiniLM-L6-v2",
}


class SentenceTransformerProcessor:
    kind = "text_embed"

    def __init__(self, model_id: str = "bge-small-en-v1.5") -> None:
        self.model_id = model_id
        self._model = None

    def load(self) -> None:
        if self._model is not None:
            return
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "sentence-transformers is not installed. Install it into the active "
                "environment (`sem_search_gpu`) to use text_embed: "
                "pip install sentence-transformers"
            ) from exc
        repo = MODEL_REPOS.get(self.model_id, self.model_id)
        self._model = SentenceTransformer(repo)

    def embed(self, text: str) -> list[float]:
        """Embed arbitrary text (used for both indexing and query time)."""
        self.load()
        return self._model.encode(text, normalize_embeddings=True).tolist()

    def process(self, img: LoadedImage) -> list[Record]:
        if not img.text.strip():
            return []
        return [TextEmbedRecord(model=self.model_id, vector=self.embed(img.text))]
