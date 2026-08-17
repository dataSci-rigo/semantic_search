from __future__ import annotations

from image_search.config import PROCESSOR_KEYS, SearchConfig
from image_search.processors.base import Processor

# kind -> constructor(model_id). Extend as later phases add processors
# (caption, image_embed, faces, tagger, layout, topic_kw).
_CONSTRUCTORS: dict[str, type] = {}


def _lazy_constructors() -> dict[str, type]:
    if not _CONSTRUCTORS:
        from image_search.processors.caption import MoondreamCaptionProcessor
        from image_search.processors.image_embed import SiglipImageEmbedProcessor
        from image_search.processors.ocr import RapidOcrProcessor
        from image_search.processors.tagger import ClipZeroShotTagger
        from image_search.processors.text_embed import SentenceTransformerProcessor

        _CONSTRUCTORS["ocr"] = RapidOcrProcessor
        _CONSTRUCTORS["text_embed"] = SentenceTransformerProcessor
        _CONSTRUCTORS["image_embed"] = SiglipImageEmbedProcessor
        _CONSTRUCTORS["caption"] = MoondreamCaptionProcessor
        _CONSTRUCTORS["tagger"] = ClipZeroShotTagger
    return _CONSTRUCTORS


class Registry:
    """Maps (kind, model_id) -> Processor, lazy-loaded. Only builds
    processors for (kind, model_id) pairs actually referenced by active
    folders (spec section 6) — never loads a model no folder uses."""

    def __init__(self, config: SearchConfig) -> None:
        self._config = config
        self._instances: dict[tuple[str, str], Processor] = {}

    def get(self, kind: str, model_id: str) -> Processor:
        key = (kind, model_id)
        if key not in self._instances:
            constructors = _lazy_constructors()
            if kind not in constructors:
                raise NotImplementedError(
                    f"No processor implemented for kind={kind!r} yet "
                    f"(model={model_id!r}) — see build phases in the spec."
                )
            if kind == "tagger":
                # Same weights as image_embed: hand it that instance (cached
                # here under its own key) so the model loads once, not twice.
                self._instances[key] = constructors[kind](
                    model_id, embedder=self.get("image_embed", model_id)
                )
            else:
                self._instances[key] = constructors[kind](model_id)
        return self._instances[key]

    def for_processors(self, processors: dict[str, str]) -> list[tuple[str, Processor]]:
        """Resolve a processors dict (kind -> model_id) to instances, in
        dispatch order (PROCESSOR_KEYS already orders text producers before
        text_embed)."""
        out = []
        for kind in PROCESSOR_KEYS:
            model_id = processors.get(kind)
            if model_id:
                out.append((kind, self.get(kind, model_id)))
        return out

    def for_folder(self, folder_key: str) -> list[tuple[str, Processor]]:
        """Processors enabled for a folder's base (non-override) pipeline."""
        return self.for_processors(self._config.folders[folder_key].processors)
