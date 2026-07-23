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
        from image_search.processors.text_embed import SentenceTransformerProcessor

        _CONSTRUCTORS["ocr"] = RapidOcrProcessor
        _CONSTRUCTORS["text_embed"] = SentenceTransformerProcessor
        _CONSTRUCTORS["image_embed"] = SiglipImageEmbedProcessor
        _CONSTRUCTORS["caption"] = MoondreamCaptionProcessor
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
            self._instances[key] = constructors[kind](model_id)
        return self._instances[key]

    def for_folder(self, folder_key: str) -> list[tuple[str, Processor]]:
        """Processors enabled for a folder, in dispatch order (PROCESSOR_KEYS
        already orders text producers before text_embed)."""
        folder = self._config.folders[folder_key]
        out = []
        for kind in PROCESSOR_KEYS:
            model_id = folder.enabled(kind)
            if model_id:
                out.append((kind, self.get(kind, model_id)))
        return out
