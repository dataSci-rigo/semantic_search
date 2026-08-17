from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol

Kind = Literal[
    "ocr", "caption", "text_embed", "image_embed", "faces", "tagger", "layout", "topic_kw"
]

# Kinds that produce the raw text a text_embed processor consumes.
TEXT_PRODUCER_KINDS: tuple[Kind, ...] = ("ocr", "caption", "topic_kw")


@dataclass(frozen=True)
class LoadedImage:
    image_id: str
    path: Path
    width: int
    height: int
    # Text accumulated so far in this dispatch (from ocr/caption/topic_kw),
    # available to text_embed processors that run later in the same pass.
    text: str = ""
    # Image vector produced by image_embed earlier in this dispatch, so the
    # tagger can classify without a second vision forward pass.
    image_vector: list[float] | None = None


@dataclass(frozen=True)
class OcrRecord:
    text: str


@dataclass(frozen=True)
class CaptionRecord:
    text: str


@dataclass(frozen=True)
class TextEmbedRecord:
    model: str
    vector: list[float]


@dataclass(frozen=True)
class ImageEmbedRecord:
    model: str
    vector: list[float]


@dataclass(frozen=True)
class FaceRecord:
    bbox: tuple[int, int, int, int]
    det_score: float
    embedding: list[float]
    model: str
    det_model: str


@dataclass(frozen=True)
class TagRecord:
    tag: str
    score: float
    source: str


Record = (
    OcrRecord | CaptionRecord | TextEmbedRecord | ImageEmbedRecord | FaceRecord | TagRecord
)


class Processor(Protocol):
    kind: Kind
    model_id: str

    def load(self) -> None:
        """Heavy init (load model weights). Called lazily on first use."""
        ...

    def process(self, img: LoadedImage) -> list[Record]:
        ...
