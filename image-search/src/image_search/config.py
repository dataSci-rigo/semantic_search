from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from pathlib import Path

import yaml

# Processor keys read out of a folder's config block, in dispatch order:
#   image_embed -> tagger   : the tagger reuses the image vector, and its tags
#                             gate the processors after it (route: auto).
#   ocr/caption/topic_kw    : text producers, before text_embed which consumes
#                             their output (see base.py TEXT_PRODUCER_KINDS).
PROCESSOR_KEYS = (
    "image_embed",
    "tagger",
    "ocr",
    "caption",
    "topic_kw",
    "text_embed",
    "faces",
    "layout",
)

# Non-processor keys allowed in a folder/override block.
ROUTING_KEYS = ("route", "ocr_when", "caption_when")

# Which zero-shot labels make a processor worth running (labels come from
# processors/tagger.py LABEL_PROMPTS).
#
# OCR is inclusive — top-2 — because label margins are narrow and missing a
# meme's text costs more than OCR-ing a stray photo.
DEFAULT_OCR_WHEN = ("meme", "chart", "screenshot", "document")
# Captioning runs on everything, text-bearing images included: a meme gets a
# caption *and* OCR *and* an image vector, so it is findable by what it says,
# what it looks like, and what it depicts. Narrow this list to trade recall
# for indexing time.
DEFAULT_CAPTION_WHEN = ("meme", "chart", "screenshot", "document", "photo", "art")
OCR_RANK_CUTOFF = 2
CAPTION_RANK_CUTOFF = 6  # i.e. any label — caption is never gated out by default


@dataclass(frozen=True)
class Routing:
    """Per-image (rather than per-folder) processor gating, driven by the
    tagger's zero-shot labels. `auto` off => every configured processor runs
    on every file, exactly as before."""

    auto: bool = False
    ocr_when: tuple[str, ...] = DEFAULT_OCR_WHEN
    caption_when: tuple[str, ...] = DEFAULT_CAPTION_WHEN

    def wants(self, kind: str, ranked: list[tuple[str, float, int]]) -> bool:
        """Should `kind` run, given this image's [(tag, score, rank)]?"""
        if not self.auto or not ranked:
            return True
        if kind == "ocr":
            labels, cutoff = self.ocr_when, OCR_RANK_CUTOFF
        elif kind == "caption":
            labels, cutoff = self.caption_when, CAPTION_RANK_CUTOFF
        else:
            return True
        return any(tag in labels and rank <= cutoff for tag, _, rank in ranked)


@dataclass(frozen=True)
class FolderConfig:
    path: Path
    processors: dict[str, str]  # kind -> model_id, "off"/absent entries excluded
    # dir name (case-insensitive) -> processors, for subtrees that need a
    # different pipeline than their siblings (e.g. nested "Screenshots" dirs
    # scattered inside otherwise-photo folders).
    overrides: dict[str, dict[str, str]] = field(default_factory=dict)
    routing: Routing = field(default_factory=Routing)

    def enabled(self, kind: str) -> str | None:
        return self.processors.get(kind)

    def processors_for_path(self, file_path: Path) -> dict[str, str]:
        parts_lower = {p.lower() for p in file_path.parts}
        for name, processors in self.overrides.items():
            if name.lower() in parts_lower:
                return processors
        return self.processors


@dataclass(frozen=True)
class SearchConfig:
    folders: dict[str, FolderConfig] = field(default_factory=dict)

    def active_processors(self) -> set[tuple[str, str]]:
        """Union of (kind, model_id) referenced by any active folder,
        including path overrides."""
        out: set[tuple[str, str]] = set()
        for folder in self.folders.values():
            out.update(folder.processors.items())
            for override_processors in folder.overrides.values():
                out.update(override_processors.items())
        return out


def _is_off(value: object) -> bool:
    # YAML parses bare `off`/`no`/`false` as bool False, not the string "off".
    if value is None or value is False:
        return True
    return isinstance(value, str) and value.strip().lower() == "off"


def _parse_routing(block: dict, context: str) -> Routing:
    route = block.get("route")
    auto = isinstance(route, str) and route.strip().lower() == "auto"
    if route is not None and not auto:
        raise ValueError(
            f"route in {context} must be 'auto' (or omitted), got {route!r}"
        )

    def labels(key: str, default: tuple[str, ...]) -> tuple[str, ...]:
        value = block.get(key)
        if value is None:
            return default
        if isinstance(value, str):
            value = [value]
        if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
            raise ValueError(f"{key} in {context} must be a list of label names, got {value!r}")
        return tuple(value)

    return Routing(
        auto=auto,
        ocr_when=labels("ocr_when", DEFAULT_OCR_WHEN),
        caption_when=labels("caption_when", DEFAULT_CAPTION_WHEN),
    )


def _parse_processors(block: dict, defaults: dict[str, str], context: str) -> dict[str, str]:
    unknown = set(block) - set(PROCESSOR_KEYS) - set(ROUTING_KEYS) - {"overrides"}
    if unknown:
        warnings.warn(
            f"Unknown processor keys {sorted(unknown)} in {context} are ignored "
            f"(known kinds: {', '.join(PROCESSOR_KEYS)})",
            stacklevel=3,
        )
    processors: dict[str, str] = {}
    for key in PROCESSOR_KEYS:
        if key in block:
            value = block[key]
        elif key in defaults:
            value = defaults[key]
        else:
            continue
        if _is_off(value):
            continue
        if not isinstance(value, str):
            raise ValueError(
                f"Processor {key!r} in {context} must be a model id string or 'off', "
                f"got {value!r} (YAML parses bare on/yes/true as booleans)"
            )
        processors[key] = value
    return processors


def load_config(path: str | Path) -> SearchConfig:
    path = Path(path)
    raw = yaml.safe_load(path.read_text()) or {}

    defaults: dict[str, str] = raw.get("defaults", {}) or {}
    raw_folders: dict[str, dict] = raw.get("folders", {}) or {}

    unknown_defaults = set(defaults) - set(PROCESSOR_KEYS)
    if unknown_defaults:
        warnings.warn(
            f"Unknown processor keys {sorted(unknown_defaults)} in defaults are ignored "
            f"(known kinds: {', '.join(PROCESSOR_KEYS)})",
            stacklevel=2,
        )

    folders: dict[str, FolderConfig] = {}
    for folder_key, folder_raw in raw_folders.items():
        folder_raw = folder_raw or {}
        processors = _parse_processors(folder_raw, defaults, f"folder {folder_key!r}")

        overrides_raw: dict[str, dict] = folder_raw.get("overrides", {}) or {}
        overrides = {
            name: _parse_processors(
                block or {}, defaults, f"override {name!r} of folder {folder_key!r}"
            )
            for name, block in overrides_raw.items()
        }

        folders[folder_key] = FolderConfig(
            path=Path(folder_key).expanduser(),
            processors=processors,
            overrides=overrides,
            routing=_parse_routing(folder_raw, f"folder {folder_key!r}"),
        )

    config = SearchConfig(folders=folders)
    _validate(config)
    return config


def _validate(config: SearchConfig) -> None:
    """Warn loudly when folders that plausibly want unified search/people
    diverge on the locked model they'd need to share (spec section 5)."""
    for kind, label in (("text_embed", "text search"), ("image_embed", "reverse-image search")):
        seen: dict[str, list[str]] = {}
        for folder_key, folder in config.folders.items():
            model = folder.processors.get(kind)
            if model:
                seen.setdefault(model, []).append(folder_key)
        if len(seen) > 1:
            warnings.warn(
                f"Folders use different {kind} models ({label} won't be comparable "
                f"across them): {seen}",
                stacklevel=2,
            )

    seen_face: dict[str, list[str]] = {}
    for folder_key, folder in config.folders.items():
        model = folder.processors.get("faces")
        if model:
            seen_face.setdefault(model, []).append(folder_key)
    if len(seen_face) > 1:
        warnings.warn(
            f"Folders use different face models (people groups won't unify across "
            f"them): {seen_face}",
            stacklevel=2,
        )
