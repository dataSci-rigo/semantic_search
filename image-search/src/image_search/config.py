from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from pathlib import Path

import yaml

# Processor keys read out of a folder's config block, in dispatch order
# (text producers before text_embed — see processors/base.py TEXT_PRODUCER_KINDS).
PROCESSOR_KEYS = (
    "ocr",
    "caption",
    "topic_kw",
    "text_embed",
    "image_embed",
    "faces",
    "tagger",
    "layout",
)


@dataclass(frozen=True)
class FolderConfig:
    path: Path
    processors: dict[str, str]  # kind -> model_id, "off"/absent entries excluded

    def enabled(self, kind: str) -> str | None:
        return self.processors.get(kind)


@dataclass(frozen=True)
class SearchConfig:
    folders: dict[str, FolderConfig] = field(default_factory=dict)

    def active_processors(self) -> set[tuple[str, str]]:
        """Union of (kind, model_id) referenced by any active folder."""
        out: set[tuple[str, str]] = set()
        for folder in self.folders.values():
            out.update(folder.processors.items())
        return out


def _is_off(value: object) -> bool:
    # YAML parses bare `off`/`no`/`false` as bool False, not the string "off".
    if value is None or value is False:
        return True
    return isinstance(value, str) and value.strip().lower() == "off"


def load_config(path: str | Path) -> SearchConfig:
    path = Path(path)
    raw = yaml.safe_load(path.read_text()) or {}

    defaults: dict[str, str] = raw.get("defaults", {}) or {}
    raw_folders: dict[str, dict] = raw.get("folders", {}) or {}

    folders: dict[str, FolderConfig] = {}
    for folder_key, folder_raw in raw_folders.items():
        folder_raw = folder_raw or {}
        processors: dict[str, str] = {}
        for key in PROCESSOR_KEYS:
            if key in folder_raw:
                value = folder_raw[key]
            elif key in defaults:
                value = defaults[key]
            else:
                continue
            if _is_off(value):
                continue
            processors[key] = value

        folders[folder_key] = FolderConfig(
            path=Path(folder_key).expanduser(),
            processors=processors,
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
