from __future__ import annotations

import sqlite3

from image_search.config import SearchConfig
from image_search.processors.base import (
    CaptionRecord,
    ImageEmbedRecord,
    LoadedImage,
    OcrRecord,
    TextEmbedRecord,
)
from image_search.registry import Registry
from image_search.store import images as images_store
from image_search.store import vectors as vectors_store


def ingest_folder(
    conn: sqlite3.Connection, config: SearchConfig, registry: Registry, folder_key: str
) -> dict[str, int]:
    """Discover files in one configured folder, skip unchanged ones, dispatch
    enabled processors in order, and persist records. Returns counts."""
    folder = config.folders[folder_key]
    discovered = images_store.discover(folder.path, folder_key)

    stats = {"seen": len(discovered), "skipped": 0, "indexed": 0}

    for disc in discovered:
        if images_store.is_unchanged(conn, disc.image_id, disc.mtime):
            stats["skipped"] += 1
            continue

        images_store.upsert_image(conn, disc)

        # Path overrides (e.g. nested "Screenshots" dirs) let one folder
        # entry route different subtrees through different pipelines.
        processors = registry.for_processors(folder.processors_for_path(disc.path))

        accumulated_text = ""
        for kind, processor in processors:
            img = LoadedImage(
                image_id=disc.image_id,
                path=disc.path,
                width=disc.width,
                height=disc.height,
                text=accumulated_text,
            )
            for record in processor.process(img):
                if isinstance(record, OcrRecord):
                    conn.execute(
                        "INSERT INTO ocr_text (image_id, model, text) VALUES (?, ?, ?)",
                        (disc.image_id, processor.model_id, record.text),
                    )
                    conn.execute(
                        "INSERT INTO text_fts (image_id, text) VALUES (?, ?)",
                        (disc.image_id, record.text),
                    )
                    accumulated_text = (accumulated_text + "\n" + record.text).strip()
                elif isinstance(record, CaptionRecord):
                    conn.execute(
                        "INSERT INTO captions (image_id, model, text) VALUES (?, ?, ?)",
                        (disc.image_id, processor.model_id, record.text),
                    )
                    conn.execute(
                        "INSERT INTO text_fts (image_id, text) VALUES (?, ?)",
                        (disc.image_id, record.text),
                    )
                    accumulated_text = (accumulated_text + "\n" + record.text).strip()
                elif isinstance(record, TextEmbedRecord):
                    vectors_store.insert_vector(
                        conn, "text", record.model, disc.image_id, record.vector
                    )
                elif isinstance(record, ImageEmbedRecord):
                    vectors_store.insert_vector(
                        conn, "image", record.model, disc.image_id, record.vector
                    )
                else:
                    raise NotImplementedError(
                        f"ingest does not yet handle record type {type(record).__name__}"
                    )

        conn.commit()
        stats["indexed"] += 1

    return stats


def ingest_all(conn: sqlite3.Connection, config: SearchConfig, registry: Registry) -> dict:
    return {
        folder_key: ingest_folder(conn, config, registry, folder_key)
        for folder_key in config.folders
    }
