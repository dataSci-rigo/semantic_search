from __future__ import annotations

import sqlite3
from pathlib import Path

from image_search import textitems
from image_search.config import FolderConfig, SearchConfig
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
    """Incrementally index one configured folder: images through the
    processor pipeline, .md/.txt files as note items, .links files as link
    items — all searchable together.

    Freshness is tracked per *path* (files table: stat-only skip for unchanged
    mtimes, no re-hashing), while processing happens once per *content id* —
    renames, metadata-only touches, and byte-identical duplicate copies never
    reprocess or duplicate derived records. Content no path references anymore
    is purged at the end. Returns counts."""
    folder = config.folders[folder_key]
    walked = images_store.walk_candidates(folder.path)
    known = images_store.load_file_state(conn, folder_key)

    stats = {"seen": len(walked), "skipped": 0, "indexed": 0, "pruned": 0}
    seen_paths: set[str] = set()

    for path, mtime in walked:
        path_str = str(path)
        seen_paths.add(path_str)
        prior = known.get(path_str)
        if prior is not None and prior[1] == mtime:
            stats["skipped"] += 1
            continue

        suffix = path.suffix.lower()
        if suffix in textitems.NOTE_EXTENSIONS:
            indexed = _ingest_note(conn, registry, folder, folder_key, path, mtime)
        elif suffix == textitems.LINKS_EXTENSION:
            indexed = _ingest_links(conn, registry, folder, folder_key, path, mtime)
        else:
            indexed = _ingest_image(conn, registry, folder, folder_key, path, mtime)
        stats["indexed" if indexed else "skipped"] += 1

    stats["pruned"] = images_store.prune_missing(conn, folder_key, seen_paths)
    conn.commit()
    return stats


def _text_embedder(registry: Registry, folder: FolderConfig, path: Path):
    """The folder's text_embed processor for this path, or None."""
    model = folder.processors_for_path(path).get("text_embed")
    return registry.get("text_embed", model) if model else None


def _ingest_note(
    conn: sqlite3.Connection,
    registry: Registry,
    folder: FolderConfig,
    folder_key: str,
    path: Path,
    mtime: float,
) -> bool:
    item_id = textitems.note_id(path)
    images_store.upsert_file(conn, str(path), folder_key, item_id, mtime)
    if conn.execute("SELECT 1 FROM items WHERE id = ?", (item_id,)).fetchone():
        conn.commit()
        return False
    title, body = textitems.parse_note(path)
    textitems.insert_item(
        conn,
        item_id=item_id,
        kind="note",
        folder=folder_key,
        src_path=str(path),
        title=title,
        url=None,
        body=body,
        text_embedder=_text_embedder(registry, folder, path),
    )
    conn.commit()
    return True


def _ingest_links(
    conn: sqlite3.Connection,
    registry: Registry,
    folder: FolderConfig,
    folder_key: str,
    path: Path,
    mtime: float,
) -> bool:
    """Diff a .links file against its previously ingested items: fetch and
    add only new URLs, drop items for URLs that left the file."""
    path_str = str(path)
    images_store.upsert_file(
        conn, path_str, folder_key, images_store.content_hash(path), mtime
    )
    links = textitems.parse_links(path)
    current_ids = {textitems.link_id(url) for url, _ in links}
    previous_ids = {
        r["id"]
        for r in conn.execute("SELECT id FROM items WHERE src_path = ?", (path_str,))
    }
    for stale in previous_ids - current_ids:
        images_store.purge_item(conn, stale)

    embedder = _text_embedder(registry, folder, path)
    added = 0
    handled: set[str] = set()
    for url, comment in links:
        item_id = textitems.link_id(url)
        if item_id in previous_ids or item_id in handled:
            continue
        handled.add(item_id)
        if conn.execute("SELECT 1 FROM items WHERE id = ?", (item_id,)).fetchone():
            # Same URL already ingested from another file — re-point it here
            # rather than duplicating its FTS/vector records.
            conn.execute(
                "UPDATE items SET src_path = ?, folder = ? WHERE id = ?",
                (path_str, folder_key, item_id),
            )
            continue
        title, page_text = textitems.fetch_page(url)
        body = "\n".join(part for part in (comment, page_text) if part)
        textitems.insert_item(
            conn,
            item_id=item_id,
            kind="link",
            folder=folder_key,
            src_path=path_str,
            title=title or url,
            url=url,
            body=body,
            text_embedder=embedder,
        )
        added += 1
    conn.commit()
    return added > 0 or bool(previous_ids - current_ids)


def _ingest_image(
    conn: sqlite3.Connection,
    registry: Registry,
    folder: FolderConfig,
    folder_key: str,
    path: Path,
    mtime: float,
) -> bool:
    disc = images_store.describe(path, folder_key, mtime)
    images_store.upsert_file(conn, str(path), folder_key, disc.image_id, mtime)
    if images_store.is_indexed(conn, disc.image_id):
        # Same content already processed under another path (duplicate
        # copy), a rename, or a metadata-only touch — nothing to redo.
        conn.commit()
        return False

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

    # Written only after every processor succeeded: the images row is the
    # done-marker is_indexed checks, and the per-image commit below makes
    # each image all-or-nothing (a crash mid-image rolls back its files
    # row too, so the next run retries it).
    images_store.upsert_image(conn, disc)
    conn.commit()
    return True


def ingest_all(conn: sqlite3.Connection, config: SearchConfig, registry: Registry) -> dict:
    return {
        folder_key: ingest_folder(conn, config, registry, folder_key)
        for folder_key in config.folders
    }
