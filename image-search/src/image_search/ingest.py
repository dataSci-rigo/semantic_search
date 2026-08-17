from __future__ import annotations

import logging
import sqlite3
from pathlib import Path

from image_search import textitems
from image_search.config import FolderConfig, SearchConfig
from image_search.processors.base import (
    CaptionRecord,
    ImageEmbedRecord,
    LoadedImage,
    OcrRecord,
    TagRecord,
    TextEmbedRecord,
)
from image_search.registry import Registry
from image_search.store import images as images_store
from image_search.store import vectors as vectors_store

logger = logging.getLogger(__name__)


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
    is purged at the end.

    One bad file costs one file: a failing processor (corrupt image, transient
    OCR/caption worker hiccup) is logged and counted, and the walk continues.
    Nothing was committed for that file, so the next full run retries it —
    which is the whole point, since a multi-day index under systemd
    Restart=on-failure would otherwise restart-loop on it forever.

    Returns counts, including "failed"."""
    folder = config.folders[folder_key]
    walked = images_store.walk_candidates(folder.path)
    known = images_store.load_file_state(conn, folder_key)

    stats = {"seen": len(walked), "skipped": 0, "indexed": 0, "pruned": 0, "failed": 0}
    seen_paths: set[str] = set()
    failed_paths: list[str] = []

    for path, mtime in walked:
        path_str = str(path)
        # Added before dispatch so a file that fails is never mistaken for a
        # deleted one and purged by prune_missing below.
        seen_paths.add(path_str)
        prior = known.get(path_str)
        if prior is not None and prior[1] == mtime:
            stats["skipped"] += 1
            continue

        try:
            suffix = path.suffix.lower()
            if suffix in textitems.NOTE_EXTENSIONS:
                indexed = _ingest_note(conn, registry, folder, folder_key, path, mtime)
            elif suffix == textitems.LINKS_EXTENSION:
                indexed = _ingest_links(conn, registry, folder, folder_key, path, mtime)
            else:
                indexed = _ingest_image(conn, registry, folder, folder_key, path, mtime)
            stats["indexed" if indexed else "skipped"] += 1
        except Exception:  # noqa: BLE001 - one bad file must not end the run
            # Discard this file's partial writes: the _ingest_* helpers commit
            # only on success, so without this rollback its half-written rows
            # would ride along on the next file's commit.
            conn.rollback()
            logger.exception("ingest failed for %s (skipped; retried next run)", path_str)
            stats["failed"] += 1
            failed_paths.append(path_str)

    stats["pruned"] = images_store.prune_missing(conn, folder_key, seen_paths)
    conn.commit()
    if failed_paths:
        logger.warning(
            "%s: %d file(s) failed and were skipped, e.g. %s",
            folder_key,
            len(failed_paths),
            ", ".join(failed_paths[:3]),
        )
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
    image_vector: list[float] | None = None
    ranked_tags: list[tuple[str, float, int]] = []
    tag_records: list[TagRecord] = []
    for kind, processor in processors:
        # Content-based routing (route: auto): the tagger ran earlier in this
        # same pass — dispatch order puts image_embed and tagger first — so
        # its labels decide whether the costly text/caption steps apply to
        # this particular image rather than to its whole folder.
        if not folder.routing.wants(kind, ranked_tags):
            continue

        img = LoadedImage(
            image_id=disc.image_id,
            path=disc.path,
            width=disc.width,
            height=disc.height,
            text=accumulated_text,
            image_vector=image_vector,
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
                # Handed to the tagger below so it classifies without a
                # second vision forward pass.
                image_vector = record.vector
            elif isinstance(record, TagRecord):
                tag_records.append(record)
            else:
                raise NotImplementedError(
                    f"ingest does not yet handle record type {type(record).__name__}"
                )

        if kind == "tagger" and tag_records:
            # Rank now (1 = best match for this image): raw cosines are not
            # comparable across images, so rank is what routing below and
            # facet search filter on.
            ordered = sorted(tag_records, key=lambda r: r.score, reverse=True)
            ranked_tags = [(r.tag, r.score, i + 1) for i, r in enumerate(ordered)]
            conn.executemany(
                "INSERT INTO tags (image_id, tag, source, score, rank) VALUES (?, ?, ?, ?, ?)",
                [
                    (disc.image_id, r.tag, r.source, r.score, i + 1)
                    for i, r in enumerate(ordered)
                ],
            )
            tag_records = []

    # Written only after every processor succeeded: the images row is the
    # done-marker is_indexed checks, and the per-image commit below makes
    # each image all-or-nothing (a crash mid-image rolls back its files
    # row too, so the next run retries it).
    images_store.upsert_image(conn, disc)
    conn.commit()
    return True


def ingest_all(conn: sqlite3.Connection, config: SearchConfig, registry: Registry) -> dict:
    """Index every configured folder. A folder that fails outright (unreadable
    path, unmounted drive — walk_candidates throws before per-file handling
    can help) is reported and skipped so the remaining folders still index."""
    out: dict[str, dict] = {}
    for folder_key in config.folders:
        try:
            out[folder_key] = ingest_folder(conn, config, registry, folder_key)
        except Exception as exc:  # noqa: BLE001 - one bad folder must not end the run
            conn.rollback()
            logger.exception("ingest failed for folder %s (skipped)", folder_key)
            out[folder_key] = {
                "seen": 0, "skipped": 0, "indexed": 0, "pruned": 0, "failed": 0,
                "error": str(exc),
            }
    return out
