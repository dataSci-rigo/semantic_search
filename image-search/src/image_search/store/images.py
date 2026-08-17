from __future__ import annotations

import hashlib
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

from image_search.store import vectors as vectors_store


def content_hash(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


@dataclass(frozen=True)
class DiscoveredImage:
    image_id: str
    path: Path
    folder: str
    mtime: float
    width: int
    height: int


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif", ".tiff"}


def walk_candidates(folder_path: Path) -> list[tuple[Path, float]]:
    """Stat-only walk: (path, mtime) for every ingestible file (images plus
    note/.links text files), sorted by path. No hashing here — ingest hashes
    only paths whose mtime changed."""
    from image_search.textitems import LINKS_EXTENSION, NOTE_EXTENSIONS, PDF_EXTENSION

    extensions = IMAGE_EXTENSIONS | NOTE_EXTENSIONS | {LINKS_EXTENSION, PDF_EXTENSION}
    out: list[tuple[Path, float]] = []
    if not folder_path.exists():
        return out
    for path in sorted(folder_path.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in extensions:
            continue
        out.append((path, path.stat().st_mtime))
    return out


def describe(path: Path, folder_key: str, mtime: float) -> DiscoveredImage:
    """Hash the file and read its dimensions (the expensive part of discovery)."""
    image_id = content_hash(path)
    with Image.open(path) as im:
        width, height = im.size
    return DiscoveredImage(
        image_id=image_id,
        path=path,
        folder=folder_key,
        mtime=mtime,
        width=width,
        height=height,
    )


def load_file_state(conn: sqlite3.Connection, folder_key: str) -> dict[str, tuple[str, float]]:
    """path -> (image_id, mtime) as of the last ingest of this folder."""
    return {
        r["path"]: (r["image_id"], r["mtime"])
        for r in conn.execute(
            "SELECT path, image_id, mtime FROM files WHERE folder = ?", (folder_key,)
        )
    }


def upsert_file(
    conn: sqlite3.Connection, path: str, folder: str, image_id: str, mtime: float
) -> None:
    conn.execute(
        """
        INSERT INTO files (path, folder, image_id, mtime) VALUES (?, ?, ?, ?)
        ON CONFLICT(path) DO UPDATE SET
          folder=excluded.folder, image_id=excluded.image_id, mtime=excluded.mtime
        """,
        (path, folder, image_id, mtime),
    )


def is_indexed(conn: sqlite3.Connection, image_id: str) -> bool:
    """True once this content has been fully processed (the images row is
    written after the processors succeed, so it doubles as a done-marker)."""
    return (
        conn.execute("SELECT 1 FROM images WHERE id = ?", (image_id,)).fetchone() is not None
    )


def upsert_image(conn: sqlite3.Connection, img: DiscoveredImage) -> None:
    conn.execute(
        """
        INSERT INTO images (id, path, folder, content_hash, mtime, width, height, indexed_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(id) DO UPDATE SET
          path=excluded.path, folder=excluded.folder, content_hash=excluded.content_hash,
          mtime=excluded.mtime, width=excluded.width, height=excluded.height,
          indexed_at=excluded.indexed_at
        """,
        (
            img.image_id,
            str(img.path),
            img.folder,
            img.image_id,
            img.mtime,
            img.width,
            img.height,
            time.time(),
        ),
    )


def purge_image(conn: sqlite3.Connection, image_id: str) -> None:
    """Delete an image's row and every derived record (text, FTS, vectors)."""
    conn.execute("DELETE FROM ocr_text WHERE image_id = ?", (image_id,))
    conn.execute("DELETE FROM captions WHERE image_id = ?", (image_id,))
    conn.execute("DELETE FROM text_fts WHERE image_id = ?", (image_id,))
    conn.execute("DELETE FROM tags WHERE image_id = ?", (image_id,))
    conn.execute("DELETE FROM faces WHERE image_id = ?", (image_id,))
    vectors_store.delete_vectors(conn, image_id)
    conn.execute("DELETE FROM images WHERE id = ?", (image_id,))


def purge_item(conn: sqlite3.Connection, item_id: str) -> None:
    """Delete a note/link item and its derived records (FTS, vectors)."""
    conn.execute("DELETE FROM text_fts WHERE image_id = ?", (item_id,))
    vectors_store.delete_vectors(conn, item_id)
    conn.execute("DELETE FROM items WHERE id = ?", (item_id,))


def prune_missing(conn: sqlite3.Connection, folder_key: str, seen_paths: set[str]) -> int:
    """Drop files rows for paths that vanished from this folder, then purge
    every image or item of this folder whose content no path references
    anymore — whether the path was deleted or edited in place (re-pointed to
    a new content id). Returns the number of images+items purged."""
    for row in conn.execute(
        "SELECT path FROM files WHERE folder = ?", (folder_key,)
    ).fetchall():
        if row["path"] not in seen_paths:
            conn.execute("DELETE FROM files WHERE path = ?", (row["path"],))

    orphans = conn.execute(
        "SELECT id FROM images WHERE folder = ? "
        "AND id NOT IN (SELECT image_id FROM files)",
        (folder_key,),
    ).fetchall()
    for row in orphans:
        purge_image(conn, row["id"])

    # Notes are content-addressed like images (files.image_id holds the note
    # id); links are keyed to their source .links file instead, since one
    # file yields many items.
    orphan_items = conn.execute(
        """
        SELECT id FROM items WHERE folder = ? AND (
          (kind = 'note' AND id NOT IN (SELECT image_id FROM files))
          OR (kind = 'link' AND src_path NOT IN (SELECT path FROM files))
        )
        """,
        (folder_key,),
    ).fetchall()
    for row in orphan_items:
        purge_item(conn, row["id"])

    return len(orphans) + len(orphan_items)


def duplicate_groups(conn: sqlite3.Connection) -> list[tuple[str, list[str]]]:
    """Content present under more than one path: [(image_id, sorted paths)]."""
    rows = conn.execute(
        """
        SELECT image_id, path FROM files
        WHERE image_id IN (
          SELECT image_id FROM files GROUP BY image_id HAVING COUNT(*) > 1
        )
        ORDER BY image_id, path
        """
    ).fetchall()
    groups: dict[str, list[str]] = {}
    for r in rows:
        groups.setdefault(r["image_id"], []).append(r["path"])
    return sorted(groups.items())
