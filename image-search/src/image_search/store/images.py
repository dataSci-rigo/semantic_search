from __future__ import annotations

import fnmatch
import hashlib
import logging
import os
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from PIL import Image

from image_search.store import vectors as vectors_store

logger = logging.getLogger(__name__)


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


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif", ".tif", ".tiff"}

# Names that carry a real extension but never real content: Office owner/
# lock files ("~$Report.docx") and macOS AppleDouble resource forks
# ("._IMG_1234.JPG", left by Finder on non-Mac volumes).
_JUNK_PREFIXES = ("~$", "._")


def _dir_excluded(name: str, patterns: Sequence[str]) -> bool:
    lowered = name.lower()
    return any(fnmatch.fnmatchcase(lowered, pat.lower()) for pat in patterns)


def walk_candidates(
    folder_path: Path, exclude_dirs: Sequence[str] = ()
) -> list[tuple[Path, float]]:
    """Stat-only walk: (path, mtime) for every ingestible file (images plus
    note/.links/document files), sorted by path. No hashing here — ingest
    hashes only paths whose mtime changed.

    A missing root RAISES rather than returning [] — an empty result would
    read as "everything was deleted" and let prune_missing wipe the folder's
    whole index the first time its drive is unplugged.

    Directories whose *name* matches an exclude_dirs pattern
    (case-insensitive fnmatch) are pruned from the walk, subtree and all."""
    from image_search.textitems import DOCUMENT_KINDS, LINKS_EXTENSION, NOTE_EXTENSIONS

    if not folder_path.is_dir():
        raise FileNotFoundError(
            f"folder root {folder_path} does not exist or is not a directory "
            "(unmounted drive?)"
        )

    extensions = IMAGE_EXTENSIONS | NOTE_EXTENSIONS | {LINKS_EXTENSION} | set(DOCUMENT_KINDS)
    out: list[tuple[Path, float]] = []

    def _onerror(err: OSError) -> None:
        # Unreadable subdirs are logged, not silently skipped — pathlib's
        # rglob would swallow these, making "no permission" look like "empty".
        logger.warning("walk: cannot read %s: %s", getattr(err, "filename", folder_path), err)

    for dirpath, dirnames, filenames in os.walk(folder_path, onerror=_onerror):
        dirnames[:] = sorted(d for d in dirnames if not _dir_excluded(d, exclude_dirs))
        for name in filenames:
            if name.startswith(_JUNK_PREFIXES):
                continue
            path = Path(dirpath) / name
            if path.suffix.lower() not in extensions:
                continue
            try:
                mtime = path.stat().st_mtime
            except OSError:
                continue  # vanished or unreadable between listing and stat
            out.append((path, mtime))
    out.sort()
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
    """`path` is the storage form: POSIX-relative to the folder root."""
    conn.execute(
        """
        INSERT INTO files (path, folder, image_id, mtime) VALUES (?, ?, ?, ?)
        ON CONFLICT(folder, path) DO UPDATE SET
          image_id=excluded.image_id, mtime=excluded.mtime
        """,
        (path, folder, image_id, mtime),
    )


def is_indexed(conn: sqlite3.Connection, image_id: str) -> bool:
    """True once this content has been fully processed (the images row is
    written after the processors succeed, so it doubles as a done-marker)."""
    return (
        conn.execute("SELECT 1 FROM images WHERE id = ?", (image_id,)).fetchone() is not None
    )


def upsert_image(conn: sqlite3.Connection, img: DiscoveredImage, db_path: str) -> None:
    """`db_path` is the storage form of img.path (folder-relative POSIX);
    img.path stays absolute for I/O."""
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
            db_path,
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
            conn.execute(
                "DELETE FROM files WHERE folder = ? AND path = ?",
                (folder_key, row["path"]),
            )

    orphans = conn.execute(
        "SELECT id FROM images WHERE folder = ? "
        "AND id NOT IN (SELECT image_id FROM files)",
        (folder_key,),
    ).fetchall()
    for row in orphans:
        purge_image(conn, row["id"])

    # Every item kind except links is content-addressed like images
    # (files.image_id holds the item id) — notes, pdf/docx/csv/xlsx documents.
    # Like the images query above, the id subquery is deliberately unscoped:
    # content-addressing is global, so a copy surviving in another folder
    # keeps the item alive. Links are keyed to their source .links file
    # instead (one file yields many items), and that subquery IS
    # folder-scoped because relative src_paths can collide across folders.
    orphan_items = conn.execute(
        """
        SELECT id FROM items WHERE folder = ? AND (
          (kind != 'link' AND id NOT IN (SELECT image_id FROM files))
          OR (kind = 'link' AND src_path NOT IN (SELECT path FROM files WHERE folder = ?))
        )
        """,
        (folder_key, folder_key),
    ).fetchall()
    for row in orphan_items:
        purge_item(conn, row["id"])

    return len(orphans) + len(orphan_items)


def duplicate_groups(conn: sqlite3.Connection) -> list[tuple[str, list[tuple[str, str]]]]:
    """Content present under more than one path:
    [(image_id, sorted [(folder, relative path)])]."""
    rows = conn.execute(
        """
        SELECT image_id, folder, path FROM files
        WHERE image_id IN (
          SELECT image_id FROM files GROUP BY image_id HAVING COUNT(*) > 1
        )
        ORDER BY image_id, folder, path
        """
    ).fetchall()
    groups: dict[str, list[tuple[str, str]]] = {}
    for r in rows:
        groups.setdefault(r["image_id"], []).append((r["folder"], r["path"]))
    return sorted(groups.items())
