from __future__ import annotations

import hashlib
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path

from PIL import Image


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


def discover(folder_path: Path, folder_key: str) -> list[DiscoveredImage]:
    """Walk a folder, hash each image, and return discovered records.
    Does not touch the DB — caller decides what's new via `is_unchanged`."""
    out: list[DiscoveredImage] = []
    if not folder_path.exists():
        return out
    for path in sorted(folder_path.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        image_id = content_hash(path)
        mtime = path.stat().st_mtime
        with Image.open(path) as im:
            width, height = im.size
        out.append(
            DiscoveredImage(
                image_id=image_id,
                path=path,
                folder=folder_key,
                mtime=mtime,
                width=width,
                height=height,
            )
        )
    return out


def is_unchanged(conn: sqlite3.Connection, image_id: str, mtime: float) -> bool:
    row = conn.execute(
        "SELECT mtime FROM images WHERE id = ?", (image_id,)
    ).fetchone()
    return row is not None and row["mtime"] == mtime


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
