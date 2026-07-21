from __future__ import annotations

import sqlite3
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS images (
  id           TEXT PRIMARY KEY,      -- sha256 of file bytes
  path         TEXT NOT NULL,
  folder       TEXT NOT NULL,         -- config key this image was matched to
  content_hash TEXT NOT NULL,
  mtime        REAL,
  width        INTEGER,
  height       INTEGER,
  indexed_at   REAL
);

-- FREE outputs: text. Multiple models may coexist (model column).
CREATE TABLE IF NOT EXISTS ocr_text (image_id TEXT, model TEXT, text TEXT);
CREATE TABLE IF NOT EXISTS captions (image_id TEXT, model TEXT, text TEXT);
CREATE VIRTUAL TABLE IF NOT EXISTS text_fts USING fts5(image_id, text);

-- FREE outputs: tags. Continuous score; threshold at query time.
CREATE TABLE IF NOT EXISTS tags (
  image_id TEXT, tag TEXT, source TEXT, score REAL
);
CREATE INDEX IF NOT EXISTS idx_tags_tag ON tags(tag);

-- Faces. cluster_id assigned by the clustering job.
CREATE TABLE IF NOT EXISTS faces (
  face_id    TEXT PRIMARY KEY,
  image_id   TEXT,
  model      TEXT,
  det_model  TEXT,
  bbox       TEXT,
  det_score  REAL,
  embedding  BLOB,
  cluster_id INTEGER
);
CREATE TABLE IF NOT EXISTS face_clusters (
  cluster_id INTEGER PRIMARY KEY, model TEXT, label TEXT, centroid BLOB,
  size INTEGER, updated_at REAL
);

-- LOCKED outputs: sidecar mapping vec0 rowid -> image_id, per vec table.
-- vec0 tables (vec_text__<model>, vec_image__<model>) are created dynamically
-- by store/vectors.py on first encounter of a given (space, model).
CREATE TABLE IF NOT EXISTS vec_map (
  vec_table TEXT, rowid INTEGER, image_id TEXT,
  PRIMARY KEY (vec_table, rowid)
);
"""


def connect(db_path: str | Path) -> sqlite3.Connection:
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def load_vec_extension(conn: sqlite3.Connection) -> None:
    """Load the sqlite-vec extension. Only required for vec0 table ops
    (store/vectors.py), not for Phase 0 schema/discovery/FTS."""
    try:
        import sqlite_vec  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "sqlite-vec is not installed. Install it into the active environment "
            "(`sem_search_gpu`) to use vector search: pip install sqlite-vec"
        ) from exc

    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)


def migrate(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA)
    conn.commit()
