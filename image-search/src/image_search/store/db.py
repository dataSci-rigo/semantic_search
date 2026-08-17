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

-- Per-path freshness state. images is keyed by content hash (so duplicate
-- copies and renames share one processed row); files tracks which on-disk
-- paths exist and their last-seen mtime, so ingest can stat-skip unchanged
-- paths without re-hashing and can prune content no path references anymore.
CREATE TABLE IF NOT EXISTS files (
  path     TEXT PRIMARY KEY,
  folder   TEXT NOT NULL,
  image_id TEXT NOT NULL,
  mtime    REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_files_image_id ON files(image_id);
CREATE INDEX IF NOT EXISTS idx_files_folder ON files(folder);

-- Non-image "interesting stuff": notes (.md/.txt files) and links (from
-- .links files or /api/save). Shares text_fts and the text vector tables
-- with images — the id columns there hold item ids just as well.
CREATE TABLE IF NOT EXISTS items (
  id         TEXT PRIMARY KEY,   -- note: sha256 of file bytes; link: sha256("link:"+url)
  kind       TEXT NOT NULL,      -- "note" | "link"
  folder     TEXT NOT NULL,      -- config key this item belongs to
  src_path   TEXT NOT NULL,      -- the file it came from
  title      TEXT,
  url        TEXT,
  body       TEXT,
  created_at REAL
);
CREATE INDEX IF NOT EXISTS idx_items_folder ON items(folder);
CREATE INDEX IF NOT EXISTS idx_items_src_path ON items(src_path);

-- FREE outputs: text. Multiple models may coexist (model column).
CREATE TABLE IF NOT EXISTS ocr_text (image_id TEXT, model TEXT, text TEXT);
CREATE TABLE IF NOT EXISTS captions (image_id TEXT, model TEXT, text TEXT);
CREATE VIRTUAL TABLE IF NOT EXISTS text_fts USING fts5(image_id, text);

-- FREE outputs: tags. Score is a raw cosine, which is NOT comparable across
-- images — `rank` (1 = this image's best-matching label) is what search and
-- routing filter on.
CREATE TABLE IF NOT EXISTS tags (
  image_id TEXT, tag TEXT, source TEXT, score REAL, rank INTEGER
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
    conn = sqlite3.connect(db_path, timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    # WAL: lets the web app read concurrently while the indexer writes
    # (a multi-day ingest run and search queries hit the same file at once).
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA busy_timeout = 30000")
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


def _add_column_if_missing(
    conn: sqlite3.Connection, table: str, column: str, decl: str
) -> None:
    """CREATE TABLE IF NOT EXISTS won't add a column to a table that already
    exists, so columns added after a DB was first created need this."""
    existing = {r["name"] for r in conn.execute(f"PRAGMA table_info({table})")}
    if column not in existing:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {decl}")


def migrate(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA)
    _add_column_if_missing(conn, "tags", "rank", "INTEGER")
    conn.commit()
