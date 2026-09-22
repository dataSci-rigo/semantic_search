from __future__ import annotations

import logging
import os
import sqlite3
from pathlib import Path

logger = logging.getLogger(__name__)

# Bumped when the stored format changes incompatibly. Version 2: paths in
# files/images/items are stored relative to their folder root (portable
# index — the drive can mount anywhere), and files is keyed by (folder, path).
DB_VERSION = 2

SCHEMA = """
CREATE TABLE IF NOT EXISTS images (
  id           TEXT PRIMARY KEY,      -- sha256 of file bytes
  path         TEXT NOT NULL,         -- POSIX path relative to the folder root
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
-- Keyed by (folder, path): paths are folder-relative, so two folders can
-- legitimately hold the same relative path.
CREATE TABLE IF NOT EXISTS files (
  path     TEXT NOT NULL,             -- POSIX path relative to the folder root
  folder   TEXT NOT NULL,
  image_id TEXT NOT NULL,
  mtime    REAL NOT NULL,
  PRIMARY KEY (folder, path)
);
CREATE INDEX IF NOT EXISTS idx_files_image_id ON files(image_id);

-- Non-image "interesting stuff": notes (.md/.txt files) and links (from
-- .links files or /api/save). Shares text_fts and the text vector tables
-- with images — the id columns there hold item ids just as well.
CREATE TABLE IF NOT EXISTS items (
  id         TEXT PRIMARY KEY,   -- note: sha256 of file bytes; link: sha256("link:"+url)
  kind       TEXT NOT NULL,      -- "note" | "link" | "pdf" | "docx" | "csv" | "xlsx" | "book"
  folder     TEXT NOT NULL,      -- config key this item belongs to
  src_path   TEXT NOT NULL,      -- the file it came from, relative to the folder root
  title      TEXT,
  url        TEXT,
  body       TEXT,
  -- Fetch outcome for links: ok | dead | blocked | thin | skipped. Rows that
  -- aren't "ok" are kept (so a bad filter is visible and reversible) but are
  -- excluded from search results.
  status     TEXT DEFAULT 'ok',
  created_at REAL
);
CREATE INDEX IF NOT EXISTS idx_items_folder ON items(folder);
CREATE INDEX IF NOT EXISTS idx_items_src_path ON items(src_path);

-- FREE outputs: text. Multiple models may coexist (model column).
CREATE TABLE IF NOT EXISTS ocr_text (image_id TEXT, model TEXT, text TEXT);
CREATE TABLE IF NOT EXISTS captions (image_id TEXT, model TEXT, text TEXT);
-- source (UNINDEXED: a filter column, not tokenized/searchable) lets search
-- scope keyword matches to "ocr" or "caption" text specifically. NULL for
-- items (notes/links/pdfs) — field-scoping doesn't apply to those, so they
-- pass through every field filter unfiltered. See search.py:search_text.
CREATE VIRTUAL TABLE IF NOT EXISTS text_fts USING fts5(image_id, text, source UNINDEXED);

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


# Journal modes sqlite accepts; the mode string is interpolated into a PRAGMA,
# so it is validated against this set first.
_JOURNAL_MODES = {"wal", "delete", "truncate", "persist", "memory", "off"}

# Filesystems where WAL's shared-memory file and POSIX locking are unreliable:
# network/translated mounts (9p = WSL drvfs, fuseblk = ntfs-3g, CIFS/NFS) and
# FAT-family disks. A DB living on one of these gets a rollback journal.
_NO_WAL_FSTYPES = {
    "9p", "v9fs", "drvfs", "fuseblk", "fuse", "ntfs", "ntfs3",
    "vfat", "exfat", "msdos", "cifs", "smb3", "nfs", "nfs4",
}


def _default_journal_mode(db_path: Path, mounts_path: str = "/proc/mounts") -> str:
    """WAL on local filesystems, TRUNCATE when the DB sits on a mount that
    can't support WAL (external drives over 9p/ntfs-3g, network shares).
    Platforms without /proc/mounts (Windows, macOS) default to WAL — their
    native filesystems handle it."""
    try:
        with open(mounts_path, encoding="utf-8") as fh:
            lines = fh.readlines()
    except OSError:
        return "wal"
    try:
        target = str(db_path.resolve().parent if db_path.suffix else db_path.resolve())
    except OSError:
        target = str(db_path.parent)
    best_len, best_fs = -1, None
    for line in lines:
        parts = line.split()
        if len(parts) < 3:
            continue
        # /proc/mounts octal-escapes spaces in mount points as \040.
        mount_point = parts[1].replace("\\040", " ")
        fstype = parts[2]
        if target == mount_point or target.startswith(mount_point.rstrip("/") + "/"):
            if len(mount_point) > best_len:
                best_len, best_fs = len(mount_point), fstype
    return "truncate" if best_fs in _NO_WAL_FSTYPES else "wal"


def connect(db_path: str | Path, journal_mode: str | None = None) -> sqlite3.Connection:
    """Open (creating parents as needed). Journal mode precedence: the
    `journal_mode` arg > IMAGE_SEARCH_JOURNAL_MODE env > filesystem-based
    default (WAL locally — it lets the web app read while the indexer
    writes — TRUNCATE on mounts that can't do WAL)."""
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path, timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    mode = (
        journal_mode
        or os.environ.get("IMAGE_SEARCH_JOURNAL_MODE")
        or _default_journal_mode(db_path)
    ).lower()
    if mode not in _JOURNAL_MODES:
        raise ValueError(f"invalid journal mode {mode!r} (one of {sorted(_JOURNAL_MODES)})")
    granted = conn.execute(f"PRAGMA journal_mode = {mode}").fetchone()[0]
    if str(granted).lower() != mode:
        logger.warning(
            "journal_mode=%s not granted for %s (got %s) — continuing with %s",
            mode, db_path, granted, granted,
        )
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


def _migrate_text_fts_add_source(conn: sqlite3.Connection) -> None:
    """FTS5 virtual tables can't ALTER TABLE ADD COLUMN, so a `text_fts`
    predating the `source` column needs a full rebuild: rename, recreate with
    the column, backfill by matching (image_id, text) against ocr_text /
    captions (whichever the row's exact text came from — items get no match,
    stay NULL), then drop the old table. Cheap even for tens of thousands of
    rows; run once per DB, guarded by whether the column already exists."""
    existing = {r["name"] for r in conn.execute("PRAGMA table_info(text_fts)")}
    if "source" in existing:
        return
    conn.execute("ALTER TABLE text_fts RENAME TO text_fts_old")
    conn.execute("CREATE VIRTUAL TABLE text_fts USING fts5(image_id, text, source UNINDEXED)")
    conn.execute(
        """
        INSERT INTO text_fts (image_id, text, source)
        SELECT
          old.image_id,
          old.text,
          CASE
            WHEN EXISTS (
              SELECT 1 FROM ocr_text o WHERE o.image_id = old.image_id AND o.text = old.text
            ) THEN 'ocr'
            WHEN EXISTS (
              SELECT 1 FROM captions c WHERE c.image_id = old.image_id AND c.text = old.text
            ) THEN 'caption'
            ELSE NULL
          END
        FROM text_fts_old old
        """
    )
    conn.execute("DROP TABLE text_fts_old")


def _legacy_files_table(conn: sqlite3.Connection) -> bool:
    """True when a `files` table exists with the pre-portable single-column
    primary key (path alone, absolute paths)."""
    exists = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='files'"
    ).fetchone()
    if not exists:
        return False
    cols = {r["name"]: r["pk"] for r in conn.execute("PRAGMA table_info(files)")}
    return cols.get("path") == 1 and cols.get("folder", 0) == 0


def migrate(conn: sqlite3.Connection) -> None:
    version = conn.execute("PRAGMA user_version").fetchone()[0]
    if version == 0 and _legacy_files_table(conn):
        if conn.execute("SELECT 1 FROM files LIMIT 1").fetchone() is not None:
            raise RuntimeError(
                "this index predates the portable (relative-path) format and "
                "holds data; re-index into a fresh DB file instead of reusing it"
            )
        # Empty legacy table: safe to rebuild with the composite key.
        conn.execute("DROP TABLE files")
        conn.execute("DROP INDEX IF EXISTS idx_files_folder")
    conn.executescript(SCHEMA)
    _add_column_if_missing(conn, "tags", "rank", "INTEGER")
    _add_column_if_missing(conn, "items", "status", "TEXT DEFAULT 'ok'")
    _migrate_text_fts_add_source(conn)
    conn.execute(f"PRAGMA user_version = {DB_VERSION}")
    conn.commit()
