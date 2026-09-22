"""Coverage for store/db.py's schema migrations — specifically
_migrate_text_fts_add_source, which can't use the simple
_add_column_if_missing path since FTS5 virtual tables don't support
ALTER TABLE ADD COLUMN."""

from image_search.store.db import connect, migrate


def test_migrate_backfills_source_from_ocr_and_caption(tmp_path):
    """A text_fts table predating the `source` column must be rebuilt and
    backfilled: OCR-matching rows get 'ocr', caption-matching rows get
    'caption', anything else (items/notes) stays NULL."""
    conn = connect(tmp_path / "test.db")
    # Reproduce the pre-migration schema by hand, bypassing migrate() so the
    # column genuinely doesn't exist yet.
    conn.executescript(
        """
        CREATE TABLE images (id TEXT PRIMARY KEY, path TEXT, folder TEXT,
            content_hash TEXT, mtime REAL, width INTEGER, height INTEGER,
            indexed_at REAL);
        CREATE TABLE ocr_text (image_id TEXT, model TEXT, text TEXT);
        CREATE TABLE captions (image_id TEXT, model TEXT, text TEXT);
        CREATE VIRTUAL TABLE text_fts USING fts5(image_id, text);
        """
    )
    conn.execute("INSERT INTO ocr_text (image_id, model, text) VALUES ('img1', 'm', 'ocr text')")
    conn.execute("INSERT INTO captions (image_id, model, text) VALUES ('img2', 'm', 'a caption')")
    conn.execute("INSERT INTO text_fts (image_id, text) VALUES ('img1', 'ocr text')")
    conn.execute("INSERT INTO text_fts (image_id, text) VALUES ('img2', 'a caption')")
    conn.execute("INSERT INTO text_fts (image_id, text) VALUES ('note1', 'a saved note body')")
    conn.commit()

    migrate(conn)

    rows = {
        r["image_id"]: r["source"]
        for r in conn.execute("SELECT image_id, source FROM text_fts")
    }
    assert rows["img1"] == "ocr"
    assert rows["img2"] == "caption"
    assert rows["note1"] is None


def test_migrate_is_idempotent(tmp_path):
    conn = connect(tmp_path / "test.db")
    migrate(conn)
    before = conn.execute("SELECT * FROM text_fts").fetchall()
    migrate(conn)  # second call must no-op on the source-column branch
    after = conn.execute("SELECT * FROM text_fts").fetchall()
    assert before == after


def test_migrate_fresh_db_has_source_column(tmp_path):
    conn = connect(tmp_path / "test.db")
    migrate(conn)
    cols = {r["name"] for r in conn.execute("PRAGMA table_info(text_fts)")}
    assert "source" in cols


# --- portable-format versioning and journal modes ----------------------------

from pathlib import Path

import pytest

from image_search.store.db import DB_VERSION, _default_journal_mode


def test_user_version_stamped_on_new_db(tmp_path):
    conn = connect(tmp_path / "t.db")
    migrate(conn)
    assert conn.execute("PRAGMA user_version").fetchone()[0] == DB_VERSION


def test_legacy_db_with_data_is_refused(tmp_path):
    """A pre-portable DB (files keyed by absolute path, user_version 0) that
    holds rows must fail loudly, not be silently misread."""
    conn = connect(tmp_path / "t.db")
    conn.executescript(
        "CREATE TABLE files (path TEXT PRIMARY KEY, folder TEXT NOT NULL, "
        "image_id TEXT NOT NULL, mtime REAL NOT NULL);"
    )
    conn.execute("INSERT INTO files VALUES ('/abs/a.png', '~/Pics', 'id1', 1.0)")
    conn.commit()
    with pytest.raises(RuntimeError, match="re-index"):
        migrate(conn)


def test_legacy_empty_files_table_is_rebuilt(tmp_path):
    conn = connect(tmp_path / "t.db")
    conn.executescript(
        "CREATE TABLE files (path TEXT PRIMARY KEY, folder TEXT NOT NULL, "
        "image_id TEXT NOT NULL, mtime REAL NOT NULL);"
    )
    conn.commit()
    migrate(conn)
    pks = {r["name"]: r["pk"] for r in conn.execute("PRAGMA table_info(files)")}
    assert pks["folder"] == 1 and pks["path"] == 2


def test_journal_mode_arg_and_env(tmp_path, monkeypatch):
    conn = connect(tmp_path / "a.db", journal_mode="truncate")
    assert conn.execute("PRAGMA journal_mode").fetchone()[0] == "truncate"

    monkeypatch.setenv("IMAGE_SEARCH_JOURNAL_MODE", "delete")
    conn2 = connect(tmp_path / "b.db")
    assert conn2.execute("PRAGMA journal_mode").fetchone()[0] == "delete"


def test_invalid_journal_mode_rejected(tmp_path):
    with pytest.raises(ValueError):
        connect(tmp_path / "a.db", journal_mode="wal; DROP TABLE files")


def test_default_journal_mode_by_filesystem(tmp_path):
    mounts = tmp_path / "mounts"
    mounts.write_text(
        "/dev/sda1 / ext4 rw 0 0\n"
        "C:\\134 /mnt/c 9p rw 0 0\n"
        "D:\\134 /mnt/d 9p rw 0 0\n"
    )
    assert _default_journal_mode(Path("/home/x/index.db"), str(mounts)) == "wal"
    assert _default_journal_mode(Path("/mnt/d/.semantic_search/index.db"), str(mounts)) == "truncate"
    # Missing mounts file (non-Linux): WAL.
    assert _default_journal_mode(Path("/anywhere/index.db"), str(tmp_path / "nope")) == "wal"
