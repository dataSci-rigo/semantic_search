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
