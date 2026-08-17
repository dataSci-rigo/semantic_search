import textwrap

import pytest
from PIL import Image

from image_search.config import load_config
from image_search.ingest import ingest_folder
from image_search.processors.base import (
    CaptionRecord,
    ImageEmbedRecord,
    LoadedImage,
    OcrRecord,
    Record,
    TagRecord,
    TextEmbedRecord,
)
from image_search.registry import Registry
from image_search.store import images as images_store
from image_search.store.db import connect, migrate


class FakeOcr:
    kind = "ocr"
    model_id = "fake-ocr"

    def load(self) -> None:
        pass

    def process(self, img: LoadedImage) -> list[Record]:
        return [OcrRecord(text=f"text for {img.image_id[:8]}")]


class FakeTextEmbed:
    kind = "text_embed"
    model_id = "fake-embed"

    def load(self) -> None:
        pass

    def embed(self, text: str) -> list[float]:
        # Deterministic fake embedding: length bucket into a fixed dim.
        return [float(len(text) % 7), 1.0, 0.0]

    def process(self, img: LoadedImage) -> list[Record]:
        if not img.text.strip():
            return []
        return [TextEmbedRecord(model=self.model_id, vector=self.embed(img.text))]


class FakeImageEmbed:
    kind = "image_embed"
    model_id = "fake-image-embed"

    def load(self) -> None:
        pass

    def process(self, img: LoadedImage) -> list[Record]:
        return [ImageEmbedRecord(model=self.model_id, vector=[1.0, 0.0, 0.0])]


class FakeCaption:
    kind = "caption"
    model_id = "fake-caption"

    def load(self) -> None:
        pass

    def process(self, img: LoadedImage) -> list[Record]:
        return [CaptionRecord(text=f"a photo, id {img.image_id[:8]}")]


def make_config(
    tmp_path,
    folder,
    with_ocr=True,
    with_text_embed=True,
    with_image_embed=False,
    with_caption=False,
):
    config_path = tmp_path / "folders.yaml"
    lines = ["folders:", f'  "{folder}":']
    if with_ocr:
        lines.append("    ocr: fake-ocr")
    if with_caption:
        lines.append("    caption: fake-caption")
    if with_text_embed:
        lines.append("    text_embed: fake-embed")
    if with_image_embed:
        lines.append("    image_embed: fake-image-embed")
    config_path.write_text("\n".join(lines) + "\n")
    return load_config(config_path)


def fake_registry(config):
    registry = Registry(config)
    registry._instances[("ocr", "fake-ocr")] = FakeOcr()
    registry._instances[("text_embed", "fake-embed")] = FakeTextEmbed()
    registry._instances[("image_embed", "fake-image-embed")] = FakeImageEmbed()
    registry._instances[("caption", "fake-caption")] = FakeCaption()
    return registry


def test_ingest_writes_ocr_and_fts_without_text_embed(tmp_path):
    """No vector store dependency needed: ocr-only folder config."""
    folder = tmp_path / "shots"
    folder.mkdir()
    Image.new("RGB", (4, 4)).save(folder / "one.png")

    config = make_config(tmp_path, str(folder), with_text_embed=False)
    registry = fake_registry(config)

    conn = connect(tmp_path / "test.db")
    migrate(conn)

    stats = ingest_folder(conn, config, registry, str(folder))
    assert stats == {"seen": 1, "skipped": 0, "indexed": 1, "pruned": 0, "failed": 0}

    ocr_rows = conn.execute("SELECT text FROM ocr_text").fetchall()
    assert len(ocr_rows) == 1
    assert ocr_rows[0]["text"].startswith("text for")

    fts_rows = conn.execute("SELECT * FROM text_fts").fetchall()
    assert len(fts_rows) == 1

    vec_rows = conn.execute("SELECT COUNT(*) AS n FROM vec_map").fetchone()
    assert vec_rows["n"] == 0


def test_ingest_writes_ocr_text_and_vector(tmp_path):
    pytest.importorskip("sqlite_vec", reason="requires sqlite-vec (Phase 1 dependency)")

    folder = tmp_path / "shots"
    folder.mkdir()
    Image.new("RGB", (4, 4)).save(folder / "one.png")

    config = make_config(tmp_path, str(folder))
    registry = fake_registry(config)

    conn = connect(tmp_path / "test.db")
    migrate(conn)

    stats = ingest_folder(conn, config, registry, str(folder))
    assert stats == {"seen": 1, "skipped": 0, "indexed": 1, "pruned": 0, "failed": 0}

    ocr_rows = conn.execute("SELECT text FROM ocr_text").fetchall()
    assert len(ocr_rows) == 1
    assert ocr_rows[0]["text"].startswith("text for")

    fts_rows = conn.execute("SELECT * FROM text_fts").fetchall()
    assert len(fts_rows) == 1

    vec_rows = conn.execute("SELECT COUNT(*) AS n FROM vec_map").fetchone()
    assert vec_rows["n"] == 1


def test_ingest_is_idempotent_on_rerun(tmp_path):
    pytest.importorskip("sqlite_vec", reason="requires sqlite-vec (Phase 1 dependency)")

    folder = tmp_path / "shots"
    folder.mkdir()
    Image.new("RGB", (4, 4)).save(folder / "one.png")

    config = make_config(tmp_path, str(folder))
    registry = fake_registry(config)

    conn = connect(tmp_path / "test.db")
    migrate(conn)

    first = ingest_folder(conn, config, registry, str(folder))
    second = ingest_folder(conn, config, registry, str(folder))

    assert first["indexed"] == 1
    assert second["indexed"] == 0
    assert second["skipped"] == 1

    # Records aren't duplicated on the re-run.
    assert conn.execute("SELECT COUNT(*) AS n FROM ocr_text").fetchone()["n"] == 1


def test_ingest_writes_image_embed_vector_to_its_own_space(tmp_path):
    pytest.importorskip("sqlite_vec", reason="requires sqlite-vec (Phase 1 dependency)")

    folder = tmp_path / "shots"
    folder.mkdir()
    Image.new("RGB", (4, 4)).save(folder / "one.png")

    config = make_config(tmp_path, str(folder), with_text_embed=True, with_image_embed=True)
    registry = fake_registry(config)

    conn = connect(tmp_path / "test.db")
    migrate(conn)

    stats = ingest_folder(conn, config, registry, str(folder))
    assert stats == {"seen": 1, "skipped": 0, "indexed": 1, "pruned": 0, "failed": 0}

    rows = conn.execute("SELECT vec_table, COUNT(*) AS n FROM vec_map GROUP BY vec_table").fetchall()
    tables = {r["vec_table"]: r["n"] for r in rows}
    assert tables == {
        "vec_text__fake_embed": 1,
        "vec_image__fake_image_embed": 1,
    }


def test_ingest_caption_feeds_text_embed_without_ocr(tmp_path):
    """Phase 3 criterion: an image with no OCR text is still retrievable by
    its captioned content — caption output must flow into text_embed the
    same way OCR output does."""
    pytest.importorskip("sqlite_vec", reason="requires sqlite-vec (Phase 1 dependency)")

    folder = tmp_path / "shots"
    folder.mkdir()
    Image.new("RGB", (4, 4)).save(folder / "one.png")

    config = make_config(tmp_path, str(folder), with_ocr=False, with_caption=True)
    registry = fake_registry(config)

    conn = connect(tmp_path / "test.db")
    migrate(conn)

    stats = ingest_folder(conn, config, registry, str(folder))
    assert stats == {"seen": 1, "skipped": 0, "indexed": 1, "pruned": 0, "failed": 0}

    assert conn.execute("SELECT COUNT(*) AS n FROM ocr_text").fetchone()["n"] == 0

    caption_rows = conn.execute("SELECT text FROM captions").fetchall()
    assert len(caption_rows) == 1
    assert caption_rows[0]["text"].startswith("a photo")

    fts_rows = conn.execute("SELECT * FROM text_fts").fetchall()
    assert len(fts_rows) == 1

    vec_rows = conn.execute("SELECT COUNT(*) AS n FROM vec_map").fetchone()
    assert vec_rows["n"] == 1


def test_ingest_routes_by_path_override(tmp_path):
    """A file under a nested "Screenshots" dir gets the override pipeline
    (OCR); a sibling file outside it gets the folder's base pipeline
    (caption) — both under one folder config, one ingest_folder call."""
    folder = tmp_path / "shots"
    (folder / "Screenshots").mkdir(parents=True)
    # Distinct pixel content so the two files don't collide on content-hash id.
    Image.new("RGB", (4, 4), (255, 0, 0)).save(folder / "photo.png")
    Image.new("RGB", (4, 4), (0, 255, 0)).save(folder / "Screenshots" / "shot.png")

    config_path = tmp_path / "folders.yaml"
    config_path.write_text(
        textwrap.dedent(
            f"""
            folders:
              "{folder}":
                caption: fake-caption
                overrides:
                  Screenshots:
                    ocr: fake-ocr
            """
        )
    )
    config = load_config(config_path)
    registry = fake_registry(config)

    conn = connect(tmp_path / "test.db")
    migrate(conn)

    stats = ingest_folder(conn, config, registry, str(folder))
    assert stats == {"seen": 2, "skipped": 0, "indexed": 2, "pruned": 0, "failed": 0}

    assert conn.execute("SELECT COUNT(*) AS n FROM captions").fetchone()["n"] == 1
    assert conn.execute("SELECT COUNT(*) AS n FROM ocr_text").fetchone()["n"] == 1


def _counts(conn, *tables):
    return {t: conn.execute(f"SELECT COUNT(*) AS n FROM {t}").fetchone()["n"] for t in tables}


def test_touch_rerun_does_not_duplicate(tmp_path):
    """A metadata-only mtime change must not reprocess or duplicate records."""
    import os

    folder = tmp_path / "shots"
    folder.mkdir()
    img_path = folder / "one.png"
    Image.new("RGB", (4, 4)).save(img_path)

    config = make_config(tmp_path, str(folder), with_text_embed=False)
    registry = fake_registry(config)
    conn = connect(tmp_path / "test.db")
    migrate(conn)

    ingest_folder(conn, config, registry, str(folder))
    os.utime(img_path, (img_path.stat().st_atime, img_path.stat().st_mtime + 5))
    second = ingest_folder(conn, config, registry, str(folder))

    assert second == {"seen": 1, "skipped": 1, "indexed": 0, "pruned": 0, "failed": 0}
    assert _counts(conn, "ocr_text", "text_fts") == {"ocr_text": 1, "text_fts": 1}


def test_duplicate_content_processed_once(tmp_path):
    """Byte-identical copies share one content id: processed once, and reruns
    stay quiet instead of flip-flopping on mtime forever."""
    folder = tmp_path / "shots"
    folder.mkdir()
    Image.new("RGB", (4, 4), (1, 2, 3)).save(folder / "one.png")
    Image.new("RGB", (4, 4), (1, 2, 3)).save(folder / "copy.png")

    config = make_config(tmp_path, str(folder), with_text_embed=False)
    registry = fake_registry(config)
    conn = connect(tmp_path / "test.db")
    migrate(conn)

    first = ingest_folder(conn, config, registry, str(folder))
    assert first == {"seen": 2, "skipped": 1, "indexed": 1, "pruned": 0, "failed": 0}
    assert _counts(conn, "images", "ocr_text", "text_fts") == {
        "images": 1, "ocr_text": 1, "text_fts": 1,
    }

    second = ingest_folder(conn, config, registry, str(folder))
    assert second == {"seen": 2, "skipped": 2, "indexed": 0, "pruned": 0, "failed": 0}
    assert _counts(conn, "ocr_text")["ocr_text"] == 1


def test_deleted_file_is_pruned(tmp_path):
    folder = tmp_path / "shots"
    folder.mkdir()
    Image.new("RGB", (4, 4), (255, 0, 0)).save(folder / "keep.png")
    Image.new("RGB", (4, 4), (0, 255, 0)).save(folder / "gone.png")

    config = make_config(tmp_path, str(folder), with_text_embed=False)
    registry = fake_registry(config)
    conn = connect(tmp_path / "test.db")
    migrate(conn)

    ingest_folder(conn, config, registry, str(folder))
    (folder / "gone.png").unlink()
    second = ingest_folder(conn, config, registry, str(folder))

    assert second == {"seen": 1, "skipped": 1, "indexed": 0, "pruned": 1, "failed": 0}
    assert _counts(conn, "images", "ocr_text", "text_fts", "files") == {
        "images": 1, "ocr_text": 1, "text_fts": 1, "files": 1,
    }


def test_deleting_one_duplicate_copy_keeps_content(tmp_path):
    folder = tmp_path / "shots"
    folder.mkdir()
    Image.new("RGB", (4, 4), (1, 2, 3)).save(folder / "one.png")
    Image.new("RGB", (4, 4), (1, 2, 3)).save(folder / "copy.png")

    config = make_config(tmp_path, str(folder), with_text_embed=False)
    registry = fake_registry(config)
    conn = connect(tmp_path / "test.db")
    migrate(conn)

    ingest_folder(conn, config, registry, str(folder))
    (folder / "copy.png").unlink()
    second = ingest_folder(conn, config, registry, str(folder))

    # The other path still references this content: nothing purged.
    assert second["pruned"] == 0
    assert _counts(conn, "images", "ocr_text") == {"images": 1, "ocr_text": 1}


def test_edited_file_reindexes_and_prunes_old_content(tmp_path):
    import os

    folder = tmp_path / "shots"
    folder.mkdir()
    img_path = folder / "one.png"
    Image.new("RGB", (4, 4), (255, 0, 0)).save(img_path)

    config = make_config(tmp_path, str(folder), with_text_embed=False)
    registry = fake_registry(config)
    conn = connect(tmp_path / "test.db")
    migrate(conn)

    ingest_folder(conn, config, registry, str(folder))
    old_id = conn.execute("SELECT id FROM images").fetchone()["id"]

    Image.new("RGB", (4, 4), (0, 0, 255)).save(img_path)  # new bytes, new hash
    os.utime(img_path, (img_path.stat().st_atime, img_path.stat().st_mtime + 5))
    second = ingest_folder(conn, config, registry, str(folder))

    assert second == {"seen": 1, "skipped": 0, "indexed": 1, "pruned": 1, "failed": 0}
    rows = conn.execute("SELECT id FROM images").fetchall()
    assert len(rows) == 1 and rows[0]["id"] != old_id
    assert _counts(conn, "ocr_text", "text_fts") == {"ocr_text": 1, "text_fts": 1}


def test_deleted_file_prunes_vectors(tmp_path):
    pytest.importorskip("sqlite_vec", reason="requires sqlite-vec (Phase 1 dependency)")

    folder = tmp_path / "shots"
    folder.mkdir()
    Image.new("RGB", (4, 4), (255, 0, 0)).save(folder / "keep.png")
    Image.new("RGB", (4, 4), (0, 255, 0)).save(folder / "gone.png")

    config = make_config(tmp_path, str(folder))
    registry = fake_registry(config)
    conn = connect(tmp_path / "test.db")
    migrate(conn)

    ingest_folder(conn, config, registry, str(folder))
    assert _counts(conn, "vec_map")["vec_map"] == 2

    (folder / "gone.png").unlink()
    ingest_folder(conn, config, registry, str(folder))
    assert _counts(conn, "vec_map")["vec_map"] == 1


# ---- notes & links ----------------------------------------------------------

def _fake_fetch(url):
    from image_search import textitems

    return "Example Domain", f"page text for {url}", textitems.STATUS_OK


def test_ingest_note_and_links_alongside_images(tmp_path, monkeypatch):
    from image_search import textitems

    monkeypatch.setattr(textitems, "fetch_page", _fake_fetch)

    folder = tmp_path / "saved"
    folder.mkdir()
    Image.new("RGB", (4, 4)).save(folder / "meme.png")
    (folder / "idea.md").write_text("# Big Idea\n\nwrite a meme search engine\n")
    (folder / "saved.links").write_text(
        "https://example.com/a  why it matters\nhttps://example.com/b\n"
    )

    config = make_config(tmp_path, str(folder), with_text_embed=False)
    registry = fake_registry(config)
    conn = connect(tmp_path / "test.db")
    migrate(conn)

    stats = ingest_folder(conn, config, registry, str(folder))
    assert stats == {"seen": 3, "skipped": 0, "indexed": 3, "pruned": 0, "failed": 0}

    items = conn.execute("SELECT kind, title, url FROM items ORDER BY kind, url").fetchall()
    assert [(r["kind"], r["title"]) for r in items] == [
        ("link", "Example Domain"),
        ("link", "Example Domain"),
        ("note", "Big Idea"),
    ]
    assert items[0]["url"] == "https://example.com/a"
    # 1 OCR row + 1 note + 2 links in FTS
    assert _counts(conn, "text_fts")["text_fts"] == 4

    # Idempotent rerun: nothing reprocessed.
    second = ingest_folder(conn, config, registry, str(folder))
    assert second == {"seen": 3, "skipped": 3, "indexed": 0, "pruned": 0, "failed": 0}
    assert _counts(conn, "text_fts")["text_fts"] == 4


def test_links_file_diff_adds_and_removes(tmp_path, monkeypatch):
    import os

    from image_search import textitems

    monkeypatch.setattr(textitems, "fetch_page", _fake_fetch)

    folder = tmp_path / "saved"
    folder.mkdir()
    links = folder / "saved.links"
    links.write_text("https://example.com/a\nhttps://example.com/b\n")

    config = make_config(tmp_path, str(folder), with_ocr=False, with_text_embed=False)
    registry = fake_registry(config)
    conn = connect(tmp_path / "test.db")
    migrate(conn)

    ingest_folder(conn, config, registry, str(folder))
    assert _counts(conn, "items")["items"] == 2

    links.write_text("https://example.com/b\nhttps://example.com/c\n")
    os.utime(links, (links.stat().st_atime, links.stat().st_mtime + 5))
    ingest_folder(conn, config, registry, str(folder))

    urls = [r["url"] for r in conn.execute("SELECT url FROM items ORDER BY url")]
    assert urls == ["https://example.com/b", "https://example.com/c"]
    assert _counts(conn, "text_fts")["text_fts"] == 2


def test_edited_note_reindexes_and_prunes_old(tmp_path):
    import os

    folder = tmp_path / "saved"
    folder.mkdir()
    note = folder / "idea.md"
    note.write_text("# Old Title\n\nold body\n")

    config = make_config(tmp_path, str(folder), with_ocr=False, with_text_embed=False)
    registry = fake_registry(config)
    conn = connect(tmp_path / "test.db")
    migrate(conn)

    ingest_folder(conn, config, registry, str(folder))
    note.write_text("# New Title\n\nnew body\n")
    os.utime(note, (note.stat().st_atime, note.stat().st_mtime + 5))
    second = ingest_folder(conn, config, registry, str(folder))

    assert second == {"seen": 1, "skipped": 0, "indexed": 1, "pruned": 1, "failed": 0}
    rows = conn.execute("SELECT title FROM items").fetchall()
    assert [r["title"] for r in rows] == ["New Title"]
    assert _counts(conn, "text_fts")["text_fts"] == 1


def test_deleted_links_file_prunes_its_items(tmp_path, monkeypatch):
    from image_search import textitems

    monkeypatch.setattr(textitems, "fetch_page", _fake_fetch)

    folder = tmp_path / "saved"
    folder.mkdir()
    links = folder / "saved.links"
    links.write_text("https://example.com/a\n")

    config = make_config(tmp_path, str(folder), with_ocr=False, with_text_embed=False)
    registry = fake_registry(config)
    conn = connect(tmp_path / "test.db")
    migrate(conn)

    ingest_folder(conn, config, registry, str(folder))
    assert _counts(conn, "items")["items"] == 1

    links.unlink()
    second = ingest_folder(conn, config, registry, str(folder))
    assert second["pruned"] == 1
    assert _counts(conn, "items", "text_fts") == {"items": 0, "text_fts": 0}


def test_note_and_link_text_vectors(tmp_path, monkeypatch):
    pytest.importorskip("sqlite_vec", reason="requires sqlite-vec (Phase 1 dependency)")
    from image_search import textitems

    monkeypatch.setattr(textitems, "fetch_page", _fake_fetch)

    folder = tmp_path / "saved"
    folder.mkdir()
    (folder / "idea.md").write_text("# Idea\n\nbody\n")
    (folder / "saved.links").write_text("https://example.com/a\n")

    config = make_config(tmp_path, str(folder), with_ocr=False, with_text_embed=True)
    registry = fake_registry(config)
    conn = connect(tmp_path / "test.db")
    migrate(conn)

    ingest_folder(conn, config, registry, str(folder))
    rows = conn.execute(
        "SELECT vec_table, COUNT(*) AS n FROM vec_map GROUP BY vec_table"
    ).fetchall()
    assert {r["vec_table"]: r["n"] for r in rows} == {"vec_text__fake_embed": 2}


# ---- per-file error isolation ----------------------------------------------

class FlakyOcr:
    """OCR that raises on chosen image ids — stands in for a corrupt file or a
    transient OCR/caption worker hiccup."""

    kind = "ocr"
    model_id = "fake-ocr"

    def __init__(self, fail_ids=(), fail_after_record=False):
        self.fail_ids = set(fail_ids)
        self.fail_after_record = fail_after_record

    def load(self) -> None:
        pass

    def process(self, img: LoadedImage) -> list[Record]:
        if img.image_id in self.fail_ids:
            if self.fail_after_record:
                # Emit a record first, so the caller has already written rows
                # when the failure lands (exercises the rollback path).
                return [OcrRecord(text="partial"), _Boom()]
            raise RuntimeError("simulated corrupt-image decode failure")
        return [OcrRecord(text=f"text for {img.image_id[:8]}")]


class _Boom:
    """A record type ingest doesn't handle -> NotImplementedError mid-file,
    after earlier records in the same file were already written."""


def _three_images(tmp_path):
    folder = tmp_path / "shots"
    folder.mkdir()
    ids = []
    for i, color in enumerate([(10, 0, 0), (0, 10, 0), (0, 0, 10)]):
        p = folder / f"img{i}.png"
        Image.new("RGB", (4, 4), color).save(p)
        ids.append(images_store.content_hash(p))
    return folder, ids


def test_one_failing_file_does_not_end_the_run(tmp_path):
    """The reported blocker: a processor failure on file 2 of 3 must not stop
    file 3 from being indexed."""
    folder, ids = _three_images(tmp_path)

    config = make_config(tmp_path, str(folder), with_text_embed=False)
    registry = Registry(config)
    registry._instances[("ocr", "fake-ocr")] = FlakyOcr(fail_ids=[ids[1]])

    conn = connect(tmp_path / "test.db")
    migrate(conn)

    stats = ingest_folder(conn, config, registry, str(folder))
    assert stats == {"seen": 3, "skipped": 0, "indexed": 2, "pruned": 0, "failed": 1}

    indexed = {r["id"] for r in conn.execute("SELECT id FROM images")}
    assert indexed == {ids[0], ids[2]}  # the third file was still attempted


def test_failed_file_leaves_no_partial_rows_and_is_retried(tmp_path):
    """Rollback: a file that fails after writing records must not leak those
    rows into the next file's commit, and must be retried on a later run."""
    folder, ids = _three_images(tmp_path)

    config = make_config(tmp_path, str(folder), with_text_embed=False)
    registry = Registry(config)
    flaky = FlakyOcr(fail_ids=[ids[1]], fail_after_record=True)
    registry._instances[("ocr", "fake-ocr")] = flaky

    conn = connect(tmp_path / "test.db")
    migrate(conn)

    stats = ingest_folder(conn, config, registry, str(folder))
    assert stats["failed"] == 1 and stats["indexed"] == 2

    # No debris from the failed file anywhere.
    for table in ("images", "files", "ocr_text", "text_fts"):
        rows = conn.execute(f"SELECT * FROM {table} WHERE image_id = ?"
                            if table in ("ocr_text", "text_fts")
                            else f"SELECT * FROM {table} WHERE {'id' if table == 'images' else 'image_id'} = ?",
                            (ids[1],)).fetchall()
        assert rows == [], f"{table} kept rows for the failed file"
    assert conn.execute("SELECT COUNT(*) AS n FROM ocr_text").fetchone()["n"] == 2

    # Once the fault clears, the retried file indexes normally.
    flaky.fail_ids = set()
    second = ingest_folder(conn, config, registry, str(folder))
    assert second == {"seen": 3, "skipped": 2, "indexed": 1, "pruned": 0, "failed": 0}
    assert conn.execute("SELECT COUNT(*) AS n FROM images").fetchone()["n"] == 3


def test_failed_edit_keeps_previously_indexed_version(tmp_path):
    """A file edited into unprocessable content must not lose what was already
    indexed for it, and must not be pruned as if it had been deleted."""
    import os

    folder, ids = _three_images(tmp_path)

    config = make_config(tmp_path, str(folder), with_text_embed=False)
    registry = Registry(config)
    ocr = FlakyOcr()
    registry._instances[("ocr", "fake-ocr")] = ocr

    conn = connect(tmp_path / "test.db")
    migrate(conn)
    ingest_folder(conn, config, registry, str(folder))

    # Rewrite file 2 with new content (new content id -> genuinely
    # reprocessed) and make that new content fail.
    p = folder / "img1.png"
    Image.new("RGB", (4, 4), (7, 7, 7)).save(p)
    os.utime(p, (p.stat().st_atime, p.stat().st_mtime + 5))
    ocr.fail_ids = {images_store.content_hash(p)}

    second = ingest_folder(conn, config, registry, str(folder))
    assert second["failed"] == 1
    assert second["pruned"] == 0
    # The old version survives: rollback left its files row pointing at the
    # previously indexed content, so prune_missing sees it as still referenced.
    assert conn.execute("SELECT COUNT(*) AS n FROM images").fetchone()["n"] == 3
    assert conn.execute(
        "SELECT 1 FROM images WHERE id = ?", (ids[1],)
    ).fetchone() is not None


def test_failing_folder_does_not_stop_other_folders(tmp_path, monkeypatch):
    """A folder that throws during discovery (unreadable path, unmounted
    drive) must not kill the whole run."""
    from image_search import ingest as ingest_mod

    good = tmp_path / "good"
    good.mkdir()
    Image.new("RGB", (4, 4)).save(good / "one.png")
    bad = tmp_path / "bad"
    bad.mkdir()

    config_path = tmp_path / "folders.yaml"
    config_path.write_text(
        f'folders:\n  "{bad}":\n    ocr: fake-ocr\n  "{good}":\n    ocr: fake-ocr\n'
    )
    config = load_config(config_path)
    registry = fake_registry(config)

    real_walk = images_store.walk_candidates

    def exploding_walk(folder_path):
        if folder_path == bad:
            raise PermissionError("simulated unreadable folder")
        return real_walk(folder_path)

    monkeypatch.setattr(ingest_mod.images_store, "walk_candidates", exploding_walk)

    conn = connect(tmp_path / "test.db")
    migrate(conn)
    stats = ingest_mod.ingest_all(conn, config, registry)

    assert "error" in stats[str(bad)]
    assert stats[str(good)]["indexed"] == 1
    assert conn.execute("SELECT COUNT(*) AS n FROM images").fetchone()["n"] == 1


# ---- content-based routing --------------------------------------------------

class FakeTagger:
    """Emits a fixed label ranking per image id, so routing is deterministic."""

    kind = "tagger"
    model_id = "fake-clip"

    def __init__(self, labels_by_id):
        self.labels_by_id = labels_by_id

    def load(self) -> None:
        pass

    def process(self, img: LoadedImage) -> list[Record]:
        labels = self.labels_by_id.get(img.image_id, ["photo", "art"])
        return [
            TagRecord(tag=t, score=1.0 - i * 0.05, source="clip-zs")
            for i, t in enumerate(labels)
        ]


class RecordingCaption(FakeCaption):
    def __init__(self):
        self.seen = []

    def process(self, img: LoadedImage) -> list[Record]:
        self.seen.append(img.image_id)
        return super().process(img)


class RecordingOcr(FakeOcr):
    def __init__(self):
        self.seen = []

    def process(self, img: LoadedImage) -> list[Record]:
        self.seen.append(img.image_id)
        return super().process(img)


def _routed_setup(tmp_path, labels_by_id_builder):
    folder = tmp_path / "mixed"
    folder.mkdir()
    paths = {}
    for name, color in (("meme", (200, 0, 0)), ("photo", (0, 120, 200))):
        p = folder / f"{name}.png"
        Image.new("RGB", (4, 4), color).save(p)
        paths[name] = images_store.content_hash(p)

    config_path = tmp_path / "folders.yaml"
    config_path.write_text(
        f'folders:\n  "{folder}":\n'
        "    route: auto\n"
        "    tagger: fake-clip\n"
        "    ocr: fake-ocr\n"
        "    caption: fake-caption\n"
    )
    config = load_config(config_path)

    registry = Registry(config)
    ocr, caption = RecordingOcr(), RecordingCaption()
    registry._instances[("ocr", "fake-ocr")] = ocr
    registry._instances[("caption", "fake-caption")] = caption
    registry._instances[("tagger", "fake-clip")] = FakeTagger(
        labels_by_id_builder(paths)
    )
    return folder, paths, config, registry, ocr, caption


def test_routing_gives_text_images_ocr_and_caption_but_photos_only_caption(tmp_path):
    folder, ids, config, registry, ocr, caption = _routed_setup(
        tmp_path,
        lambda p: {
            p["meme"]: ["meme", "art"],
            p["photo"]: ["photo", "art"],
        },
    )
    conn = connect(tmp_path / "test.db")
    migrate(conn)

    ingest_folder(conn, config, registry, str(folder))

    # The meme gets both text paths; the photo is captioned but not OCR'd.
    assert ocr.seen == [ids["meme"]]
    assert sorted(caption.seen) == sorted([ids["meme"], ids["photo"]])


def test_routing_writes_ranked_tags(tmp_path):
    folder, ids, config, registry, _, _ = _routed_setup(
        tmp_path,
        lambda p: {p["meme"]: ["meme", "chart"], p["photo"]: ["photo", "art"]},
    )
    conn = connect(tmp_path / "test.db")
    migrate(conn)
    ingest_folder(conn, config, registry, str(folder))

    rows = conn.execute(
        "SELECT tag, rank, source FROM tags WHERE image_id = ? ORDER BY rank",
        (ids["meme"],),
    ).fetchall()
    assert [(r["tag"], r["rank"]) for r in rows] == [("meme", 1), ("chart", 2)]
    assert {r["source"] for r in rows} == {"clip-zs"}


def test_no_route_auto_runs_every_configured_processor(tmp_path):
    """Without route: auto, behavior is exactly as before — no gating."""
    folder = tmp_path / "mixed"
    folder.mkdir()
    p = folder / "photo.png"
    Image.new("RGB", (4, 4), (0, 120, 200)).save(p)

    config_path = tmp_path / "folders.yaml"
    config_path.write_text(
        f'folders:\n  "{folder}":\n    ocr: fake-ocr\n    caption: fake-caption\n'
    )
    config = load_config(config_path)
    registry = Registry(config)
    ocr, caption = RecordingOcr(), RecordingCaption()
    registry._instances[("ocr", "fake-ocr")] = ocr
    registry._instances[("caption", "fake-caption")] = caption

    conn = connect(tmp_path / "test.db")
    migrate(conn)
    ingest_folder(conn, config, registry, str(folder))

    assert len(ocr.seen) == 1 and len(caption.seen) == 1
