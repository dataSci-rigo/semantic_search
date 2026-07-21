from PIL import Image

from image_search.store import images as images_store
from image_search.store.db import connect, migrate


def make_image(path, color=(255, 0, 0)):
    Image.new("RGB", (4, 4), color).save(path)


def test_content_hash_is_deterministic_and_content_addressed(tmp_path):
    a = tmp_path / "a.png"
    b = tmp_path / "b.png"
    make_image(a, (10, 20, 30))
    make_image(b, (10, 20, 30))  # identical bytes, different path
    assert images_store.content_hash(a) == images_store.content_hash(b)

    c = tmp_path / "c.png"
    make_image(c, (99, 99, 99))
    assert images_store.content_hash(a) != images_store.content_hash(c)


def test_discover_finds_images_and_reads_dims(tmp_path):
    folder = tmp_path / "shots"
    folder.mkdir()
    make_image(folder / "one.png")
    (folder / "not-an-image.txt").write_text("hi")

    found = images_store.discover(folder, "shots")
    assert len(found) == 1
    assert found[0].width == 4 and found[0].height == 4
    assert found[0].folder == "shots"


def test_discover_missing_folder_returns_empty(tmp_path):
    assert images_store.discover(tmp_path / "does-not-exist", "x") == []


def test_dedupe_on_unchanged_mtime(tmp_path):
    conn = connect(tmp_path / "test.db")
    migrate(conn)

    folder = tmp_path / "shots"
    folder.mkdir()
    make_image(folder / "one.png")
    [disc] = images_store.discover(folder, "shots")

    assert images_store.is_unchanged(conn, disc.image_id, disc.mtime) is False
    images_store.upsert_image(conn, disc)
    conn.commit()
    assert images_store.is_unchanged(conn, disc.image_id, disc.mtime) is True


def test_upsert_is_idempotent_on_content_id(tmp_path):
    conn = connect(tmp_path / "test.db")
    migrate(conn)

    folder = tmp_path / "shots"
    folder.mkdir()
    make_image(folder / "one.png")
    [disc] = images_store.discover(folder, "shots")

    images_store.upsert_image(conn, disc)
    images_store.upsert_image(conn, disc)
    conn.commit()

    count = conn.execute("SELECT COUNT(*) AS n FROM images").fetchone()["n"]
    assert count == 1
