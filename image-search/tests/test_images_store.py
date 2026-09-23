import pytest
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


def test_walk_candidates_is_stat_only_and_filters_extensions(tmp_path):
    folder = tmp_path / "shots"
    folder.mkdir()
    make_image(folder / "one.png")
    (folder / "note.txt").write_text("a note — ingestible")
    (folder / "junk.log").write_text("not ingestible")

    walked = images_store.walk_candidates(folder)
    assert [p.name for p, _ in walked] == ["note.txt", "one.png"]
    for path, mtime in walked:
        assert mtime == path.stat().st_mtime


def test_walk_candidates_missing_folder_raises(tmp_path):
    # An empty return would read as "everything was deleted" and let
    # prune_missing wipe the folder's index when its drive is unplugged.
    with pytest.raises(FileNotFoundError):
        images_store.walk_candidates(tmp_path / "does-not-exist")


def test_walk_candidates_skips_excluded_dirs(tmp_path):
    root = tmp_path / "drive"
    (root / "photos").mkdir(parents=True)
    (root / "$RECYCLE.BIN" / "sub").mkdir(parents=True)
    (root / ".trash-1000").mkdir()  # case-insensitive match of .Trash-*
    make_image(root / "photos" / "keep.png")
    make_image(root / "$RECYCLE.BIN" / "sub" / "junk.png")
    make_image(root / ".trash-1000" / "junk2.png")

    walked = images_store.walk_candidates(
        root, exclude_dirs=("$RECYCLE.BIN", ".Trash-*")
    )
    assert [p.name for p, _ in walked] == ["keep.png"]


def test_describe_hashes_and_reads_dims(tmp_path):
    folder = tmp_path / "shots"
    folder.mkdir()
    path = folder / "one.png"
    make_image(path)
    [(walked_path, mtime)] = images_store.walk_candidates(folder)

    disc = images_store.describe(walked_path, "shots", mtime)
    assert disc.width == 4 and disc.height == 4
    assert disc.image_id == images_store.content_hash(path)
    assert disc.folder == "shots"
    assert disc.mtime == mtime


def test_file_state_roundtrip(tmp_path):
    conn = connect(tmp_path / "test.db")
    migrate(conn)

    images_store.upsert_file(conn, "/x/one.png", "shots", "id1", 100.0)
    assert images_store.load_file_state(conn, "shots") == {"/x/one.png": ("id1", 100.0)}
    assert images_store.load_file_state(conn, "other") == {}

    # Upsert on the same path replaces, never duplicates.
    images_store.upsert_file(conn, "/x/one.png", "shots", "id2", 200.0)
    assert images_store.load_file_state(conn, "shots") == {"/x/one.png": ("id2", 200.0)}


def test_upsert_is_idempotent_on_content_id(tmp_path):
    conn = connect(tmp_path / "test.db")
    migrate(conn)

    folder = tmp_path / "shots"
    folder.mkdir()
    make_image(folder / "one.png")
    [(path, mtime)] = images_store.walk_candidates(folder)
    disc = images_store.describe(path, "shots", mtime)

    images_store.upsert_image(conn, disc, "one.png")
    images_store.upsert_image(conn, disc, "one.png")
    conn.commit()

    count = conn.execute("SELECT COUNT(*) AS n FROM images").fetchone()["n"]
    assert count == 1
    assert images_store.is_indexed(conn, disc.image_id)
    assert not images_store.is_indexed(conn, "nope")


def test_purge_image_removes_derived_records(tmp_path):
    conn = connect(tmp_path / "test.db")
    migrate(conn)

    conn.execute(
        "INSERT INTO images (id, path, folder, content_hash) VALUES ('id1', '/x', 'f', 'id1')"
    )
    conn.execute("INSERT INTO ocr_text (image_id, model, text) VALUES ('id1', 'm', 't')")
    conn.execute("INSERT INTO captions (image_id, model, text) VALUES ('id1', 'm', 't')")
    conn.execute("INSERT INTO text_fts (image_id, text) VALUES ('id1', 't')")

    images_store.purge_image(conn, "id1")
    conn.commit()

    for table in ("images", "ocr_text", "captions", "text_fts"):
        assert conn.execute(f"SELECT COUNT(*) AS n FROM {table}").fetchone()["n"] == 0


def test_duplicate_groups(tmp_path):
    conn = connect(tmp_path / "test.db")
    migrate(conn)

    images_store.upsert_file(conn, "x/b.png", "f", "dup-id", 1.0)
    images_store.upsert_file(conn, "x/a.png", "f", "dup-id", 1.0)
    images_store.upsert_file(conn, "x/unique.png", "f", "other-id", 1.0)

    groups = images_store.duplicate_groups(conn)
    assert groups == [("dup-id", [("f", "x/a.png"), ("f", "x/b.png")])]


def test_walk_candidates_skips_lock_and_appledouble_files_and_sees_tif(tmp_path):
    root = tmp_path / "d"
    root.mkdir()
    make_image(root / "photo.JPG")
    Image.new("RGB", (4, 4)).save(root / "scan.tif")
    (root / "~$report.docx").write_bytes(b"\x00" * 162)  # Office owner/lock file
    (root / "._photo.JPG").write_bytes(b"\x00\x05\x16\x07")  # macOS AppleDouble fork
    walked = [p.name for p, _ in images_store.walk_candidates(root)]
    assert walked == ["photo.JPG", "scan.tif"]


def test_default_exclude_dirs_skip_the_index_metadata_dir(tmp_path):
    from image_search.config import DEFAULT_EXCLUDE_DIRS

    root = tmp_path / "drive"
    (root / ".semantic_search").mkdir(parents=True)
    (root / ".semantic_search" / "scratch.txt").write_text("index working file")
    (root / "real.txt").write_text("a real note")
    walked = [p.name for p, _ in images_store.walk_candidates(root, DEFAULT_EXCLUDE_DIRS)]
    assert walked == ["real.txt"]
