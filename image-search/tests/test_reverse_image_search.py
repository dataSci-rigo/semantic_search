import textwrap

import pytest

from image_search.config import load_config
from image_search.search import search_similar_images
from image_search.store import vectors as vectors_store
from image_search.store.db import connect, migrate

pytestmark = pytest.mark.filterwarnings("ignore")


def _skip_without_sqlite_vec():
    pytest.importorskip("sqlite_vec", reason="requires sqlite-vec (Phase 1+ dependency)")


def make_config(tmp_path, folder):
    config_path = tmp_path / "folders.yaml"
    config_path.write_text(
        textwrap.dedent(
            f"""
            folders:
              "{folder}":
                image_embed: fake-image-embed
            """
        )
    )
    return load_config(config_path)


def insert_image_row(conn, image_id, folder, path="x.png"):
    conn.execute(
        "INSERT INTO images (id, path, folder, content_hash, mtime, width, height, indexed_at) "
        "VALUES (?, ?, ?, ?, 0, 4, 4, 0)",
        (image_id, path, folder, image_id),
    )


def test_get_vector_roundtrips_through_vec0(tmp_path):
    _skip_without_sqlite_vec()
    conn = connect(tmp_path / "test.db")
    migrate(conn)

    vectors_store.insert_vector(conn, "image", "fake-image-embed", "img1", [0.1, 0.2, 0.3])
    got = vectors_store.get_vector(conn, "image", "fake-image-embed", "img1")
    assert got == pytest.approx([0.1, 0.2, 0.3], abs=1e-6)


def test_get_vector_missing_returns_none(tmp_path):
    _skip_without_sqlite_vec()
    conn = connect(tmp_path / "test.db")
    migrate(conn)
    assert vectors_store.get_vector(conn, "image", "fake-image-embed", "nope") is None


def test_search_similar_images_excludes_self_and_ranks_by_distance(tmp_path):
    _skip_without_sqlite_vec()
    folder = str(tmp_path / "shots")
    config = make_config(tmp_path, folder)

    conn = connect(tmp_path / "test.db")
    migrate(conn)

    # img1 is the query; img2 is close to it; img3 is far.
    vectors_store.insert_vector(conn, "image", "fake-image-embed", "img1", [1.0, 0.0, 0.0])
    vectors_store.insert_vector(conn, "image", "fake-image-embed", "img2", [0.9, 0.1, 0.0])
    vectors_store.insert_vector(conn, "image", "fake-image-embed", "img3", [-1.0, 0.0, 0.0])
    for iid in ("img1", "img2", "img3"):
        insert_image_row(conn, iid, folder)
    conn.commit()

    hits = search_similar_images(conn, config, folder, "img1")
    assert [h.image_id for h in hits] == ["img2", "img3"]


def test_search_similar_images_requires_image_embed_configured(tmp_path):
    _skip_without_sqlite_vec()
    folder = str(tmp_path / "shots")
    config_path = tmp_path / "folders.yaml"
    config_path.write_text(f'folders:\n  "{folder}":\n    ocr: rapidocr\n')
    config = load_config(config_path)

    conn = connect(tmp_path / "test.db")
    migrate(conn)

    with pytest.raises(ValueError, match="image_embed"):
        search_similar_images(conn, config, folder, "img1")


def test_search_similar_images_requires_indexed_image(tmp_path):
    _skip_without_sqlite_vec()
    folder = str(tmp_path / "shots")
    config = make_config(tmp_path, folder)

    conn = connect(tmp_path / "test.db")
    migrate(conn)

    with pytest.raises(ValueError, match="No .* image vector"):
        search_similar_images(conn, config, folder, "never-indexed")
