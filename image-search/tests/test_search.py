import textwrap

from image_search.config import load_config
from image_search.registry import Registry
from image_search.search import search_text
from image_search.store.db import connect, migrate


def make_config_no_embed(tmp_path, folder):
    config_path = tmp_path / "folders.yaml"
    config_path.write_text(
        textwrap.dedent(
            f"""
            folders:
              "{folder}":
                ocr: fake-ocr
            """
        )
    )
    return load_config(config_path)


def test_fts_fallback_finds_keyword_match(tmp_path):
    """No text_embed configured -> pure FTS5 keyword path, no sqlite-vec needed."""
    folder = str(tmp_path / "shots")
    config = make_config_no_embed(tmp_path, folder)
    registry = Registry(config)

    conn = connect(tmp_path / "test.db")
    migrate(conn)

    conn.execute(
        "INSERT INTO images (id, path, folder, content_hash, mtime, width, height, indexed_at) "
        "VALUES ('img1', '/x/one.png', ?, 'img1', 0, 4, 4, 0)",
        (folder,),
    )
    conn.execute(
        "INSERT INTO text_fts (image_id, text) VALUES ('img1', 'quarterly unemployment chart')"
    )
    conn.commit()

    hits = search_text(conn, config, registry, folder, "unemployment")
    assert len(hits) == 1
    assert hits[0].image_id == "img1"
    assert hits[0].source == "fts"


def test_no_hits_for_unmatched_query(tmp_path):
    folder = str(tmp_path / "shots")
    config = make_config_no_embed(tmp_path, folder)
    registry = Registry(config)

    conn = connect(tmp_path / "test.db")
    migrate(conn)
    conn.execute(
        "INSERT INTO text_fts (image_id, text) VALUES ('img1', 'quarterly unemployment chart')"
    )
    conn.commit()

    hits = search_text(conn, config, registry, folder, "giraffe")
    assert hits == []


def make_config_two_folders(tmp_path, folder_a, folder_b):
    config_path = tmp_path / "folders.yaml"
    config_path.write_text(
        textwrap.dedent(
            f"""
            folders:
              "{folder_a}":
                ocr: fake-ocr
              "{folder_b}":
                ocr: fake-ocr
            """
        )
    )
    return load_config(config_path)


def _insert_image(conn, image_id, folder, text):
    conn.execute(
        "INSERT INTO images (id, path, folder, content_hash, mtime, width, height, indexed_at) "
        "VALUES (?, ?, ?, ?, 0, 4, 4, 0)",
        (image_id, f"/x/{image_id}.png", folder, image_id),
    )
    conn.execute("INSERT INTO text_fts (image_id, text) VALUES (?, ?)", (image_id, text))


def test_search_is_folder_scoped(tmp_path):
    folder_a = str(tmp_path / "a")
    folder_b = str(tmp_path / "b")
    config = make_config_two_folders(tmp_path, folder_a, folder_b)
    registry = Registry(config)

    conn = connect(tmp_path / "test.db")
    migrate(conn)
    _insert_image(conn, "img-a", folder_a, "quarterly unemployment chart")
    _insert_image(conn, "img-b", folder_b, "quarterly unemployment chart")
    conn.commit()

    hits = search_text(conn, config, registry, folder_a, "unemployment")
    assert [h.image_id for h in hits] == ["img-a"]

    hits = search_text(conn, config, registry, folder_b, "unemployment")
    assert [h.image_id for h in hits] == ["img-b"]


def test_query_with_embedded_quotes_does_not_error(tmp_path):
    folder = str(tmp_path / "shots")
    config = make_config_no_embed(tmp_path, folder)
    registry = Registry(config)

    conn = connect(tmp_path / "test.db")
    migrate(conn)
    conn.execute(
        "INSERT INTO text_fts (image_id, text) VALUES ('img1', 'quarterly unemployment chart')"
    )
    conn.commit()

    # Previously raised sqlite3.OperationalError from malformed FTS5 syntax.
    assert search_text(conn, config, registry, folder, 'say "hi" now') == []
