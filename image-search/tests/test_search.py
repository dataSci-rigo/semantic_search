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
