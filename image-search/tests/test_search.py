import textwrap

import pytest

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


def test_search_returns_note_and_link_hits(tmp_path):
    folder = str(tmp_path / "saved")
    config = make_config_no_embed(tmp_path, folder)
    registry = Registry(config)

    conn = connect(tmp_path / "test.db")
    migrate(conn)
    conn.execute(
        "INSERT INTO items (id, kind, folder, src_path, title, url, body) "
        "VALUES ('note1', 'note', ?, '/x/idea.md', 'Big Idea', NULL, 'meme search engine')",
        (folder,),
    )
    conn.execute(
        "INSERT INTO items (id, kind, folder, src_path, title, url, body) "
        "VALUES ('link1', 'link', ?, '/x/saved.links', 'Example', "
        "'https://example.com', 'meme reference page')",
        (folder,),
    )
    conn.execute("INSERT INTO text_fts (image_id, text) VALUES ('note1', 'meme search engine')")
    conn.execute("INSERT INTO text_fts (image_id, text) VALUES ('link1', 'meme reference page')")
    conn.commit()

    hits = search_text(conn, config, registry, folder, "meme")
    by_kind = {h.kind: h for h in hits}
    assert set(by_kind) == {"note", "link"}
    assert by_kind["note"].title == "Big Idea"
    assert by_kind["note"].snippet == "meme search engine"
    assert by_kind["link"].url == "https://example.com"


def _tag(conn, image_id, tag, rank):
    conn.execute(
        "INSERT INTO tags (image_id, tag, source, score, rank) VALUES (?, ?, 'clip-zs', 0.4, ?)",
        (image_id, tag, rank),
    )


def test_query_facet_filters_by_image_type(tmp_path):
    """Spec 7.3 worked example: 'population graphs' filters to charts and
    ranks on 'population'."""
    from image_search.search import parse_facets

    folder = str(tmp_path / "mixed")
    config = make_config_no_embed(tmp_path, folder)
    registry = Registry(config)

    conn = connect(tmp_path / "test.db")
    migrate(conn)
    _insert_image(conn, "chart1", folder, "world population growth by year")
    _insert_image(conn, "photo1", folder, "population of the island beach resort")
    _tag(conn, "chart1", "chart", 1)
    _tag(conn, "photo1", "photo", 1)
    conn.commit()

    # The facet word is consumed as a filter, not searched for.
    assert parse_facets("population graphs") == ("population", {"chart"})

    hits = search_text(conn, config, registry, folder, "population graphs")
    assert [h.image_id for h in hits] == ["chart1"]

    # Without the facet word, both match.
    hits = search_text(conn, config, registry, folder, "population")
    assert {h.image_id for h in hits} == {"chart1", "photo1"}


def test_explicit_tag_argument_filters(tmp_path):
    folder = str(tmp_path / "mixed")
    config = make_config_no_embed(tmp_path, folder)
    registry = Registry(config)

    conn = connect(tmp_path / "test.db")
    migrate(conn)
    _insert_image(conn, "meme1", folder, "stock market numbers money")
    _insert_image(conn, "shot1", folder, "stock market numbers dashboard")
    _tag(conn, "meme1", "meme", 1)
    _tag(conn, "shot1", "screenshot", 1)
    conn.commit()

    hits = search_text(conn, config, registry, folder, "stock market", tags={"meme"})
    assert [h.image_id for h in hits] == ["meme1"]


def test_facet_matches_second_ranked_tag(tmp_path):
    """Rank <= 2 counts: label margins are narrow, so a strict argmax would
    drop genuine charts."""
    folder = str(tmp_path / "mixed")
    config = make_config_no_embed(tmp_path, folder)
    registry = Registry(config)

    conn = connect(tmp_path / "test.db")
    migrate(conn)
    _insert_image(conn, "diagram1", folder, "retrieval filtering scoring ordering")
    _tag(conn, "diagram1", "screenshot", 1)
    _tag(conn, "diagram1", "chart", 2)
    _tag(conn, "diagram1", "document", 3)
    conn.commit()

    assert [h.image_id for h in search_text(
        conn, config, registry, folder, "retrieval", tags={"chart"})] == ["diagram1"]
    # Rank 3 is beyond the cutoff.
    assert search_text(conn, config, registry, folder, "retrieval", tags={"document"}) == []


def _insert_image_row(conn, image_id, folder):
    conn.execute(
        "INSERT INTO images (id, path, folder, content_hash, mtime, width, height, indexed_at) "
        "VALUES (?, ?, ?, ?, 0, 4, 4, 0)",
        (image_id, f"/x/{image_id}.png", folder, image_id),
    )


def _insert_fts(conn, image_id, text, source):
    conn.execute(
        "INSERT INTO text_fts (image_id, text, source) VALUES (?, ?, ?)",
        (image_id, text, source),
    )


def test_field_ocr_only_excludes_caption_matches(tmp_path):
    folder = str(tmp_path / "mixed")
    config = make_config_no_embed(tmp_path, folder)
    registry = Registry(config)

    conn = connect(tmp_path / "test.db")
    migrate(conn)
    _insert_image_row(conn, "ocr1", folder)
    _insert_image_row(conn, "cap1", folder)
    _insert_fts(conn, "ocr1", "unemployment chart", "ocr")
    _insert_fts(conn, "cap1", "unemployment chart", "caption")
    conn.commit()

    hits = search_text(conn, config, registry, folder, "unemployment", field="ocr")
    assert [h.image_id for h in hits] == ["ocr1"]


def test_field_caption_only_excludes_ocr_matches(tmp_path):
    folder = str(tmp_path / "mixed")
    config = make_config_no_embed(tmp_path, folder)
    registry = Registry(config)

    conn = connect(tmp_path / "test.db")
    migrate(conn)
    _insert_image_row(conn, "ocr1", folder)
    _insert_image_row(conn, "cap1", folder)
    _insert_fts(conn, "ocr1", "unemployment chart", "ocr")
    _insert_fts(conn, "cap1", "unemployment chart", "caption")
    conn.commit()

    hits = search_text(conn, config, registry, folder, "unemployment", field="caption")
    assert [h.image_id for h in hits] == ["cap1"]


def test_field_none_returns_both(tmp_path):
    folder = str(tmp_path / "mixed")
    config = make_config_no_embed(tmp_path, folder)
    registry = Registry(config)

    conn = connect(tmp_path / "test.db")
    migrate(conn)
    _insert_image_row(conn, "ocr1", folder)
    _insert_image_row(conn, "cap1", folder)
    _insert_fts(conn, "ocr1", "unemployment chart", "ocr")
    _insert_fts(conn, "cap1", "unemployment chart", "caption")
    conn.commit()

    hits = search_text(conn, config, registry, folder, "unemployment")
    assert {h.image_id for h in hits} == {"ocr1", "cap1"}


def test_field_does_not_hide_items(tmp_path):
    folder = str(tmp_path / "mixed")
    config = make_config_no_embed(tmp_path, folder)
    registry = Registry(config)

    conn = connect(tmp_path / "test.db")
    migrate(conn)
    conn.execute(
        "INSERT INTO items (id, kind, folder, src_path, title, url, body) "
        "VALUES ('note1', 'note', ?, '/x/idea.md', 'Big Idea', NULL, 'unemployment thoughts')",
        (folder,),
    )
    # No source column value given -> NULL, mirroring how textitems.py inserts items.
    conn.execute(
        "INSERT INTO text_fts (image_id, text) VALUES ('note1', 'unemployment thoughts')"
    )
    conn.commit()

    for field in ("ocr", "caption"):
        hits = search_text(conn, config, registry, folder, "unemployment", field=field)
        assert [h.image_id for h in hits] == ["note1"]


def test_invalid_field_raises(tmp_path):
    folder = str(tmp_path / "shots")
    config = make_config_no_embed(tmp_path, folder)
    registry = Registry(config)
    conn = connect(tmp_path / "test.db")
    migrate(conn)
    with pytest.raises(ValueError):
        search_text(conn, config, registry, folder, "x", field="bogus")
