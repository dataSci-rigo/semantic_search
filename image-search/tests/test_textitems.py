from image_search import textitems


def test_parse_note_title_from_heading(tmp_path):
    note = tmp_path / "idea.md"
    note.write_text("some preamble\n# The Real Title\nbody text\n")
    title, body = textitems.parse_note(note)
    assert title == "some preamble"  # first non-empty line wins unless it's a heading
    note.write_text("# The Real Title\n\nbody text\n")
    title, body = textitems.parse_note(note)
    assert title == "The Real Title"
    assert "body text" in body


def test_parse_note_title_falls_back_to_stem(tmp_path):
    note = tmp_path / "empty-note.txt"
    note.write_text("\n\n")
    title, body = textitems.parse_note(note)
    assert title == "empty-note"


def test_parse_links_skips_comments_and_junk(tmp_path):
    links = tmp_path / "saved.links"
    links.write_text(
        "https://example.com/a  first one\n"
        "# a comment line\n"
        "\n"
        "not a url at all\n"
        "https://example.com/b\n"
    )
    assert textitems.parse_links(links) == [
        ("https://example.com/a", "first one"),
        ("https://example.com/b", ""),
    ]


def test_link_id_is_deterministic_and_distinct():
    assert textitems.link_id("https://a") == textitems.link_id("https://a")
    assert textitems.link_id("https://a") != textitems.link_id("https://b")


def test_fetch_page_survives_bad_urls():
    assert textitems.fetch_page("http://256.256.256.256/nope") == (None, "")
