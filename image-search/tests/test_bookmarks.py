"""Importer tests. No browser, no network — profiles are built on disk."""

import json
import sqlite3
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import import_bookmarks as ib  # noqa: E402


# ---- Firefox ---------------------------------------------------------------

def _make_firefox_profile(tmp_path: Path, rows: list[tuple[str, str, int]]) -> Path:
    """rows: (title, url, parent_folder_id). Folder 2 = 'Bookmarks Toolbar'."""
    profile = tmp_path / "abc.default-release"
    profile.mkdir(parents=True)
    conn = sqlite3.connect(profile / "places.sqlite")
    conn.executescript(
        """
        CREATE TABLE moz_places (id INTEGER PRIMARY KEY, url TEXT);
        CREATE TABLE moz_bookmarks (
          id INTEGER PRIMARY KEY, type INTEGER, fk INTEGER,
          parent INTEGER, title TEXT
        );
        INSERT INTO moz_bookmarks (id, type, fk, parent, title)
          VALUES (1, 2, NULL, 0, 'root'), (2, 2, NULL, 1, 'Bookmarks Toolbar'),
                 (3, 2, NULL, 2, 'Reading');
        """
    )
    for i, (title, url, parent) in enumerate(rows, start=100):
        conn.execute("INSERT INTO moz_places (id, url) VALUES (?, ?)", (i, url))
        conn.execute(
            "INSERT INTO moz_bookmarks (id, type, fk, parent, title) VALUES (?, 1, ?, ?, ?)",
            (i + 1000, i, parent, title),
        )
    conn.commit()
    conn.close()
    return profile


def test_reads_firefox_bookmarks_with_folder_path(tmp_path):
    profile = _make_firefox_profile(tmp_path, [
        ("Example", "https://example.com/a", 3),
        ("Toolbar link", "https://example.org/b", 2),
    ])
    marks = ib.read_firefox(profile)

    by_url = {m.url: m for m in marks}
    assert set(by_url) == {"https://example.com/a", "https://example.org/b"}
    assert by_url["https://example.com/a"].folder == "Bookmarks Toolbar/Reading"
    assert by_url["https://example.org/b"].folder == "Bookmarks Toolbar"
    assert all(m.browser == "firefox" for m in marks)


def test_firefox_read_is_lock_free(tmp_path):
    """places.sqlite opens read-only even with a writer holding the file —
    which is the normal case, since Firefox is usually running."""
    profile = _make_firefox_profile(tmp_path, [("A", "https://example.com/a", 2)])
    holder = sqlite3.connect(profile / "places.sqlite")
    holder.execute("BEGIN EXCLUSIVE")
    try:
        assert len(ib.read_firefox(profile)) == 1
    finally:
        holder.rollback()
        holder.close()


def test_missing_or_corrupt_profile_returns_empty(tmp_path):
    assert ib.read_firefox(tmp_path / "nope") == []
    broken = tmp_path / "broken"
    broken.mkdir()
    (broken / "places.sqlite").write_text("not a database")
    assert ib.read_firefox(broken) == []


# ---- Chromium --------------------------------------------------------------

def _make_chromium(home: Path, browser: str, tree: dict) -> None:
    path = home / ib.CHROMIUM_PATHS[browser]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(tree))


def test_reads_chrome_bookmarks_recursively(tmp_path):
    _make_chromium(tmp_path, "chrome", {
        "roots": {
            "bookmark_bar": {
                "name": "Bookmarks bar", "type": "folder",
                "children": [
                    {"type": "url", "name": "Top", "url": "https://example.com/top"},
                    {"type": "folder", "name": "Nested", "children": [
                        {"type": "url", "name": "Deep", "url": "https://example.com/deep"},
                        {"type": "url", "name": "JS", "url": "javascript:void(0)"},
                    ]},
                ],
            }
        }
    })
    marks = ib.read_chromium(tmp_path, "chrome")
    by_url = {m.url: m for m in marks}

    assert set(by_url) == {"https://example.com/top", "https://example.com/deep"}
    assert by_url["https://example.com/deep"].folder == "Bookmarks bar/Nested"


def test_missing_or_corrupt_chromium_file_returns_empty(tmp_path):
    assert ib.read_chromium(tmp_path, "chrome") == []
    path = tmp_path / ib.CHROMIUM_PATHS["edge"]
    path.parent.mkdir(parents=True)
    path.write_text("{ not json")
    assert ib.read_chromium(tmp_path, "edge") == []


# ---- dedupe ----------------------------------------------------------------

def _mark(url, title="", folder="", browser="chrome"):
    return ib.Bookmark(url=url, title=title, folder=folder, browser=browser)


def test_dedupes_the_same_page_across_browsers():
    merged, stats = ib.deduplicate([
        _mark("https://example.com/post", "Post", "News", "chrome"),
        _mark("https://www.example.com/post/", "The Full Post Title", "", "edge"),
        _mark("https://example.com/post?utm_source=x", "Post", "", "firefox"),
    ])

    assert len(merged) == 1
    entry = merged[0]
    assert entry.browsers == {"chrome", "edge", "firefox"}
    assert entry.title == "The Full Post Title"  # richest title wins
    assert entry.folders == {"News"}
    assert stats["unique"] == 1 and stats["duplicates merged"] == 2


def test_distinct_pages_are_kept_apart():
    merged, stats = ib.deduplicate([
        _mark("https://example.com/a"), _mark("https://example.com/b"),
        _mark("https://other.com/a"),
    ])
    assert stats["unique"] == 3 and len(merged) == 3


def test_unfetchable_urls_are_dropped_with_a_reason():
    merged, stats = ib.deduplicate([
        _mark("https://example.com/good"),
        _mark("http://localhost:3000/app"),
        _mark("https://accounts.google.com/signin"),
    ])
    assert stats["unique"] == 1
    assert stats["skipped: local address"] == 1
    assert stats["skipped: auth endpoint"] == 1
    assert merged[0].url == "https://example.com/good"


def test_merged_url_is_the_original_not_the_normalized_form():
    """Normalization is for comparison only — we must fetch a real URL."""
    merged, _ = ib.deduplicate([_mark("https://www.Example.com/Post?utm_source=x")])
    assert merged[0].url == "https://www.Example.com/Post?utm_source=x"


# ---- output ----------------------------------------------------------------

def test_written_links_file_round_trips_through_the_parser(tmp_path):
    from image_search.textitems import parse_links

    entries, _ = ib.deduplicate([
        _mark("https://example.com/a", "First", "News", "chrome"),
        _mark("https://example.com/b", "Second\nwith newline", "", "firefox"),
    ])
    out = tmp_path / "bookmarks.links"
    ib.write_links(entries, out)

    parsed = parse_links(out)
    assert [u for u, _ in parsed] == ["https://example.com/a", "https://example.com/b"]
    # Titles survive as comments, and no comment spans two lines.
    assert "First" in dict(parsed)["https://example.com/a"]
    assert "\n" not in dict(parsed)["https://example.com/b"]


def test_written_comment_records_provenance(tmp_path):
    entries, _ = ib.deduplicate([
        _mark("https://example.com/x", "Title", "Recipes", "chrome"),
        _mark("https://example.com/x", "Title", "Cooking", "firefox"),
    ])
    out = tmp_path / "b.links"
    ib.write_links(entries, out)
    line = out.read_text().strip().splitlines()[-1]

    assert "chrome,firefox" in line
    assert "Cooking" in line and "Recipes" in line


@pytest.mark.parametrize("browser", ["chrome", "edge", "brave"])
def test_all_chromium_browsers_have_a_known_path(browser):
    assert ib.CHROMIUM_PATHS[browser].endswith("Bookmarks")
