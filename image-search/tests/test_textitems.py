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


def test_fetch_page_rejects_unfetchable_urls_without_network():
    """URL-shape rejection happens before any request is made."""
    assert textitems.fetch_page("http://localhost:8080/admin") == (
        None, "", textitems.STATUS_SKIPPED,
    )
    assert textitems.fetch_page("file:///etc/passwd")[2] == textitems.STATUS_SKIPPED


def test_fetch_page_reports_dead_instead_of_raising(monkeypatch):
    def boom(*args, **kwargs):
        raise OSError("no route to host")

    monkeypatch.setattr(textitems.urllib.request, "urlopen", boom)
    monkeypatch.setattr(textitems.time, "sleep", lambda _s: None)
    assert textitems.fetch_page("https://example.com/gone") == (
        None, "", textitems.STATUS_DEAD,
    )


# ---- URL normalization & dedupe --------------------------------------------

def test_normalize_url_collapses_equivalent_forms():
    canonical = textitems.normalize_url("https://example.com/post")
    for variant in (
        "https://www.example.com/post",
        "https://EXAMPLE.com/post",
        "https://example.com/post/",
        "https://example.com:443/post",
        "https://example.com/post#section",
        "https://example.com/post?utm_source=twitter&utm_campaign=x",
        "https://example.com/post?fbclid=abc123",
    ):
        assert textitems.normalize_url(variant) == canonical, variant


def test_normalize_url_keeps_meaningful_differences():
    base = textitems.normalize_url("https://example.com/post")
    # Real query params, other paths, and hashbang routes are content.
    assert textitems.normalize_url("https://example.com/post?id=7") != base
    assert textitems.normalize_url("https://example.com/other") != base
    assert textitems.normalize_url("http://example.com/post") != base  # scheme differs
    assert "#!" in textitems.normalize_url("https://example.com/app#!/route")


def test_normalize_url_sorts_params_so_order_does_not_matter():
    assert textitems.normalize_url(
        "https://example.com/x?b=2&a=1"
    ) == textitems.normalize_url("https://example.com/x?a=1&b=2")


def test_is_fetchable_rejects_local_and_auth_urls():
    for url in (
        "http://localhost/x", "http://127.0.0.1/x", "http://192.168.1.5/admin",
        "https://accounts.google.com/signin", "https://site.com/login",
        "ftp://files.example.com/x", "https://site.com/search?q=cats",
    ):
        assert textitems.is_fetchable(url)[0] is False, url

    for url in ("https://example.com/article", "http://blog.example.org/2024/post"):
        assert textitems.is_fetchable(url)[0] is True, url


def _http_error(code):
    import urllib.error

    def raiser(*args, **kwargs):
        raise urllib.error.HTTPError("https://x", code, "err", {}, None)

    return raiser


def test_bot_blocked_pages_are_blocked_not_dead(monkeypatch):
    """A 403 from a crawler-hostile site (Medium, Cloudflare) still means the
    page exists — keep it findable by title instead of hiding it."""
    monkeypatch.setattr(textitems.time, "sleep", lambda _s: None)
    for code in (401, 402, 403, 429):
        monkeypatch.setattr(textitems.urllib.request, "urlopen", _http_error(code))
        assert textitems.fetch_page("https://medium.com/p/x") == (
            None, "", textitems.STATUS_BLOCKED,
        ), code


def test_missing_pages_are_dead(monkeypatch):
    monkeypatch.setattr(textitems.time, "sleep", lambda _s: None)
    for code in (404, 410):
        monkeypatch.setattr(textitems.urllib.request, "urlopen", _http_error(code))
        assert textitems.fetch_page("https://example.com/gone")[2] == textitems.STATUS_DEAD


def test_server_errors_are_retried_once(monkeypatch):
    calls = []

    def flaky(*args, **kwargs):
        calls.append(1)
        raise textitems.urllib.error.HTTPError("https://x", 503, "busy", {}, None)

    monkeypatch.setattr(textitems.time, "sleep", lambda _s: None)
    monkeypatch.setattr(textitems.urllib.request, "urlopen", flaky)
    assert textitems.fetch_page("https://example.com/x")[2] == textitems.STATUS_DEAD
    assert len(calls) == 2  # one retry, then give up


class _FakeResponse:
    def __init__(self, body, content_type="text/html"):
        self._body = body.encode()
        self.headers = {"Content-Type": content_type}

    def read(self, _n=None):
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def test_paywalled_page_is_blocked(monkeypatch):
    html = "<html><title>Big Story</title><body>" + "Subscribe to read this article. " * 20 + "</body></html>"
    monkeypatch.setattr(
        textitems.urllib.request, "urlopen", lambda *a, **k: _FakeResponse(html)
    )
    monkeypatch.setattr(textitems.time, "sleep", lambda _s: None)
    title, text, status = textitems.fetch_page("https://news.example.com/story")
    assert status == textitems.STATUS_BLOCKED
    assert title == "Big Story"  # still findable by what you saved


def test_javascript_shell_is_thin(monkeypatch):
    html = "<html><title>Dashboard</title><body><div id=root></div></body></html>"
    monkeypatch.setattr(
        textitems.urllib.request, "urlopen", lambda *a, **k: _FakeResponse(html)
    )
    monkeypatch.setattr(textitems.time, "sleep", lambda _s: None)
    assert textitems.fetch_page("https://app.example.com/")[2] == textitems.STATUS_THIN


def test_real_article_is_ok(monkeypatch):
    body = "Interest rates rose sharply this quarter. " * 30
    html = f"<html><title>Rates</title><body><p>{body}</p></body></html>"
    monkeypatch.setattr(
        textitems.urllib.request, "urlopen", lambda *a, **k: _FakeResponse(html)
    )
    monkeypatch.setattr(textitems.time, "sleep", lambda _s: None)
    title, text, status = textitems.fetch_page("https://example.com/rates")
    assert status == textitems.STATUS_OK
    assert title == "Rates" and "Interest rates rose" in text


def test_json_api_is_skipped_not_indexed(monkeypatch):
    monkeypatch.setattr(
        textitems.urllib.request,
        "urlopen",
        lambda *a, **k: _FakeResponse('{"a": 1}', "application/json"),
    )
    monkeypatch.setattr(textitems.time, "sleep", lambda _s: None)
    assert textitems.fetch_page("https://api.example.com/v2/data")[2] == (
        textitems.STATUS_SKIPPED
    )


# ---- chunked embeddings ----------------------------------------------------


def test_chunk_text_small_returns_whole():
    from image_search.textitems import chunk_text

    assert chunk_text("short note") == ["short note"]
    assert chunk_text("   ") == []


def test_chunk_text_splits_on_paragraphs_with_overlap():
    from image_search.textitems import chunk_text

    paras = [
        f"Paragraph {i}. " + " ".join(f"w{i}x{j}" for j in range(60))
        for i in range(12)
    ]
    text = "\n\n".join(paras)
    chunks = chunk_text(text, target=1000, overlap=150)
    assert len(chunks) > 1
    assert all(len(c) <= 1000 for c in chunks)
    # Nothing lost: every paragraph's head appears in some chunk.
    for para in paras:
        head = para[:40]
        assert any(head in c for c in chunks)
    # Consecutive chunks overlap (tokens are unique, so this is meaningful).
    assert all(chunks[i][-30:] in chunks[i + 1] for i in range(len(chunks) - 1))


def test_long_note_gets_chunk_vectors_and_one_search_hit(tmp_path):
    import sqlite3

    from image_search.config import load_config
    from image_search.registry import Registry
    from image_search.ingest import ingest_folder
    from image_search.search import search_text
    from image_search.store.db import connect, migrate

    folder = tmp_path / "notes"
    folder.mkdir()
    body = "\n\n".join(
        f"Section {i}. " + " ".join(f"tok{i}x{j}" for j in range(80))
        for i in range(10)
    )
    (folder / "big.md").write_text(f"# Big Note\n\n{body}")

    config_path = tmp_path / "folders.yaml"
    config_path.write_text(
        f'folders:\n  "{folder}":\n    text_embed: fake-embed\n'
    )
    config = load_config(config_path)
    registry = Registry(config)

    class FakeEmbed:
        kind, model_id = "text_embed", "fake-embed"
        def load(self): pass
        def embed(self, text): return [float(len(text) % 7), 1.0, 0.0]
        def process(self, img): return []

    registry._instances[("text_embed", "fake-embed")] = FakeEmbed()
    conn = connect(tmp_path / "t.db")
    migrate(conn)
    ingest_folder(conn, config, registry, str(folder))

    chunk_rows = conn.execute(
        "SELECT COUNT(*) AS n FROM vec_map WHERE image_id LIKE '%#c%'"
    ).fetchone()["n"]
    assert chunk_rows > 1  # long note stored as multiple chunk vectors

    hits = search_text(conn, config, registry, str(folder), "tok5x40", k=10)
    ids = [h.image_id for h in hits]
    assert len(ids) == len(set(ids)) == 1  # chunks resolve to ONE parent item
    assert "#c" not in ids[0]
    assert hits[0].title == "Big Note"

    # Deleting the file purges chunk vectors too.
    (folder / "big.md").unlink()
    ingest_folder(conn, config, registry, str(folder))
    left = conn.execute("SELECT COUNT(*) AS n FROM vec_map").fetchone()["n"]
    assert left == 0


# ---- tier-0 video metadata items -------------------------------------------


def _video_setup(tmp_path, video_bytes=b"not really a video"):
    from image_search.config import load_config
    from image_search.registry import Registry
    from image_search.store.db import connect, migrate

    folder = tmp_path / "vids"
    folder.mkdir()
    video = folder / "VID_20190704_beach.mp4"
    video.write_bytes(video_bytes)
    config_path = tmp_path / "folders.yaml"
    config_path.write_text(f'folders:\n  "{folder}":\n    ocr: fake-ocr\n')
    config = load_config(config_path)
    conn = connect(tmp_path / "t.db")
    migrate(conn)
    return folder, video, config, Registry(config), conn


def test_video_indexed_with_sidecar_metadata(tmp_path):
    import json

    from image_search.ingest import ingest_folder
    from image_search.search import search_text

    folder, video, config, registry, conn = _video_setup(tmp_path)
    (folder / "VID_20190704_beach.mp4.json").write_text(json.dumps({
        "title": "Fourth of July fireworks",
        "description": "fireworks over the lake with the kids",
        "photoTakenTime": {"formatted": "Jul 4, 2019, 9:12:33 PM UTC"},
        "geoData": {"latitude": 34.05, "longitude": -118.24},
    }))

    stats = ingest_folder(conn, config, registry, str(folder))
    assert stats["indexed"] >= 1
    row = conn.execute("SELECT * FROM items WHERE kind='video'").fetchone()
    assert row["title"] == "Fourth of July fireworks"
    assert "fireworks over the lake" in row["body"]
    assert "34.0500" in row["body"]

    hits = search_text(conn, config, registry, str(folder), "fireworks")
    assert any(h.kind == "video" for h in hits)


def test_unreadable_video_still_indexes_by_name(tmp_path):
    from image_search.ingest import ingest_folder

    folder, video, config, registry, conn = _video_setup(tmp_path)
    stats = ingest_folder(conn, config, registry, str(folder))
    assert stats["failed"] == 0
    row = conn.execute("SELECT * FROM items WHERE kind='video'").fetchone()
    assert row["title"] == "VID_20190704_beach"
    assert "VID_20190704_beach.mp4" in row["body"]


def test_real_video_gets_duration_and_resolution(tmp_path):
    import shutil
    import subprocess

    import pytest

    if not shutil.which("ffmpeg"):
        pytest.skip("ffmpeg not installed")
    from image_search.ingest import ingest_folder

    folder, video, config, registry, conn = _video_setup(tmp_path)
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i",
         "testsrc=duration=2:size=320x240:rate=5", str(video)],
        check=True,
    )
    ingest_folder(conn, config, registry, str(folder))
    row = conn.execute("SELECT body FROM items WHERE kind='video'").fetchone()
    assert "duration 0m02s" in row["body"]
    assert "320x240" in row["body"]


# ---- HTML notes (Keep exports, saved pages) and bookmark files -------------


def test_parse_html_note_cleans_and_titles(tmp_path):
    from image_search.textitems import parse_html_note

    page = tmp_path / "API Plant life.html"
    page.write_text(
        "<html><head><title>API   Plant life</title>"
        "<script>var junk = 'never index me';</script>"
        "<style>.x{color:red}</style></head>"
        "<body><p>Water the   monstera weekly.</p>\n\n\n\n"
        "<p>Perenual API key is in the env.</p></body></html>"
    )
    title, body = parse_html_note(page)
    assert title == "API Plant life"
    assert "monstera weekly" in body and "Perenual API" in body
    assert "junk" not in body and "color:red" not in body
    assert "\n\n\n" not in body  # blank runs collapsed


def test_bookmarks_file_becomes_link_items_with_fetched_text(tmp_path, monkeypatch):
    from image_search import textitems
    from image_search.config import load_config
    from image_search.ingest import ingest_folder
    from image_search.registry import Registry
    from image_search.search import search_text
    from image_search.store.db import connect, migrate

    # Pages are fetched for their text body (never real network in tests).
    fetched: list[str] = []

    def fake_fetch(url):
        fetched.append(url)
        return None, f"page text body for {url}", textitems.STATUS_OK

    monkeypatch.setattr(textitems, "fetch_page", fake_fetch)

    folder = tmp_path / "vault"
    folder.mkdir()
    (folder / "Bookmarks.html").write_text(
        '<DL><DT><A HREF="https://sqlite.org/vec.html" ADD_DATE="1">sqlite-vec docs</A>'
        '<DT><A HREF="https://example.com/rl">Omaha RL notes</A>'
        '<DT><A HREF="javascript:void(0)">junk pseudo-link</A></DL>'
    )
    config_path = tmp_path / "folders.yaml"
    config_path.write_text(f'folders:\n  "{folder}":\n    ocr: fake-ocr\n')
    config = load_config(config_path)
    registry = Registry(config)
    conn = connect(tmp_path / "t.db")
    migrate(conn)

    ingest_folder(conn, config, registry, str(folder))
    rows = conn.execute("SELECT * FROM items WHERE kind='link' ORDER BY title").fetchall()
    assert [r["title"] for r in rows] == ["Omaha RL notes", "sqlite-vec docs"]
    assert rows[1]["url"] == "https://sqlite.org/vec.html"
    # The javascript: pseudo-link was never fetched; real URLs were, and the
    # page text landed in the searchable body.
    assert sorted(fetched) == ["https://example.com/rl", "https://sqlite.org/vec.html"]
    assert "page text body for https://sqlite.org/vec.html" in rows[1]["body"]

    hits = search_text(conn, config, registry, str(folder), "sqlite-vec")
    assert any(h.url == "https://sqlite.org/vec.html" for h in hits)

    # Removing a bookmark from the export purges its item on rescan.
    (folder / "Bookmarks.html").write_text(
        '<DL><DT><A HREF="https://sqlite.org/vec.html">sqlite-vec docs</A></DL>'
    )
    ingest_folder(conn, config, registry, str(folder))
    left = conn.execute("SELECT title FROM items WHERE kind='link'").fetchall()
    assert [r["title"] for r in left] == ["sqlite-vec docs"]


# ---- Word documents and Google Tasks exports -------------------------------


def test_parse_docx_extracts_paragraph_text(tmp_path):
    import zipfile

    from image_search.textitems import parse_doc

    docx = tmp_path / "cover letter.docx"
    xml = (
        '<?xml version="1.0"?><w:document xmlns:w="ns"><w:body>'
        "<w:p><w:r><w:t>To Whom It May Concern</w:t></w:r></w:p>"
        "<w:p><w:r><w:t>I am writing about the R&amp;D role.</w:t></w:r></w:p>"
        "</w:body></w:document>"
    )
    with zipfile.ZipFile(docx, "w") as zf:
        zf.writestr("word/document.xml", xml)
    title, body = parse_doc(docx)
    assert title == "To Whom It May Concern"
    assert "R&D role" in body
    assert "<w:" not in body


def test_parse_doc_legacy_via_converter(tmp_path):
    import shutil
    import subprocess

    import pytest

    if not shutil.which("soffice"):
        pytest.skip("LibreOffice not installed")
    from image_search.textitems import parse_doc

    src = tmp_path / "notes.txt"
    src.write_text("Thesis meeting notes\nDiscussed adaptive filters.")
    subprocess.run(
        ["soffice", "--headless", f"-env:UserInstallation=file://{tmp_path}/lo",
         "--convert-to", "doc", "--outdir", str(tmp_path), str(src)],
        capture_output=True, timeout=120,
    )
    if not (tmp_path / "notes.doc").exists():
        # soffice present but non-functional (e.g. libreoffice-writer not
        # installed) — parse_doc's empty-body fallback covers production.
        pytest.skip("soffice cannot convert on this machine")
    title, body = parse_doc(tmp_path / "notes.doc")
    assert "adaptive filters" in body.lower()


def test_tasks_json_parses_lists_and_status(tmp_path):
    import json

    from image_search.textitems import parse_note

    tasks = tmp_path / "Tasks.json"
    tasks.write_text(json.dumps({
        "kind": "tasks#taskLists",
        "items": [{
            "title": "House",
            "items": [
                {"title": "fix the gate", "status": "needsAction",
                 "due": "2026-10-01T00:00:00.000Z"},
                {"title": "buy paint", "status": "completed",
                 "notes": "behr  ultra,  eggshell"},
            ],
        }],
    }))
    title, body = parse_note(tasks)
    assert title == "Google Tasks"
    assert "# House" in body
    assert "- [ ] fix the gate (due 2026-10-01)" in body
    assert "- [x] buy paint" in body
    assert "behr ultra, eggshell" in body
