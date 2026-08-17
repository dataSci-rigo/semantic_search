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
