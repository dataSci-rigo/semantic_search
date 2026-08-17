"""Non-image "interesting stuff": notes and links.

Notes are .md/.txt files; links come from .links files (one URL per line,
optionally followed by a comment). Both become rows in the items table plus
text_fts entries and, when the folder has a text_embed model, a text vector —
so they rank alongside memes in the same hybrid search.
"""

from __future__ import annotations

import hashlib
import ipaddress
import re
import sqlite3
import time
import urllib.error
import urllib.parse
import urllib.request
from html.parser import HTMLParser
from pathlib import Path

NOTE_EXTENSIONS = {".md", ".txt"}
LINKS_EXTENSION = ".links"
PDF_EXTENSION = ".pdf"

FETCH_TIMEOUT = 10
FETCH_BODY_CAP = 5000
FETCH_SIZE_CAP = 1 << 20  # 1 MB — enough for any article's HTML
FETCH_USER_AGENT = "image-search/0.1 (personal bookmark indexer)"
# Below this much extracted text a page is "thin" — a shell, a redirect stub,
# or a JS-only app: keep the link, don't pretend we indexed its content.
THIN_TEXT_CHARS = 200
_URL_RE = re.compile(r"^https?://\S+$")

# Item status after fetching (stored on items.status; search shows only "ok").
STATUS_OK = "ok"
STATUS_DEAD = "dead"  # 404, timeout, DNS failure — nothing to index
STATUS_BLOCKED = "blocked"  # login/paywall/captcha wall
STATUS_THIN = "thin"  # reachable, but no real text
STATUS_SKIPPED = "skipped"  # rejected before fetching (URL shape)


def note_id(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def link_id(url: str) -> str:
    return hashlib.sha256(f"link:{url}".encode()).hexdigest()


# Params that identify a campaign or referrer, never the content itself.
TRACKING_PARAMS = {
    "fbclid", "gclid", "dclid", "msclkid", "mc_eid", "mc_cid",
    "igshid", "igsh", "ref", "ref_src", "ref_url", "si", "spm",
    "_ga", "_gl", "yclid", "twclid", "trk", "trkCampaign",
}
TRACKING_PREFIXES = ("utm_", "pk_", "piwik_", "matomo_", "hsa_", "vero_")

DEFAULT_PORTS = {"http": "80", "https": "443"}

# URL shapes that can't yield indexable content — rejected before any request.
_SEARCH_PATHS = ("/search", "/results")
_AUTH_MARKERS = ("/login", "/signin", "/sign-in", "/auth", "/oauth", "/logout", "/register")
_AUTH_HOSTS = ("accounts.google.com", "login.microsoftonline.com", "login.live.com")


def normalize_url(url: str) -> str:
    """Canonical form for de-duplication: the same page saved in two browsers,
    or once with a campaign tag, collapses to one string. Never used as the
    fetch target — the original URL is what gets requested."""
    parts = urllib.parse.urlsplit(url.strip())
    scheme = parts.scheme.lower()
    host = (parts.hostname or "").lower()
    if host.startswith("www."):
        host = host[4:]

    netloc = host
    if parts.port and str(parts.port) != DEFAULT_PORTS.get(scheme):
        netloc = f"{host}:{parts.port}"

    kept = [
        (k, v)
        for k, v in urllib.parse.parse_qsl(parts.query, keep_blank_values=True)
        if k.lower() not in TRACKING_PARAMS
        and not k.lower().startswith(TRACKING_PREFIXES)
    ]
    query = urllib.parse.urlencode(sorted(kept))

    path = parts.path.rstrip("/") or "/"
    # Fragments are client-side anchors, except in hashbang SPA routes where
    # they select the actual document.
    fragment = parts.fragment if parts.fragment.startswith("!") else ""
    return urllib.parse.urlunsplit((scheme, netloc, path, query, fragment))


def is_fetchable(url: str) -> tuple[bool, str]:
    """(fetchable, reason). Rejects URL shapes that can never yield content —
    checked before any network request is made."""
    try:
        parts = urllib.parse.urlsplit(url)
    except ValueError:
        return False, "unparseable url"
    if parts.scheme not in ("http", "https"):
        return False, f"unsupported scheme {parts.scheme!r}"

    host = (parts.hostname or "").lower()
    if not host:
        return False, "no host"
    if host in ("localhost", "127.0.0.1", "::1") or host.endswith(".local"):
        return False, "local address"
    try:
        if ipaddress.ip_address(host).is_private:
            return False, "private address"
    except ValueError:
        pass  # a hostname, not a literal IP

    path = parts.path.lower()
    if host in _AUTH_HOSTS or any(m in path for m in _AUTH_MARKERS):
        return False, "auth endpoint"
    if any(path.rstrip("/").endswith(p) for p in _SEARCH_PATHS) and parts.query:
        return False, "search results page"
    return True, ""


def parse_note(path: Path) -> tuple[str, str]:
    """(title, body): title is the first markdown heading, else the first
    non-empty line."""
    body = path.read_text(errors="replace")
    title = ""
    for line in body.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if title == "":
            title = stripped.lstrip("#").strip()
        if stripped.startswith("#"):
            title = stripped.lstrip("#").strip()
            break
        break
    return title or path.stem, body


def parse_links(path: Path) -> list[tuple[str, str]]:
    """[(url, comment)] from a .links file: one URL per line, anything after
    whitespace is a comment; blank lines and #-comment lines are skipped.
    Non-URL lines are ignored rather than fatal (hand-edited files)."""
    out: list[tuple[str, str]] = []
    for line in path.read_text(errors="replace").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split(None, 1)
        url = parts[0]
        if not _URL_RE.match(url):
            continue
        comment = parts[1].strip() if len(parts) > 1 else ""
        out.append((url, comment))
    return out


class _TextExtractor(HTMLParser):
    _SKIP = {"script", "style", "noscript", "template"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.title_parts: list[str] = []
        self.text_parts: list[str] = []
        self._in_title = False
        self._skip_depth = 0

    def handle_starttag(self, tag, attrs):
        if tag == "title":
            self._in_title = True
        if tag in self._SKIP:
            self._skip_depth += 1

    def handle_endtag(self, tag):
        if tag == "title":
            self._in_title = False
        if tag in self._SKIP and self._skip_depth:
            self._skip_depth -= 1

    def handle_data(self, data):
        if self._skip_depth:
            return
        if self._in_title:
            self.title_parts.append(data)
        elif data.strip():
            self.text_parts.append(data.strip())


# Text that means "you are not allowed to read this page" rather than content.
_WALL_MARKERS = (
    "sign in to continue", "log in to continue", "please enable javascript",
    "subscribe to read", "subscribers only", "create a free account to",
    "verify you are human", "checking your browser", "enable cookies",
    "access denied", "403 forbidden",
)

# Per-host politeness: never hit the same host twice inside this window.
HOST_DELAY_SECONDS = 1.0
_last_hit: dict[str, float] = {}


def _wait_for_host(host: str) -> None:
    now = time.monotonic()
    earliest = _last_hit.get(host, 0.0) + HOST_DELAY_SECONDS
    if now < earliest:
        time.sleep(earliest - now)
    _last_hit[host] = time.monotonic()


def fetch_page(url: str) -> tuple[str | None, str, str]:
    """Best-effort (title, text, status) for a URL. Never raises — a dead or
    hostile link is a status, not an exception. Status is one of the STATUS_*
    constants; see the classification table in docs/saved-ingestion.md.

    Tests monkeypatch this."""
    ok, _reason = is_fetchable(url)
    if not ok:
        return None, "", STATUS_SKIPPED

    host = (urllib.parse.urlsplit(url).hostname or "").lower()
    for attempt in (1, 2):
        try:
            _wait_for_host(host)
            req = urllib.request.Request(url, headers={"User-Agent": FETCH_USER_AGENT})
            with urllib.request.urlopen(req, timeout=FETCH_TIMEOUT) as resp:
                content_type = resp.headers.get("Content-Type", "").lower()
                if "pdf" in content_type:
                    # Handled by the PDF path, not the HTML extractor.
                    return None, "", STATUS_SKIPPED
                if "html" not in content_type and "text" not in content_type:
                    return None, "", STATUS_SKIPPED
                raw = resp.read(FETCH_SIZE_CAP).decode("utf-8", errors="replace")
            break
        except urllib.error.HTTPError as exc:
            # A refusal is not a dead link. Sites that block crawlers (403) or
            # sit behind auth (401) still hold the page you bookmarked, so keep
            # them findable by title instead of hiding them like a 404.
            if exc.code in (401, 402, 403, 429):
                return None, "", STATUS_BLOCKED
            if attempt == 1 and exc.code >= 500:
                time.sleep(1.0)  # server-side blip: worth one retry
                continue
            return None, "", STATUS_DEAD
        except Exception:  # noqa: BLE001 - classify, never propagate
            if attempt == 1:
                time.sleep(1.0)  # one retry covers most transient blips
                continue
            return None, "", STATUS_DEAD

    parser = _TextExtractor()
    try:
        parser.feed(raw)
    except Exception:  # noqa: BLE001 - malformed markup is not fatal
        pass
    title = " ".join("".join(parser.title_parts).split()) or None
    text = " ".join(parser.text_parts)

    lowered = text[:2000].lower()
    if any(marker in lowered for marker in _WALL_MARKERS):
        return title, "", STATUS_BLOCKED
    if len(text) < THIN_TEXT_CHARS:
        return title, "", STATUS_THIN
    return title, text[:FETCH_BODY_CAP], STATUS_OK


# Filenames that look like personal financial records. Excluded by default —
# indexing them puts tax data behind the same search box as memes.
# Word boundaries here use explicit lookarounds rather than \b: filenames are
# full of underscores ("W-2_2023.pdf"), and _ counts as a word character, so
# \b would not match there.
_NOT_ALNUM = r"(?<![a-z0-9])"
_NOT_ALNUM_AFTER = r"(?![a-z0-9])"
FINANCIAL_PATTERNS = (
    r"1099",
    rf"{_NOT_ALNUM}w-?2{_NOT_ALNUM_AFTER}",
    rf"{_NOT_ALNUM}1040{_NOT_ALNUM_AFTER}",
    rf"{_NOT_ALNUM}tax(es|-?return)?{_NOT_ALNUM_AFTER}",
    r"statement", r"invoice", r"receipt", r"payroll", r"paystub",
    rf"{_NOT_ALNUM}k-?1{_NOT_ALNUM_AFTER}",
)
_FINANCIAL_RE = re.compile("|".join(FINANCIAL_PATTERNS), re.IGNORECASE)

PDF_SAMPLE_PAGES = 5


# Phrases that only appear in financial/account paperwork. Checked against the
# *extracted text*, because filenames lie: an account-verification letter
# named "1_RODRIGO__LUNA.pdf" carries a home address and account number while
# matching none of the filename patterns above.
FINANCIAL_CONTENT_MARKERS = (
    "account verification", "routing number", "account number",
    "social security number", "taxable income", "year-end summary",
    "form 1099", "form w-2", "form 1040", "consolidated 1099",
    "designated bene plan", "brokerage account", "statement period",
)
_SSN_RE = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")


def looks_financial(path: Path) -> bool:
    """Filename check — fast, runs before the file is opened."""
    return bool(_FINANCIAL_RE.search(path.name))


def content_looks_financial(title: str, body: str) -> str | None:
    """Second gate, on extracted text. Returns the marker that matched, or
    None. Filename patterns miss anything not named for its contents, and
    'exclude financial documents' has to mean the documents, not the names."""
    haystack = f"{title}\n{body[:4000]}".lower()
    for marker in FINANCIAL_CONTENT_MARKERS:
        if marker in haystack:
            return marker
    if _SSN_RE.search(body[:4000]):
        return "social-security-number pattern"
    return None


def sample_page_numbers(total: int, sample: int = PDF_SAMPLE_PAGES) -> list[int]:
    """0-indexed pages to read: an even spread from the first page to the
    last. The first N pages of a long PDF are all front matter, so a spread
    describes the document far better than its opening."""
    if total <= sample:
        return list(range(total))
    step = (total - 1) / (sample - 1)
    return [round(i * step) for i in range(sample)]


def parse_pdf(path: Path, ocr_processor=None) -> tuple[str, str]:
    """(title, text) from a sample of the PDF's pages. Pages with no text
    layer (scans) fall back to OCR when a processor is supplied."""
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    pages = reader.pages
    chunks: list[str] = []
    needs_ocr: list[int] = []

    for index in sample_page_numbers(len(pages)):
        try:
            text = (pages[index].extract_text() or "").strip()
        except Exception:  # noqa: BLE001 - one broken page is not fatal
            text = ""
        if text:
            chunks.append(text)
        else:
            needs_ocr.append(index)

    if needs_ocr and ocr_processor is not None:
        chunks.extend(_ocr_pdf_pages(path, needs_ocr, ocr_processor))

    meta_title = ""
    try:
        meta_title = (reader.metadata.title or "").strip() if reader.metadata else ""
    except Exception:  # noqa: BLE001 - malformed metadata is common
        meta_title = ""

    body = "\n".join(chunks).strip()
    title = meta_title or path.stem
    return title, body[:FETCH_BODY_CAP * 4]


def _ocr_pdf_pages(path: Path, page_indexes: list[int], ocr_processor) -> list[str]:
    """Render scanned pages to images and OCR them. Requires pypdfium2 (or
    any renderer); absent that, scanned pages are simply skipped."""
    try:
        import pypdfium2
    except ImportError:
        return []

    import tempfile

    out: list[str] = []
    try:
        doc = pypdfium2.PdfDocument(str(path))
    except Exception:  # noqa: BLE001
        return []
    with tempfile.TemporaryDirectory() as tmp:
        for index in page_indexes:
            try:
                image = doc[index].render(scale=2).to_pil()
                page_path = Path(tmp) / f"page{index}.png"
                image.save(page_path)
                text = ocr_processor._call(page_path)  # noqa: SLF001 - bridge API
                if text.strip():
                    out.append(text.strip())
            except Exception:  # noqa: BLE001 - a page that won't render is skipped
                continue
    return out


def _searchable_text(title: str, url: str | None, body: str) -> str:
    return "\n".join(part for part in (title, url or "", body) if part).strip()


def insert_item(
    conn: sqlite3.Connection,
    *,
    item_id: str,
    kind: str,
    folder: str,
    src_path: str,
    title: str,
    url: str | None,
    body: str,
    text_embedder=None,
    status: str = STATUS_OK,
) -> None:
    """Write one note/link/pdf/book item plus its FTS row and (if an embedder
    is given) its text vector. Caller owns the transaction, like ingest."""
    from image_search.store import vectors as vectors_store

    conn.execute(
        """
        INSERT INTO items (id, kind, folder, src_path, title, url, body, status, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(id) DO UPDATE SET
          kind=excluded.kind, folder=excluded.folder, src_path=excluded.src_path,
          title=excluded.title, url=excluded.url, body=excluded.body,
          status=excluded.status
        """,
        (item_id, kind, folder, src_path, title, url, body, status, time.time()),
    )
    text = _searchable_text(title, url, body)
    if text:
        conn.execute(
            "INSERT INTO text_fts (image_id, text) VALUES (?, ?)", (item_id, text)
        )
        if text_embedder is not None:
            vectors_store.insert_vector(
                conn, "text", text_embedder.model_id, item_id, text_embedder.embed(text)
            )
