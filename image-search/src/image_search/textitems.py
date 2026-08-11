"""Non-image "interesting stuff": notes and links.

Notes are .md/.txt files; links come from .links files (one URL per line,
optionally followed by a comment). Both become rows in the items table plus
text_fts entries and, when the folder has a text_embed model, a text vector —
so they rank alongside memes in the same hybrid search.
"""

from __future__ import annotations

import hashlib
import re
import sqlite3
import time
import urllib.request
from html.parser import HTMLParser
from pathlib import Path

NOTE_EXTENSIONS = {".md", ".txt"}
LINKS_EXTENSION = ".links"

FETCH_TIMEOUT = 10
FETCH_BODY_CAP = 5000
_URL_RE = re.compile(r"^https?://\S+$")


def note_id(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def link_id(url: str) -> str:
    return hashlib.sha256(f"link:{url}".encode()).hexdigest()


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


def fetch_page(url: str) -> tuple[str | None, str]:
    """Best-effort (title, text) for a URL. Any failure — offline laptop,
    dead link, non-HTML — degrades to (None, ""), never raises. Tests
    monkeypatch this."""
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "image-search/0.1"})
        with urllib.request.urlopen(req, timeout=FETCH_TIMEOUT) as resp:
            content_type = resp.headers.get("Content-Type", "")
            if "html" not in content_type and "text" not in content_type:
                return None, ""
            raw = resp.read(1 << 20).decode("utf-8", errors="replace")
        parser = _TextExtractor()
        parser.feed(raw)
        title = " ".join("".join(parser.title_parts).split()) or None
        text = " ".join(parser.text_parts)[:FETCH_BODY_CAP]
        return title, text
    except Exception:  # noqa: BLE001 - ingest must survive any bad URL
        return None, ""


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
) -> None:
    """Write one note/link item plus its FTS row and (if an embedder is
    given) its text vector. Caller owns the transaction, like ingest."""
    from image_search.store import vectors as vectors_store

    conn.execute(
        """
        INSERT INTO items (id, kind, folder, src_path, title, url, body, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(id) DO UPDATE SET
          kind=excluded.kind, folder=excluded.folder, src_path=excluded.src_path,
          title=excluded.title, url=excluded.url, body=excluded.body
        """,
        (item_id, kind, folder, src_path, title, url, body, time.time()),
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
