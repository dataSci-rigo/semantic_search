#!/usr/bin/env python3
"""Turn photos of your bookshelves into searchable book records.

  python scripts/import_books.py shelf1.jpg shelf2.jpg -o ~/Saved/books.links

Two stages, deliberately separated:

  1. Claude vision reads the spines and returns structured titles/authors.
     Vision is good at this but not perfect — stylized spines, glare, and
     vertical text all cost accuracy, so every extraction carries a
     confidence and anything low-confidence goes to a review file instead of
     into the index.
  2. Open Library supplies the description and subjects. The summary is
     *looked up*, never recalled by a model — an obscure title gets a real
     description or none at all, rather than a confident invention.

Output is a `.links` file of Open Library work URLs with the title, author,
and subjects as the comment, so the existing ingest path indexes it.

Needs ANTHROPIC_API_KEY (read from the environment, else from a .env file —
see --env-file). Open Library needs no credentials.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import re
import sys
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path

MODEL = "claude-opus-5"
OPENLIBRARY_SEARCH = "https://openlibrary.org/search.json"
USER_AGENT = "image-search/0.1 (personal library indexer)"
CONFIDENCE_FLOOR = 0.7

SPINE_SCHEMA = {
    "type": "object",
    "properties": {
        "books": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "title": {"type": "string", "description": "Title as printed"},
                    "author": {"type": "string", "description": "Author, '' if unreadable"},
                    "confidence": {
                        "type": "number",
                        "description": "0-1: how sure you are of this reading",
                    },
                },
                "required": ["title", "author", "confidence"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["books"],
    "additionalProperties": False,
}

PROMPT = """List every book you can read in this photo of a bookshelf.

For each: the title as printed, the author if legible (empty string if not),
and a confidence from 0 to 1 for how certain you are of the reading.

Be honest about uncertainty — a spine that is blurry, partly hidden, at a
steep angle, or in an unfamiliar script should get a low confidence rather
than a confident guess. Do not invent books that are not visible, and do not
complete a partly-readable title from memory: report what you can actually
see. Skip objects that are not books."""


@dataclass
class Book:
    title: str
    author: str
    confidence: float
    source: str


def _load_api_key(env_file: Path | None) -> str:
    if os.environ.get("ANTHROPIC_API_KEY"):
        return os.environ["ANTHROPIC_API_KEY"]
    candidates = [env_file] if env_file else [
        Path.home() / "code20/.env",
        Path.cwd() / ".env",
    ]
    for path in candidates:
        if path and path.is_file():
            try:
                from dotenv import dotenv_values
            except ImportError:
                break
            value = dotenv_values(path).get("ANTHROPIC_API_KEY")
            if value:
                return value
    raise SystemExit(
        "No ANTHROPIC_API_KEY found. Export it, or pass --env-file pointing at "
        "a .env that defines it."
    )


def read_spines(client, image_path: Path) -> list[Book]:
    """One vision call per photo. Structured outputs keep the response a
    parseable object rather than prose we'd have to scrape."""
    media = {
        ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
        ".png": "image/png", ".webp": "image/webp", ".gif": "image/gif",
    }.get(image_path.suffix.lower())
    if media is None:
        raise SystemExit(f"Unsupported image type: {image_path}")

    data = base64.standard_b64encode(image_path.read_bytes()).decode()
    response = client.messages.create(
        model=MODEL,
        max_tokens=8000,
        thinking={"type": "adaptive"},
        output_config={"format": {"type": "json_schema", "schema": SPINE_SCHEMA}},
        messages=[{
            "role": "user",
            "content": [
                {"type": "image",
                 "source": {"type": "base64", "media_type": media, "data": data}},
                {"type": "text", "text": PROMPT},
            ],
        }],
    )
    if response.stop_reason == "refusal":
        print(f"  {image_path.name}: request was declined, skipping", file=sys.stderr)
        return []

    text = next((b.text for b in response.content if b.type == "text"), "")
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        print(f"  {image_path.name}: unparseable response, skipping", file=sys.stderr)
        return []

    return [
        Book(
            title=(entry.get("title") or "").strip(),
            author=(entry.get("author") or "").strip(),
            confidence=float(entry.get("confidence") or 0),
            source=image_path.name,
        )
        for entry in payload.get("books", [])
        if (entry.get("title") or "").strip()
    ]


def _norm(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def deduplicate(books: list[Book]) -> list[Book]:
    """Overlapping photos of one shelf show the same spine twice."""
    best: dict[tuple[str, str], Book] = {}
    for book in books:
        key = (_norm(book.title), _norm(book.author))
        if key not in best or book.confidence > best[key].confidence:
            best[key] = book
    return sorted(best.values(), key=lambda b: (_norm(b.author), _norm(b.title)))


def _get_json(url: str) -> dict | None:
    try:
        req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(req, timeout=15) as resp:
            return json.loads(resp.read(1 << 20).decode("utf-8", errors="replace"))
    except Exception:  # noqa: BLE001 - a lookup miss is not fatal
        return None


def lookup_openlibrary(book: Book) -> dict | None:
    """Canonical metadata + description. Returns None when Open Library has
    no confident match — better an unindexed book than a wrong one."""
    params = {"title": book.title, "limit": "3", "fields": "key,title,author_name,first_publish_year,subject"}
    if book.author:
        params["author"] = book.author
    data = _get_json(f"{OPENLIBRARY_SEARCH}?{urllib.parse.urlencode(params)}")
    if not data or not data.get("docs"):
        return None

    doc = data["docs"][0]
    # Guard against the search returning something unrelated.
    if _norm(doc.get("title", "")).split() [:3] != _norm(book.title).split()[:3]:
        found, wanted = _norm(doc.get("title", "")), _norm(book.title)
        if found not in wanted and wanted not in found:
            return None

    work_key = doc.get("key", "")
    description = ""
    if work_key:
        work = _get_json(f"https://openlibrary.org{work_key}.json") or {}
        raw = work.get("description")
        description = raw.get("value", "") if isinstance(raw, dict) else (raw or "")

    return {
        "url": f"https://openlibrary.org{work_key}" if work_key else "",
        "title": doc.get("title") or book.title,
        "authors": ", ".join(doc.get("author_name") or []) or book.author,
        "year": doc.get("first_publish_year") or "",
        "subjects": (doc.get("subject") or [])[:12],
        "description": " ".join(description.split())[:1500],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("photos", nargs="+", help="Bookshelf photos")
    parser.add_argument("-o", "--output", required=True, help=".links file to write")
    parser.add_argument("--review", help="Where to write low-confidence rows "
                                         "(default: alongside the output)")
    parser.add_argument("--env-file", type=Path, help=".env holding ANTHROPIC_API_KEY")
    parser.add_argument("--min-confidence", type=float, default=CONFIDENCE_FLOOR)
    args = parser.parse_args()

    import anthropic

    client = anthropic.Anthropic(api_key=_load_api_key(args.env_file))

    found: list[Book] = []
    for photo in args.photos:
        path = Path(photo)
        if not path.is_file():
            print(f"  {photo}: not found, skipping", file=sys.stderr)
            continue
        print(f"Reading {path.name} …", flush=True)
        spines = read_spines(client, path)
        print(f"  {len(spines)} titles")
        found.extend(spines)

    books = deduplicate(found)
    confident = [b for b in books if b.confidence >= args.min_confidence]
    unsure = [b for b in books if b.confidence < args.min_confidence]
    print(f"\n{len(found)} readings -> {len(books)} unique "
          f"({len(confident)} confident, {len(unsure)} need review)")

    lines = ["# Generated by scripts/import_books.py — descriptions from Open Library.\n"]
    matched = unmatched = 0
    for book in confident:
        meta = lookup_openlibrary(book)
        time.sleep(0.5)  # be polite to a free service
        if not meta or not meta["url"]:
            unsure.append(book)
            unmatched += 1
            continue
        matched += 1
        comment = f"{meta['title']} — {meta['authors']}"
        if meta["year"]:
            comment += f" ({meta['year']})"
        if meta["subjects"]:
            comment += " [" + "; ".join(meta["subjects"]) + "]"
        if meta["description"]:
            comment += " " + meta["description"]
        lines.append(f"{meta['url']} {' '.join(comment.split())}\n")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("".join(lines), encoding="utf-8")

    review_path = Path(args.review) if args.review else out_path.with_suffix(".review.txt")
    if unsure:
        review_path.write_text(
            "# Not indexed: low confidence or no Open Library match.\n"
            "# Check these against the shelf, then add them to the .links file.\n"
            + "".join(
                f"{b.confidence:.2f}\t{b.title}\t{b.author}\t({b.source})\n"
                for b in sorted(unsure, key=lambda b: b.confidence)
            ),
            encoding="utf-8",
        )

    print(f"matched {matched} in Open Library, {unmatched} had no match")
    print(f"\nWrote {matched} books to {out_path}")
    if unsure:
        print(f"{len(unsure)} need a look: {review_path}")


if __name__ == "__main__":
    main()
