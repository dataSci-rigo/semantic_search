#!/usr/bin/env python3
"""Export browser bookmarks to a .links file the indexer already understands.

Reads Firefox, Chrome, and Edge, de-duplicates across all of them (the same
page is often saved in several browsers, sometimes with a campaign tag), and
writes one normalized `.links` file. Nothing is fetched here — that happens at
ingest time, so this step is fast and offline.

  python scripts/import_bookmarks.py ~/Saved/bookmarks.links
  python scripts/import_bookmarks.py out.links --browser firefox --dry-run

Firefox's places.sqlite is opened read-only via SQLite's `immutable=1` URI, so
it works while Firefox is running and cannot modify the profile.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from image_search.textitems import is_fetchable, normalize_url  # noqa: E402

WINDOWS_USERS = Path("/mnt/c/Users")


@dataclass
class Bookmark:
    url: str
    title: str
    folder: str
    browser: str


@dataclass
class Merged:
    url: str  # the original URL of the first occurrence — what gets fetched
    title: str
    browsers: set[str] = field(default_factory=set)
    folders: set[str] = field(default_factory=set)


def _windows_home() -> Path | None:
    """The Windows user profile, when running under WSL."""
    if not WINDOWS_USERS.is_dir():
        return None
    for entry in WINDOWS_USERS.iterdir():
        if entry.is_dir() and entry.name not in (
            "Public", "Default", "Default User", "All Users"
        ):
            candidate = entry / "AppData"
            if candidate.is_dir():
                return entry
    return None


# ---- Firefox ---------------------------------------------------------------

def firefox_profiles(home: Path) -> list[Path]:
    root = home / "AppData/Roaming/Mozilla/Firefox/Profiles"
    if not root.is_dir():
        return []
    return [p for p in sorted(root.iterdir()) if (p / "places.sqlite").is_file()]


def read_firefox(profile: Path) -> list[Bookmark]:
    """Bookmarks from one profile. Opened immutable so a running Firefox
    neither blocks us nor risks the profile."""
    db = profile / "places.sqlite"
    try:
        conn = sqlite3.connect(f"file:{db}?immutable=1", uri=True)
    except sqlite3.Error:
        return []
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            """
            SELECT b.title AS title, p.url AS url, b.parent AS parent
            FROM moz_bookmarks b JOIN moz_places p ON p.id = b.fk
            WHERE b.type = 1 AND p.url LIKE 'http%'
            """
        ).fetchall()
        folders = {
            r["id"]: (r["title"], r["parent"])
            for r in conn.execute(
                "SELECT id, title, parent FROM moz_bookmarks WHERE type = 2"
            )
        }
    except sqlite3.Error:
        return []
    finally:
        conn.close()

    def folder_path(parent_id: int) -> str:
        names, seen = [], set()
        while parent_id in folders and parent_id not in seen:
            seen.add(parent_id)
            name, grandparent = folders[parent_id]
            # The tree's top node (its own parent is 0) is a synthetic root —
            # "root/Bookmarks Toolbar/Reading" adds nothing.
            if name and grandparent != 0:
                names.append(name)
            parent_id = grandparent
        return "/".join(reversed(names))

    return [
        Bookmark(
            url=r["url"],
            title=(r["title"] or "").strip(),
            folder=folder_path(r["parent"]),
            browser="firefox",
        )
        for r in rows
    ]


# ---- Chrome / Edge (shared JSON format) ------------------------------------

CHROMIUM_PATHS = {
    "chrome": "AppData/Local/Google/Chrome/User Data/Default/Bookmarks",
    "edge": "AppData/Local/Microsoft/Edge/User Data/Default/Bookmarks",
    "brave": "AppData/Local/BraveSoftware/Brave-Browser/User Data/Default/Bookmarks",
}


def read_chromium(home: Path, browser: str) -> list[Bookmark]:
    path = home / CHROMIUM_PATHS[browser]
    if not path.is_file():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []

    out: list[Bookmark] = []

    def walk(node: dict, trail: list[str]) -> None:
        if node.get("type") == "url":
            url = node.get("url", "")
            if url.startswith("http"):
                out.append(
                    Bookmark(url, (node.get("name") or "").strip(),
                             "/".join(trail), browser)
                )
            return
        name = node.get("name") or ""
        for child in node.get("children", []):
            walk(child, trail + [name] if name else trail)

    for root in (data.get("roots") or {}).values():
        if isinstance(root, dict):
            walk(root, [])
    return out


# ---- dedupe + write --------------------------------------------------------

def deduplicate(bookmarks: list[Bookmark]) -> tuple[list[Merged], dict[str, int]]:
    """Collapse by normalized URL. Returns (merged, stats)."""
    merged: dict[str, Merged] = {}
    stats: dict[str, int] = defaultdict(int)

    for mark in bookmarks:
        stats["total"] += 1
        ok, reason = is_fetchable(mark.url)
        if not ok:
            stats[f"skipped: {reason}"] += 1
            continue
        key = normalize_url(mark.url)
        entry = merged.get(key)
        if entry is None:
            merged[key] = Merged(url=mark.url, title=mark.title)
            entry = merged[key]
        else:
            stats["duplicates merged"] += 1
            # Prefer the more descriptive title.
            if len(mark.title) > len(entry.title):
                entry.title = mark.title
        entry.browsers.add(mark.browser)
        if mark.folder:
            entry.folders.add(mark.folder)

    stats["unique"] = len(merged)
    return list(merged.values()), dict(stats)


def write_links(entries: list[Merged], out_path: Path) -> None:
    lines = ["# Generated by scripts/import_bookmarks.py — one URL per line.\n"]
    for entry in sorted(entries, key=lambda e: e.url):
        comment_parts = [entry.title] if entry.title else []
        if entry.folders:
            comment_parts.append("[" + "; ".join(sorted(entry.folders)) + "]")
        comment_parts.append("(" + ",".join(sorted(entry.browsers)) + ")")
        comment = " ".join(comment_parts).replace("\n", " ").strip()
        lines.append(f"{entry.url} {comment}".rstrip() + "\n")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", help="Path of the .links file to write")
    parser.add_argument(
        "--browser", action="append",
        choices=["firefox", "chrome", "edge", "brave"],
        help="Limit to these browsers (repeatable; default: all found)",
    )
    parser.add_argument("--dry-run", action="store_true",
                        help="Report counts without writing the file")
    args = parser.parse_args()

    home = _windows_home() or Path.home()
    wanted = set(args.browser or ["firefox", "chrome", "edge", "brave"])

    bookmarks: list[Bookmark] = []
    per_browser: dict[str, int] = {}

    if "firefox" in wanted:
        found = []
        for profile in firefox_profiles(home):
            found.extend(read_firefox(profile))
        if found:
            per_browser["firefox"] = len(found)
            bookmarks.extend(found)

    for browser in ("chrome", "edge", "brave"):
        if browser in wanted:
            found = read_chromium(home, browser)
            if found:
                per_browser[browser] = len(found)
                bookmarks.extend(found)

    if not bookmarks:
        print("No bookmarks found. Checked home:", home, file=sys.stderr)
        raise SystemExit(1)

    entries, stats = deduplicate(bookmarks)

    print("Bookmarks found:")
    for browser, count in sorted(per_browser.items()):
        print(f"  {browser:<8} {count}")
    print(f"\n  {stats.get('total', 0)} total"
          f" -> {stats['unique']} unique"
          f" ({stats.get('duplicates merged', 0)} duplicates merged)")
    for key, count in sorted(stats.items()):
        if key.startswith("skipped: "):
            print(f"  skipped {count}: {key[9:]}")

    if args.dry_run:
        print("\n(dry run — nothing written)")
        return
    write_links(entries, Path(args.output))
    print(f"\nWrote {len(entries)} links to {args.output}")
    print("Run `image-search index` to fetch and index them.")


if __name__ == "__main__":
    main()
