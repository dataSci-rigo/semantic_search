#!/usr/bin/env python3
"""UNTESTED scaffold: pull your Reddit saved posts into the drop folder.

Image posts are downloaded as files; link/text posts are appended to
saved.links / written as notes — the same formats `image-search index`
already ingests (see docs/saved-ingestion.md). Requires a script-type app
(reddit.com -> prefs -> apps) and these env vars:

  REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET, REDDIT_USERNAME, REDDIT_PASSWORD

Usage: python scripts/fetch_reddit_saved.py <drop_folder>

This script has never been run against the live API (written without
credentials available). It only writes into the drop folder, so the worst
failure mode is junk files there — review its output on first run.
"""

from __future__ import annotations

import os
import sys
import urllib.parse
import urllib.request
import json
from pathlib import Path

USER_AGENT = "image-search-saved-fetcher/0.1"
IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".webp", ".gif")


def get_token() -> str:
    auth = f"{os.environ['REDDIT_CLIENT_ID']}:{os.environ['REDDIT_CLIENT_SECRET']}"
    import base64

    body = urllib.parse.urlencode(
        {
            "grant_type": "password",
            "username": os.environ["REDDIT_USERNAME"],
            "password": os.environ["REDDIT_PASSWORD"],
        }
    ).encode()
    req = urllib.request.Request(
        "https://www.reddit.com/api/v1/access_token",
        data=body,
        headers={
            "Authorization": "Basic " + base64.b64encode(auth.encode()).decode(),
            "User-Agent": USER_AGENT,
        },
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.load(resp)["access_token"]


def fetch_saved(token: str):
    after = None
    while True:
        params = {"limit": "100"}
        if after:
            params["after"] = after
        url = (
            f"https://oauth.reddit.com/user/{os.environ['REDDIT_USERNAME']}/saved?"
            + urllib.parse.urlencode(params)
        )
        req = urllib.request.Request(
            url, headers={"Authorization": f"bearer {token}", "User-Agent": USER_AGENT}
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            payload = json.load(resp)["data"]
        yield from payload["children"]
        after = payload.get("after")
        if not after:
            return


def main() -> None:
    if len(sys.argv) != 2:
        sys.exit("usage: fetch_reddit_saved.py <drop_folder>")
    drop = Path(sys.argv[1]).expanduser()
    drop.mkdir(parents=True, exist_ok=True)
    links_file = drop / "saved.links"

    token = get_token()
    downloaded = linked = noted = 0
    for child in fetch_saved(token):
        data = child["data"]
        title = " ".join((data.get("title") or data.get("link_title") or "").split())
        post_url = data.get("url_overridden_by_dest") or data.get("url") or ""
        selftext = (data.get("selftext") or data.get("body") or "").strip()

        if post_url.lower().endswith(IMAGE_SUFFIXES):
            name = f"reddit-{data['id']}{Path(urllib.parse.urlparse(post_url).path).suffix}"
            target = drop / name
            if not target.exists():
                req = urllib.request.Request(post_url, headers={"User-Agent": USER_AGENT})
                with urllib.request.urlopen(req, timeout=30) as resp:
                    target.write_bytes(resp.read())
                downloaded += 1
        elif selftext:
            target = drop / f"reddit-{data['id']}.md"
            if not target.exists():
                target.write_text(f"# {title}\n\n{selftext}\n", encoding="utf-8")
                noted += 1
        elif post_url:
            line = f"{post_url} {title}".strip() + "\n"
            existing = links_file.read_text() if links_file.exists() else ""
            if post_url not in existing:
                with links_file.open("a", encoding="utf-8") as fh:
                    fh.write(line)
                linked += 1

    print(f"downloaded {downloaded} images, {linked} links, {noted} notes -> {drop}")


if __name__ == "__main__":
    main()
