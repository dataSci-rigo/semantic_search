#!/usr/bin/env python3
"""
Photo search — web UI + JSON API over the image-search index.
Served at http://<tailscale-ip>:9100/. Safe to use without auth because
access is Tailscale-only (same assumption as server/panel/app.py on the VM).

This has to run on the laptop, not the VM: the search index, embedding
models, and GPU all live here (see docs/gpu-setup.md). The Discord bot
(hosted on the VM) calls /api/search over Tailscale instead of importing
this code directly.
"""

from __future__ import annotations

import io
import os
import sys
from pathlib import Path

from flask import Flask, Response, abort, jsonify, render_template, request

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from image_search.config import load_config  # noqa: E402
from image_search.registry import Registry  # noqa: E402
from image_search.search import SearchHit, search_similar_images, search_text  # noqa: E402
from image_search.store.db import connect, migrate  # noqa: E402

CONFIG_PATH = os.environ.get("IMAGE_SEARCH_CONFIG", str(PROJECT_ROOT / "config" / "folders.yaml"))
DB_PATH = os.environ.get(
    "IMAGE_SEARCH_DB", str(PROJECT_ROOT / "data" / "pictures_index.db")
)
PORT = int(os.environ.get("IMAGE_SEARCH_WEB_PORT", 9100))
THUMB_SIZE = (480, 480)

app = Flask(__name__)

_config = load_config(CONFIG_PATH)
_registry = Registry(_config)

# So the web app can start (and show "no results yet") even before the
# indexer has run for the first time, instead of erroring on missing tables.
migrate(connect(DB_PATH))
# This is the only folder in the real config today (see config/folders.yaml);
# a multi-folder library would need a folder selector in the UI/API.
_folder_key = next(iter(_config.folders))


def _db():
    # A fresh connection per request — sqlite3 connections aren't safe to
    # share across Flask's threaded request handling.
    return connect(DB_PATH)


def _hit_to_dict(hit: SearchHit) -> dict:
    return {
        "image_id": hit.image_id,
        "path": hit.path,
        "score": hit.score,
        "source": hit.source,
    }


@app.route("/")
def index():
    query = request.args.get("q", "").strip()
    hits: list[SearchHit] = []
    error = None
    if query:
        conn = _db()
        try:
            hits = search_text(conn, _config, _registry, _folder_key, query, k=40)
        except Exception as exc:  # noqa: BLE001 - surface to the page, don't 500
            error = str(exc)
        finally:
            conn.close()
    return render_template("index.html", query=query, hits=hits, error=error)


@app.route("/similar/<image_id>")
def similar(image_id: str):
    conn = _db()
    try:
        hits = search_similar_images(conn, _config, _folder_key, image_id, k=40)
        error = None
    except ValueError as exc:
        hits = []
        error = str(exc)
    finally:
        conn.close()
    return render_template(
        "index.html", query=f"(similar to {image_id[:8]})", hits=hits, error=error
    )


@app.route("/api/search")
def api_search():
    query = request.args.get("q", "").strip()
    if not query:
        return jsonify({"ok": False, "error": "missing ?q="}), 400
    k = int(request.args.get("k", 10))
    conn = _db()
    try:
        hits = search_text(conn, _config, _registry, _folder_key, query, k=k)
    finally:
        conn.close()
    return jsonify({"ok": True, "query": query, "hits": [_hit_to_dict(h) for h in hits]})


@app.route("/image/<image_id>")
def image_thumb(image_id: str):
    from PIL import Image

    conn = _db()
    try:
        row = conn.execute("SELECT path FROM images WHERE id = ?", (image_id,)).fetchone()
    finally:
        conn.close()
    if row is None:
        abort(404)

    path = Path(row["path"])
    if not path.exists():
        abort(404)

    img = Image.open(path).convert("RGB")
    img.thumbnail(THUMB_SIZE)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return Response(buf.getvalue(), mimetype="image/jpeg")


@app.route("/full/<image_id>")
def image_full(image_id: str):
    conn = _db()
    try:
        row = conn.execute("SELECT path FROM images WHERE id = ?", (image_id,)).fetchone()
    finally:
        conn.close()
    if row is None:
        abort(404)
    path = Path(row["path"])
    if not path.exists():
        abort(404)
    from flask import send_file

    return send_file(path)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORT, debug=False)
