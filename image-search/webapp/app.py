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

import hashlib
import io
import os
import sys
import time
from pathlib import Path

from flask import Flask, Response, abort, jsonify, render_template, request

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from image_search.config import load_config  # noqa: E402
from image_search.registry import Registry  # noqa: E402
from image_search.search import (  # noqa: E402
    SearchHit,
    private_ids,
    search_similar_images,
    search_text,
)
from image_search.store.db import connect, migrate  # noqa: E402

CONFIG_PATH = os.environ.get("IMAGE_SEARCH_CONFIG", str(PROJECT_ROOT / "config" / "folders.yaml"))
PORT = int(os.environ.get("IMAGE_SEARCH_WEB_PORT", 9100))
THUMB_SIZE = (480, 480)
# Guest mode: hide private-path and nsfw-tagged images from every route.
# Process-level by design — set at launch, no runtime toggle to spoof.
GUEST = os.environ.get("IMAGE_SEARCH_GUEST") == "1" or "--guest" in sys.argv

app = Flask(__name__)

_config = load_config(CONFIG_PATH)
_registry = Registry(_config)

# Same precedence as the CLI: env > the config's `db:` > project default.
DB_PATH = os.environ.get("IMAGE_SEARCH_DB") or (
    str(_config.db_path)
    if _config.db_path is not None
    else str(PROJECT_ROOT / "data" / "pictures_index.db")
)

# So the web app can start (and show "no results yet") even before the
# indexer has run for the first time, instead of erroring on missing tables.
_boot_conn = connect(DB_PATH)
migrate(_boot_conn)
_boot_conn.close()
_folder_keys = list(_config.folders)


def _folder_from_request() -> str:
    """Folder key from ?folder=, defaulting to the first configured folder."""
    folder = request.args.get("folder", "").strip() or _folder_keys[0]
    if folder not in _config.folders:
        abort(400, f"unknown folder {folder!r}")
    return folder


def _db():
    # A fresh connection per request — sqlite3 connections aren't safe to
    # share across Flask's threaded request handling.
    return connect(DB_PATH)


def _excluded(conn) -> set[str] | None:
    """ids hidden this request: the guest-mode exclusion set, else None."""
    return private_ids(conn, _config) if GUEST else None


def _guard_private(image_id: str) -> None:
    """404 direct image routes for hidden ids in guest mode — same response
    as a nonexistent id, so guests can't probe what's being hidden."""
    if not GUEST:
        return
    conn = _db()
    try:
        if image_id in private_ids(conn, _config):
            abort(404)
    finally:
        conn.close()


def _hit_to_dict(hit: SearchHit) -> dict:
    return {
        "image_id": hit.image_id,
        "path": hit.path,
        "score": hit.score,
        "source": hit.source,
        "kind": hit.kind,
        "title": hit.title,
        "url": hit.url,
        "snippet": hit.snippet,
    }


TAG_CHIPS = ("meme", "chart", "screenshot", "document", "photo", "art")
FIELD_CHOICES = ("ocr", "caption")  # unset/anything else -> None ("all fields")


def _tags_from_request() -> set[str]:
    tag = request.args.get("tag", "").strip().lower()
    return {tag} if tag in TAG_CHIPS else set()


def _field_from_request() -> str | None:
    field = request.args.get("field", "").strip().lower()
    return field if field in FIELD_CHOICES else None


@app.route("/")
def index():
    query = request.args.get("q", "").strip()
    folder = _folder_from_request()
    tags = _tags_from_request()
    field = _field_from_request()
    hits: list[SearchHit] = []
    error = None
    if query:
        conn = _db()
        try:
            hits = search_text(
                conn, _config, _registry, folder, query, k=40, tags=tags, field=field,
                exclude_ids=_excluded(conn),
            )
        except Exception as exc:  # noqa: BLE001 - surface to the page, don't 500
            error = str(exc)
        finally:
            conn.close()
    return render_template(
        "index.html", query=query, hits=hits, error=error,
        folders=_folder_keys, folder=folder,
        tag_chips=TAG_CHIPS, tag=(next(iter(tags)) if tags else ""),
        field=(field or ""), guest=GUEST,
    )


@app.route("/similar/<image_id>")
def similar(image_id: str):
    _guard_private(image_id)
    folder = _folder_from_request()
    conn = _db()
    try:
        hits = search_similar_images(
            conn, _config, folder, image_id, k=40, exclude_ids=_excluded(conn)
        )
        error = None
    except ValueError as exc:
        hits = []
        error = str(exc)
    finally:
        conn.close()
    return render_template(
        "index.html", query=f"(similar to {image_id[:8]})", hits=hits, error=error,
        folders=_folder_keys, folder=folder, tag_chips=TAG_CHIPS, tag="", field="",
        guest=GUEST,
    )


@app.route("/api/search")
def api_search():
    query = request.args.get("q", "").strip()
    if not query:
        return jsonify({"ok": False, "error": "missing ?q="}), 400
    folder = _folder_from_request()
    try:
        k = int(request.args.get("k", 10))
    except ValueError:
        return jsonify({"ok": False, "error": "k must be an integer"}), 400
    k = max(1, min(100, k))
    tags = _tags_from_request()
    field = _field_from_request()
    conn = _db()
    try:
        hits = search_text(
            conn, _config, _registry, folder, query, k=k, tags=tags, field=field,
            exclude_ids=_excluded(conn),
        )
    except Exception as exc:  # noqa: BLE001 - JSON error for API callers, not an HTML 500
        return jsonify({"ok": False, "error": str(exc)}), 500
    finally:
        conn.close()
    return jsonify(
        {
            "ok": True, "query": query, "folder": folder,
            "tag": (next(iter(tags)) if tags else None),
            "field": field,
            "hits": [_hit_to_dict(h) for h in hits],
        }
    )


FILES_PAGE_MAX = 5000


@app.route("/api/files")
def api_files():
    """Read-only inventory for cross-machine dedup: every indexed file as
    (folder, relative path, sha256 of its bytes, mtime), paged in path order.
    ?sha256=<hex> instead returns every path holding that content. Hashes are
    the same content ids ingest computes everywhere, so another machine can
    count copies of a file without transferring it."""
    conn = _db()
    try:
        hidden = _excluded(conn) or set()
        sha = request.args.get("sha256", "").strip().lower()
        if sha:
            rows = conn.execute(
                "SELECT folder, path, image_id, mtime FROM files WHERE image_id = ? "
                "ORDER BY folder, path",
                (sha,),
            ).fetchall()
            next_cursor = None
        else:
            try:
                limit = max(1, min(FILES_PAGE_MAX, int(request.args.get("limit", 1000))))
            except ValueError:
                return jsonify({"ok": False, "error": "limit must be an integer"}), 400
            # Paged within one folder: (folder, path) is the key, so a path
            # cursor is exact there.
            folder = _folder_from_request()
            rows = conn.execute(
                "SELECT folder, path, image_id, mtime FROM files "
                "WHERE folder = ? AND path > ? ORDER BY path LIMIT ?",
                (folder, request.args.get("after", ""), limit),
            ).fetchall()
            next_cursor = rows[-1]["path"] if len(rows) == limit else None
        total = conn.execute("SELECT COUNT(*) FROM files").fetchone()[0]
    finally:
        conn.close()
    files = [
        {"folder": r["folder"], "path": r["path"], "sha256": r["image_id"], "mtime": r["mtime"]}
        for r in rows
        if r["image_id"] not in hidden
    ]
    return jsonify({"ok": True, "total_indexed_files": total, "files": files, "next": next_cursor})


@app.route("/api/manifest.csv")
def api_manifest():
    """The full sha256,size_bytes,path manifest of every file under the
    indexed root (written beside the DB by a manifest job), for dedup tools
    that need files the index doesn't cover (.doc, .heic, ...)."""
    from flask import send_file

    manifest = Path(DB_PATH).parent / "sha256-manifest.csv"
    if not manifest.exists():
        abort(404, "no manifest has been generated for this index")
    return send_file(manifest, mimetype="text/csv", as_attachment=True,
                     download_name="sha256-manifest.csv")


@app.route("/api/save", methods=["POST"])
def api_save():
    """Capture endpoint for external savers (e.g. the Discord bot): drop an
    image upload, a URL, or a text note into IMAGE_SEARCH_SAVE_DIR, where the
    next `image-search index` run picks it up. Tailscale-only trust, like
    every other route here."""
    save_dir_env = os.environ.get("IMAGE_SEARCH_SAVE_DIR")
    if not save_dir_env:
        return jsonify(
            {"ok": False, "error": "IMAGE_SEARCH_SAVE_DIR is not configured on the server"}
        ), 400
    save_dir = Path(save_dir_env)
    save_dir.mkdir(parents=True, exist_ok=True)

    if "file" in request.files:
        from werkzeug.utils import secure_filename

        upload = request.files["file"]
        name = secure_filename(upload.filename or "") or "upload.png"
        target = save_dir / f"{int(time.time())}-{name}"
        upload.save(target)
        return jsonify({"ok": True, "kind": "image", "saved": str(target)})

    data = request.get_json(silent=True) or {}
    url = (data.get("url") or "").strip()
    text = (data.get("text") or "").strip()
    if url:
        comment = " ".join((data.get("comment") or "").split())
        links_file = save_dir / "saved.links"
        with links_file.open("a", encoding="utf-8") as fh:
            fh.write(f"{url} {comment}".strip() + "\n")
        return jsonify({"ok": True, "kind": "link", "saved": str(links_file)})
    if text:
        title = (data.get("title") or "").strip()
        content = (f"# {title}\n\n{text}" if title else text) + "\n"
        digest = hashlib.sha256(content.encode()).hexdigest()[:8]
        note_file = save_dir / f"note-{digest}.md"
        note_file.write_text(content, encoding="utf-8")
        return jsonify({"ok": True, "kind": "note", "saved": str(note_file)})
    return jsonify(
        {"ok": False, "error": "provide multipart 'file', or JSON with 'url' or 'text'"}
    ), 400


def _image_path(image_id: str) -> Path:
    """Resolve a stored (folder, relative-path) image row to a local absolute
    path, 404ing when the id is unknown, its folder isn't in this config, or
    the file is gone (e.g. the drive isn't mounted right now)."""
    conn = _db()
    try:
        row = conn.execute(
            "SELECT path, folder FROM images WHERE id = ?", (image_id,)
        ).fetchone()
    finally:
        conn.close()
    if row is None:
        abort(404)
    path = _config.resolve_stored(row["folder"], row["path"])
    if path is None or not path.exists():
        abort(404)
    return path


@app.route("/image/<image_id>")
def image_thumb(image_id: str):
    from PIL import Image

    _guard_private(image_id)
    path = _image_path(image_id)

    img = Image.open(path).convert("RGB")
    img.thumbnail(THUMB_SIZE)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return Response(buf.getvalue(), mimetype="image/jpeg")


@app.route("/full/<image_id>")
def image_full(image_id: str):
    _guard_private(image_id)
    path = _image_path(image_id)
    from flask import send_file

    return send_file(path)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORT, debug=False)
