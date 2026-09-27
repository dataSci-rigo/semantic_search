import importlib.util
import io
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load_app(tmp_path, monkeypatch):
    """Load webapp/app.py fresh with a scratch config/db/save-dir env."""
    config = tmp_path / "folders.yaml"
    photos = tmp_path / "photos"
    photos.mkdir(exist_ok=True)
    config.write_text(f'folders:\n  "{photos}":\n    ocr: fake-ocr\n')
    monkeypatch.setenv("IMAGE_SEARCH_CONFIG", str(config))
    monkeypatch.setenv("IMAGE_SEARCH_DB", str(tmp_path / "index.db"))
    monkeypatch.setenv("IMAGE_SEARCH_SAVE_DIR", str(tmp_path / "saved"))

    spec = importlib.util.spec_from_file_location("webapp_app_under_test", ROOT / "webapp" / "app.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_save_url_appends_to_links_file(tmp_path, monkeypatch):
    mod = _load_app(tmp_path, monkeypatch)
    client = mod.app.test_client()

    resp = client.post("/api/save", json={"url": "https://example.com/a", "comment": "neat"})
    assert resp.status_code == 200 and resp.get_json()["kind"] == "link"
    resp = client.post("/api/save", json={"url": "https://example.com/b"})
    assert resp.status_code == 200

    links = (tmp_path / "saved" / "saved.links").read_text().splitlines()
    assert links == ["https://example.com/a neat", "https://example.com/b"]


def test_save_text_writes_note_file(tmp_path, monkeypatch):
    mod = _load_app(tmp_path, monkeypatch)
    client = mod.app.test_client()

    resp = client.post("/api/save", json={"text": "remember this", "title": "A Thought"})
    body = resp.get_json()
    assert resp.status_code == 200 and body["kind"] == "note"
    content = Path(body["saved"]).read_text()
    assert content.startswith("# A Thought")
    assert "remember this" in content


def test_save_file_stores_upload(tmp_path, monkeypatch):
    mod = _load_app(tmp_path, monkeypatch)
    client = mod.app.test_client()

    resp = client.post(
        "/api/save",
        data={"file": (io.BytesIO(b"fake image bytes"), "my meme.png")},
        content_type="multipart/form-data",
    )
    body = resp.get_json()
    assert resp.status_code == 200 and body["kind"] == "image"
    saved = Path(body["saved"])
    assert saved.parent == tmp_path / "saved"
    assert saved.name.endswith("my_meme.png")
    assert saved.read_bytes() == b"fake image bytes"


def test_save_rejects_empty_and_unconfigured(tmp_path, monkeypatch):
    mod = _load_app(tmp_path, monkeypatch)
    client = mod.app.test_client()

    assert client.post("/api/save", json={}).status_code == 400

    monkeypatch.delenv("IMAGE_SEARCH_SAVE_DIR")
    resp = client.post("/api/save", json={"url": "https://example.com"})
    assert resp.status_code == 400
    assert "IMAGE_SEARCH_SAVE_DIR" in resp.get_json()["error"]


def test_files_inventory_pages_and_looks_up_by_hash(tmp_path, monkeypatch):
    mod = _load_app(tmp_path, monkeypatch)
    from image_search.store import images as images_store
    from image_search.store.db import connect

    folder = next(iter(mod._config.folders))
    conn = connect(tmp_path / "index.db")
    for name, sha in (("a.jpg", "h1"), ("b.jpg", "h2"), ("c.jpg", "h1")):
        images_store.upsert_file(conn, name, folder, sha, 1.0)
    conn.commit()
    conn.close()
    client = mod.app.test_client()

    first = client.get("/api/files?limit=2").get_json()
    assert [f["path"] for f in first["files"]] == ["a.jpg", "b.jpg"]
    assert first["total_indexed_files"] == 3 and first["next"] == "b.jpg"
    rest = client.get(f"/api/files?limit=2&after={first['next']}").get_json()
    assert [f["path"] for f in rest["files"]] == ["c.jpg"] and rest["next"] is None

    copies = client.get("/api/files?sha256=H1").get_json()["files"]
    assert sorted(f["path"] for f in copies) == ["a.jpg", "c.jpg"]
