import argparse

from PIL import Image

from image_search.cli import cmd_dupes
from image_search.store import images as images_store
from image_search.store.db import connect, migrate


def _setup(tmp_path):
    folder = tmp_path / "shots"
    folder.mkdir()
    Image.new("RGB", (4, 4), (9, 9, 9)).save(folder / "a.png")
    Image.new("RGB", (4, 4), (9, 9, 9)).save(folder / "b.png")  # identical bytes
    Image.new("RGB", (4, 4), (1, 1, 1)).save(folder / "unique.png")

    config_path = tmp_path / "folders.yaml"
    config_path.write_text(f"folders:\n  shots:\n    path: {folder}\n")

    db = tmp_path / "test.db"
    conn = connect(db)
    migrate(conn)
    for name in ("a.png", "b.png", "unique.png"):
        path = folder / name
        images_store.upsert_file(
            conn, name, "shots", images_store.content_hash(path), path.stat().st_mtime
        )
    conn.commit()
    conn.close()
    return folder, db, config_path


def test_dupes_lists_without_deleting(tmp_path, capsys):
    folder, db, config_path = _setup(tmp_path)
    cmd_dupes(argparse.Namespace(config=str(config_path), db=str(db), delete=False))
    out = capsys.readouterr().out
    assert "keep" in out and "dup" in out and "--delete" in out
    assert (folder / "a.png").exists() and (folder / "b.png").exists()


def test_dupes_delete_keeps_first_path_per_group(tmp_path, capsys):
    folder, db, config_path = _setup(tmp_path)
    cmd_dupes(argparse.Namespace(config=str(config_path), db=str(db), delete=True))
    out = capsys.readouterr().out
    assert "Deleted 1 duplicate" in out
    # First path alphabetically is kept; the extra copy is gone from disk and DB.
    assert (folder / "a.png").exists()
    assert not (folder / "b.png").exists()
    assert (folder / "unique.png").exists()

    conn = connect(db)
    assert conn.execute("SELECT COUNT(*) AS n FROM files").fetchone()["n"] == 2


def test_dupes_ignores_folders_missing_from_config(tmp_path, capsys):
    """Copies whose folder isn't in the config are never listed or deleted."""
    folder, db, config_path = _setup(tmp_path)
    conn = connect(db)
    dup_id = images_store.content_hash(folder / "a.png")
    images_store.upsert_file(conn, "elsewhere/a.png", "other-folder", dup_id, 1.0)
    conn.commit()
    conn.close()

    cmd_dupes(argparse.Namespace(config=str(config_path), db=str(db), delete=False))
    captured = capsys.readouterr()
    assert "other-folder" in captured.err
    assert "elsewhere" not in captured.out
