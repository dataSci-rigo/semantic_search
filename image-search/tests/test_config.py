import textwrap
from pathlib import Path

import pytest

from image_search.config import load_config


def write_config(tmp_path, text):
    path = tmp_path / "folders.yaml"
    path.write_text(textwrap.dedent(text))
    return path


def test_defaults_are_inherited(tmp_path):
    path = write_config(
        tmp_path,
        """
        defaults:
          text_embed: bge-small-en-v1.5
        folders:
          "~/Screenshots":
            ocr: paddle-ppocrv5
        """,
    )
    config = load_config(path)
    folder = config.folders["~/Screenshots"]
    assert folder.processors["ocr"] == "paddle-ppocrv5"
    assert folder.processors["text_embed"] == "bge-small-en-v1.5"


def test_off_excludes_processor(tmp_path):
    path = write_config(
        tmp_path,
        """
        defaults:
          faces: buffalo_l
        folders:
          "~/Photos":
            faces: off
        """,
    )
    config = load_config(path)
    assert "faces" not in config.folders["~/Photos"].processors


def test_unknown_keys_warn(tmp_path):
    path = write_config(
        tmp_path,
        """
        defaults:
          face_model: buffalo_l
        folders:
          "~/Photos":
            ocr: rapidocr
            face_detect: scrfd_10g
        """,
    )
    with pytest.warns(UserWarning, match="face_model"):
        with pytest.warns(UserWarning, match="face_detect"):
            config = load_config(path)
    # Unknown keys are dropped, known ones survive.
    assert config.folders["~/Photos"].processors == {"ocr": "rapidocr"}


def test_bool_model_value_raises(tmp_path):
    # YAML parses bare `on` as boolean True — reject loudly instead of
    # passing True downstream as a model id.
    path = write_config(
        tmp_path,
        """
        folders:
          "~/Photos":
            ocr: on
        """,
    )
    with pytest.raises(ValueError, match="ocr"):
        load_config(path)


def test_explicit_override_beats_default(tmp_path):
    path = write_config(
        tmp_path,
        """
        defaults:
          text_embed: bge-small-en-v1.5
        folders:
          "~/A":
            text_embed: all-MiniLM-L6-v2
        """,
    )
    config = load_config(path)
    assert config.folders["~/A"].processors["text_embed"] == "all-MiniLM-L6-v2"


def test_warns_on_divergent_text_embed_models(tmp_path):
    path = write_config(
        tmp_path,
        """
        folders:
          "~/A":
            text_embed: bge-small-en-v1.5
          "~/B":
            text_embed: all-MiniLM-L6-v2
        """,
    )
    with pytest.warns(UserWarning, match="text_embed"):
        load_config(path)


def test_active_processors_union(tmp_path):
    path = write_config(
        tmp_path,
        """
        folders:
          "~/A":
            ocr: paddle-ppocrv5
            text_embed: bge-small-en-v1.5
          "~/B":
            text_embed: bge-small-en-v1.5
        """,
    )
    config = load_config(path)
    assert config.active_processors() == {
        ("ocr", "paddle-ppocrv5"),
        ("text_embed", "bge-small-en-v1.5"),
    }


def test_path_override_used_for_matching_subtree(tmp_path):
    path = write_config(
        tmp_path,
        """
        defaults:
          text_embed: bge-small-en-v1.5
        folders:
          "~/Pictures":
            caption: moondream2
            overrides:
              Screenshots:
                ocr: rapidocr
        """,
    )
    config = load_config(path)
    folder = config.folders["~/Pictures"]

    assert folder.processors_for_path(Path("~/Pictures/dw2/photo.jpg")) == {
        "caption": "moondream2",
        "text_embed": "bge-small-en-v1.5",
    }
    assert folder.processors_for_path(Path("~/Pictures/dw2/Screenshots/shot.png")) == {
        "ocr": "rapidocr",
        "text_embed": "bge-small-en-v1.5",
    }


def test_path_override_match_is_case_insensitive(tmp_path):
    path = write_config(
        tmp_path,
        """
        folders:
          "~/Pictures":
            caption: moondream2
            overrides:
              Screenshots:
                ocr: rapidocr
        """,
    )
    config = load_config(path)
    folder = config.folders["~/Pictures"]
    assert folder.processors_for_path(Path("~/Pictures/screenshots/shot.png")) == {
        "ocr": "rapidocr"
    }


def test_active_processors_includes_overrides(tmp_path):
    path = write_config(
        tmp_path,
        """
        folders:
          "~/Pictures":
            caption: moondream2
            overrides:
              Screenshots:
                ocr: rapidocr
        """,
    )
    config = load_config(path)
    assert config.active_processors() == {
        ("caption", "moondream2"),
        ("ocr", "rapidocr"),
    }


def test_private_patterns_and_is_private_path(tmp_path):
    path = write_config(
        tmp_path,
        """
        folders:
          "~/Pictures":
            ocr: rapidocr
        private:
          - "~/Pictures/private"
          - "/mnt/vault/*.png"
        """,
    )
    config = load_config(path)
    home = str(Path("~").expanduser())
    # A bare directory pattern covers its whole subtree.
    assert config.is_private_path(f"{home}/Pictures/private/a/b.jpg")
    assert config.is_private_path("~/Pictures/private/c.jpg")
    assert not config.is_private_path(f"{home}/Pictures/beach.jpg")
    # Glob patterns match with fnmatch semantics.
    assert config.is_private_path("/mnt/vault/x.png")
    assert not config.is_private_path("/mnt/vault/x.txt")


def test_private_defaults_empty(tmp_path):
    path = write_config(
        tmp_path,
        """
        folders:
          "~/Pictures":
            ocr: rapidocr
        """,
    )
    config = load_config(path)
    assert config.private_patterns == ()
    assert not config.is_private_path("/anything/at/all.jpg")


def test_private_rejects_non_list(tmp_path):
    path = write_config(
        tmp_path,
        """
        folders:
          "~/Pictures":
            ocr: rapidocr
        private: 5
        """,
    )
    with pytest.raises(ValueError):
        load_config(path)


# --- named folders, config-relative paths, exclude_dirs ----------------------


def test_named_folder_with_explicit_path(tmp_path):
    root = tmp_path / "library"
    root.mkdir()
    config_path = tmp_path / "folders.yaml"
    config_path.write_text(f"folders:\n  photos:\n    path: {root}\n    ocr: x\n")
    config = load_config(config_path)
    assert set(config.folders) == {"photos"}
    assert config.folders["photos"].path == root


def test_drive_resident_config_resolves_relative_path_and_db(tmp_path):
    meta = tmp_path / ".semantic_search"
    meta.mkdir()
    config_path = meta / "folders.yaml"
    config_path.write_text("db: index.db\nfolders:\n  drive:\n    path: ..\n    ocr: x\n")
    config = load_config(config_path)
    assert config.folders["drive"].path == tmp_path.resolve()
    assert config.db_path == meta.resolve() / "index.db"


def test_legacy_key_is_path_still_works(tmp_path):
    config_path = tmp_path / "folders.yaml"
    config_path.write_text('folders:\n  "~/Pictures":\n    ocr: x\n')
    config = load_config(config_path)
    folder = config.folders["~/Pictures"]
    assert folder.path == Path("~/Pictures").expanduser()


def test_to_from_db_path_roundtrip(tmp_path):
    config_path = tmp_path / "folders.yaml"
    config_path.write_text(f"folders:\n  d:\n    path: {tmp_path}\n    ocr: x\n")
    folder = load_config(config_path).folders["d"]
    original = tmp_path / "nested dir" / "uni-códe.png"
    stored = folder.to_db_path(original)
    assert stored == "nested dir/uni-códe.png"  # POSIX separators, relative
    assert folder.from_db_path(stored) == original


def test_exclude_dirs_default_and_override(tmp_path):
    config_path = tmp_path / "folders.yaml"
    config_path.write_text(
        f"folders:\n"
        f"  a:\n    path: {tmp_path}\n    ocr: x\n"
        f"  b:\n    path: {tmp_path}\n    ocr: x\n    exclude_dirs: [backup]\n"
        f"  c:\n    path: {tmp_path}\n    ocr: x\n    exclude_dirs: []\n"
    )
    config = load_config(config_path)
    from image_search.config import DEFAULT_EXCLUDE_DIRS

    assert config.folders["a"].effective_exclude_dirs() == DEFAULT_EXCLUDE_DIRS
    assert config.folders["b"].effective_exclude_dirs() == ("backup",)
    assert config.folders["c"].effective_exclude_dirs() == ()


def test_relative_private_patterns_anchor_to_config_dir(tmp_path):
    meta = tmp_path / ".semantic_search"
    meta.mkdir()
    config_path = meta / "folders.yaml"
    config_path.write_text(
        "folders:\n  d:\n    path: ..\n    ocr: x\nprivate:\n  - pictures/private\n"
    )
    config = load_config(config_path)
    assert config.private_patterns == (str(meta.resolve() / "pictures" / "private"),)
