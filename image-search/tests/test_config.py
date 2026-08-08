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
