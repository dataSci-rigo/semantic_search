import textwrap

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
          face_model: buffalo_l
        folders:
          "~/Photos":
            faces: off
        """,
    )
    config = load_config(path)
    assert "faces" not in config.folders["~/Photos"].processors


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
