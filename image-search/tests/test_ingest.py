import textwrap

import pytest
from PIL import Image

from image_search.config import load_config
from image_search.ingest import ingest_folder
from image_search.processors.base import (
    ImageEmbedRecord,
    LoadedImage,
    OcrRecord,
    Record,
    TextEmbedRecord,
)
from image_search.registry import Registry
from image_search.store.db import connect, migrate


class FakeOcr:
    kind = "ocr"
    model_id = "fake-ocr"

    def load(self) -> None:
        pass

    def process(self, img: LoadedImage) -> list[Record]:
        return [OcrRecord(text=f"text for {img.image_id[:8]}")]


class FakeTextEmbed:
    kind = "text_embed"
    model_id = "fake-embed"

    def load(self) -> None:
        pass

    def embed(self, text: str) -> list[float]:
        # Deterministic fake embedding: length bucket into a fixed dim.
        return [float(len(text) % 7), 1.0, 0.0]

    def process(self, img: LoadedImage) -> list[Record]:
        if not img.text.strip():
            return []
        return [TextEmbedRecord(model=self.model_id, vector=self.embed(img.text))]


class FakeImageEmbed:
    kind = "image_embed"
    model_id = "fake-image-embed"

    def load(self) -> None:
        pass

    def process(self, img: LoadedImage) -> list[Record]:
        return [ImageEmbedRecord(model=self.model_id, vector=[1.0, 0.0, 0.0])]


def make_config(tmp_path, folder, with_text_embed=True, with_image_embed=False):
    config_path = tmp_path / "folders.yaml"
    lines = [
        "folders:",
        f'  "{folder}":',
        "    ocr: fake-ocr",
    ]
    if with_text_embed:
        lines.append("    text_embed: fake-embed")
    if with_image_embed:
        lines.append("    image_embed: fake-image-embed")
    config_path.write_text("\n".join(lines) + "\n")
    return load_config(config_path)


def fake_registry(config):
    registry = Registry(config)
    registry._instances[("ocr", "fake-ocr")] = FakeOcr()
    registry._instances[("text_embed", "fake-embed")] = FakeTextEmbed()
    registry._instances[("image_embed", "fake-image-embed")] = FakeImageEmbed()
    return registry


def test_ingest_writes_ocr_and_fts_without_text_embed(tmp_path):
    """No vector store dependency needed: ocr-only folder config."""
    folder = tmp_path / "shots"
    folder.mkdir()
    Image.new("RGB", (4, 4)).save(folder / "one.png")

    config = make_config(tmp_path, str(folder), with_text_embed=False)
    registry = fake_registry(config)

    conn = connect(tmp_path / "test.db")
    migrate(conn)

    stats = ingest_folder(conn, config, registry, str(folder))
    assert stats == {"seen": 1, "skipped": 0, "indexed": 1}

    ocr_rows = conn.execute("SELECT text FROM ocr_text").fetchall()
    assert len(ocr_rows) == 1
    assert ocr_rows[0]["text"].startswith("text for")

    fts_rows = conn.execute("SELECT * FROM text_fts").fetchall()
    assert len(fts_rows) == 1

    vec_rows = conn.execute("SELECT COUNT(*) AS n FROM vec_map").fetchone()
    assert vec_rows["n"] == 0


def test_ingest_writes_ocr_text_and_vector(tmp_path):
    pytest.importorskip("sqlite_vec", reason="requires sqlite-vec (Phase 1 dependency)")

    folder = tmp_path / "shots"
    folder.mkdir()
    Image.new("RGB", (4, 4)).save(folder / "one.png")

    config = make_config(tmp_path, str(folder))
    registry = fake_registry(config)

    conn = connect(tmp_path / "test.db")
    migrate(conn)

    stats = ingest_folder(conn, config, registry, str(folder))
    assert stats == {"seen": 1, "skipped": 0, "indexed": 1}

    ocr_rows = conn.execute("SELECT text FROM ocr_text").fetchall()
    assert len(ocr_rows) == 1
    assert ocr_rows[0]["text"].startswith("text for")

    fts_rows = conn.execute("SELECT * FROM text_fts").fetchall()
    assert len(fts_rows) == 1

    vec_rows = conn.execute("SELECT COUNT(*) AS n FROM vec_map").fetchone()
    assert vec_rows["n"] == 1


def test_ingest_is_idempotent_on_rerun(tmp_path):
    pytest.importorskip("sqlite_vec", reason="requires sqlite-vec (Phase 1 dependency)")

    folder = tmp_path / "shots"
    folder.mkdir()
    Image.new("RGB", (4, 4)).save(folder / "one.png")

    config = make_config(tmp_path, str(folder))
    registry = fake_registry(config)

    conn = connect(tmp_path / "test.db")
    migrate(conn)

    first = ingest_folder(conn, config, registry, str(folder))
    second = ingest_folder(conn, config, registry, str(folder))

    assert first["indexed"] == 1
    assert second["indexed"] == 0
    assert second["skipped"] == 1

    # Records aren't duplicated on the re-run.
    assert conn.execute("SELECT COUNT(*) AS n FROM ocr_text").fetchone()["n"] == 1


def test_ingest_writes_image_embed_vector_to_its_own_space(tmp_path):
    pytest.importorskip("sqlite_vec", reason="requires sqlite-vec (Phase 1 dependency)")

    folder = tmp_path / "shots"
    folder.mkdir()
    Image.new("RGB", (4, 4)).save(folder / "one.png")

    config = make_config(tmp_path, str(folder), with_text_embed=True, with_image_embed=True)
    registry = fake_registry(config)

    conn = connect(tmp_path / "test.db")
    migrate(conn)

    stats = ingest_folder(conn, config, registry, str(folder))
    assert stats == {"seen": 1, "skipped": 0, "indexed": 1}

    rows = conn.execute("SELECT vec_table, COUNT(*) AS n FROM vec_map GROUP BY vec_table").fetchall()
    tables = {r["vec_table"]: r["n"] for r in rows}
    assert tables == {
        "vec_text__fake_embed": 1,
        "vec_image__fake_image_embed": 1,
    }
