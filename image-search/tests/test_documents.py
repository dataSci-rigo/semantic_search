"""docx/csv/xlsx document ingestion: parsers (title + body extraction rules)
and the shared _ingest_document pipeline. CSV/XLSX bodies are METADATA ONLY —
filename/sheet names and column names, never row data."""

import textwrap
from pathlib import Path

import pytest

from image_search import textitems
from image_search.ingest import _ingest_phase, ingest_folder
from image_search.store.db import connect, migrate

from test_ingest import fake_registry, make_config


# --- parse_csv (stdlib only, always runnable) --------------------------------


def test_parse_csv_header_row_only(tmp_path):
    path = tmp_path / "signups.csv"
    path.write_text("name,email,signup_date\nalice,a@x.com,2024-01-01\nbob,b@x.com,2024-02-02\n")
    title, body = textitems.parse_csv(path)
    assert title == "signups"
    assert "name" in body and "email" in body and "signup_date" in body
    assert "alice" not in body and "a@x.com" not in body  # never row data


def test_parse_csv_semicolon_dialect(tmp_path):
    path = tmp_path / "eu.csv"
    path.write_text("stadt;land;fluss\nBerlin;DE;Spree\n")
    _, body = textitems.parse_csv(path)
    assert "stadt" in body and "fluss" in body
    assert "Berlin" not in body


def test_parse_csv_headerless_is_empty(tmp_path):
    path = tmp_path / "readings.csv"
    path.write_text("1,2,3\n4,5,6\n7,8,9\n")
    title, body = textitems.parse_csv(path)
    assert title == "readings"
    assert body == ""  # no header -> nothing safe to index


def test_parse_csv_single_column_falls_back(tmp_path):
    # Sniffer raises on single-column files; the excel-dialect fallback and
    # header default keep the first cell.
    path = tmp_path / "hosts.csv"
    path.write_text("hostname\nweb01\nweb02\n")
    _, body = textitems.parse_csv(path)
    assert "hostname" in body
    assert "web01" not in body


# --- parse_docx --------------------------------------------------------------


def _make_docx(path: Path, heading: str | None, paragraphs: list[str]):
    docx = pytest.importorskip("docx", reason="requires python-docx")
    document = docx.Document()
    if heading:
        document.add_heading(heading, level=1)
    for para in paragraphs:
        document.add_paragraph(para)
    document.save(str(path))


def test_parse_docx_title_from_heading(tmp_path):
    path = tmp_path / "plan.docx"
    _make_docx(path, "Quarterly Plan", ["First goal is shipping.", "Second goal is speed."])
    title, body = textitems.parse_docx(path)
    assert title == "Quarterly Plan"
    assert "shipping" in body and "speed" in body


def test_parse_docx_title_falls_back_to_stem(tmp_path):
    path = tmp_path / "meeting-notes.docx"
    _make_docx(path, None, ["Just some prose."])
    title, body = textitems.parse_docx(path)
    assert title == "meeting-notes"
    assert "prose" in body


def test_parse_docx_body_capped(tmp_path):
    path = tmp_path / "big.docx"
    _make_docx(path, None, ["word " * 200] * 40)
    _, body = textitems.parse_docx(path)
    assert len(body) <= textitems.DOC_BODY_CAP


# --- parse_xlsx --------------------------------------------------------------


def _make_xlsx(path: Path, sheets: dict[str, list[list]]):
    openpyxl = pytest.importorskip("openpyxl", reason="requires openpyxl")
    wb = openpyxl.Workbook()
    wb.remove(wb.active)
    for name, rows in sheets.items():
        ws = wb.create_sheet(title=name)
        for row in rows:
            ws.append(row)
    wb.save(str(path))


def test_parse_xlsx_sheets_and_headers_only(tmp_path):
    path = tmp_path / "finances2.xlsx"
    _make_xlsx(
        path,
        {
            "Budget": [["category", "planned", "actual"], ["rent", 1000, 990]],
            "Staff": [["name", "role"], ["dana", "engineer"]],
        },
    )
    title, body = textitems.parse_xlsx(path)
    assert title == "finances2"
    assert "Budget" in body and "Staff" in body
    assert "category" in body and "role" in body
    assert "rent" not in body and "dana" not in body  # never row data


def test_parse_xlsx_empty_sheet_name_still_counts(tmp_path):
    path = tmp_path / "tabs.xlsx"
    _make_xlsx(path, {"Ideas 2026": []})
    _, body = textitems.parse_xlsx(path)
    assert "Ideas 2026" in body


def test_parse_xlsx_sheet_cap(tmp_path):
    path = tmp_path / "many.xlsx"
    _make_xlsx(path, {f"S{i}": [["col"]] for i in range(textitems.XLSX_SHEET_CAP + 5)})
    _, body = textitems.parse_xlsx(path)
    assert len(body.splitlines()) == textitems.XLSX_SHEET_CAP


# --- financial filename gate is extension-agnostic ---------------------------


@pytest.mark.parametrize(
    "name", ["statement.xlsx", "invoice-4471.docx", "payroll_june.csv", "W-2_2023.docx"]
)
def test_looks_financial_matches_document_extensions(name):
    assert textitems.looks_financial(Path(name)) is True


# --- ingestion through _ingest_document --------------------------------------


def _counts(conn, *tables):
    return {
        t: conn.execute(f"SELECT COUNT(*) AS n FROM {t}").fetchone()["n"] for t in tables
    }


def test_ingest_documents_alongside_notes(tmp_path):
    pytest.importorskip("docx")
    pytest.importorskip("openpyxl")
    folder = tmp_path / "docs"
    folder.mkdir()
    _make_docx(folder / "report.docx", "Annual Report", ["Findings within."])
    (folder / "users.csv").write_text("user_id,plan,mrr\n1,pro,49\n")
    _make_xlsx(folder / "sheets.xlsx", {"Metrics": [["kpi", "target"]]})
    (folder / "note.md").write_text("# A note\nhello\n")

    config = make_config(tmp_path, str(folder), with_ocr=False, with_text_embed=False)
    registry = fake_registry(config)
    conn = connect(tmp_path / "test.db")
    migrate(conn)

    stats = ingest_folder(conn, config, registry, str(folder))
    assert stats["indexed"] == 4 and stats["failed"] == 0

    kinds = {r["kind"] for r in conn.execute("SELECT kind FROM items")}
    assert kinds == {"docx", "csv", "xlsx", "note"}
    assert _counts(conn, "text_fts")["text_fts"] == 4

    # Rerun: everything stat-skips.
    second = ingest_folder(conn, config, registry, str(folder))
    assert second["skipped"] == 4 and second["indexed"] == 0

    # Stored paths are folder-relative.
    paths = {r["path"] for r in conn.execute("SELECT path FROM files")}
    assert "report.docx" in paths and "users.csv" in paths


def test_corrupt_docx_fails_one_file_not_the_run(tmp_path):
    pytest.importorskip("docx")
    folder = tmp_path / "docs"
    folder.mkdir()
    (folder / "bad.docx").write_bytes(b"this is not a zip archive")
    (folder / "note.md").write_text("# ok\nfine\n")

    config = make_config(tmp_path, str(folder), with_ocr=False, with_text_embed=False)
    registry = fake_registry(config)
    conn = connect(tmp_path / "test.db")
    migrate(conn)

    stats = ingest_folder(conn, config, registry, str(folder))
    assert stats["failed"] == 1 and stats["indexed"] == 1
    # Rollback: no files/items row for the corrupt file, so it retries next run.
    assert conn.execute(
        "SELECT 1 FROM files WHERE path = 'bad.docx'"
    ).fetchone() is None


def test_headerless_csv_is_thin_and_hidden_from_search(tmp_path):
    folder = tmp_path / "docs"
    folder.mkdir()
    (folder / "dump.csv").write_text("1,2,3\n4,5,6\n")

    config = make_config(tmp_path, str(folder), with_ocr=False, with_text_embed=False)
    registry = fake_registry(config)
    conn = connect(tmp_path / "test.db")
    migrate(conn)

    ingest_folder(conn, config, registry, str(folder))
    row = conn.execute("SELECT status FROM items WHERE kind = 'csv'").fetchone()
    assert row["status"] == textitems.STATUS_THIN


def test_financial_named_xlsx_is_excluded_but_stat_skips(tmp_path):
    pytest.importorskip("openpyxl")
    folder = tmp_path / "docs"
    folder.mkdir()
    _make_xlsx(folder / "statement.xlsx", {"Q1": [["account", "balance"]]})

    config = make_config(tmp_path, str(folder), with_ocr=False, with_text_embed=False)
    registry = fake_registry(config)
    conn = connect(tmp_path / "test.db")
    migrate(conn)

    ingest_folder(conn, config, registry, str(folder))
    assert _counts(conn, "items")["items"] == 0
    assert conn.execute("SELECT 1 FROM files WHERE path = 'statement.xlsx'").fetchone()

    second = ingest_folder(conn, config, registry, str(folder))
    assert second["skipped"] == 1


def test_deleted_document_is_pruned(tmp_path):
    """Regression for prune_missing: content-addressed kinds beyond note
    (pdf/docx/csv/xlsx) must purge when their file disappears."""
    folder = tmp_path / "docs"
    folder.mkdir()
    doomed = folder / "old.csv"
    doomed.write_text("alpha,beta\n1,2\n")
    (folder / "keep.md").write_text("# keeper\nstays\n")

    config = make_config(tmp_path, str(folder), with_ocr=False, with_text_embed=False)
    registry = fake_registry(config)
    conn = connect(tmp_path / "test.db")
    migrate(conn)

    ingest_folder(conn, config, registry, str(folder))
    assert _counts(conn, "items")["items"] == 2

    doomed.unlink()
    stats = ingest_folder(conn, config, registry, str(folder))
    assert stats["pruned"] == 1
    kinds = {r["kind"] for r in conn.execute("SELECT kind FROM items")}
    assert kinds == {"note"}


def test_edited_csv_reindexes_and_prunes_old_content(tmp_path):
    import os

    folder = tmp_path / "docs"
    folder.mkdir()
    path = folder / "cols.csv"
    path.write_text("first_name,last_name\nann,lee\n")

    config = make_config(tmp_path, str(folder), with_ocr=False, with_text_embed=False)
    registry = fake_registry(config)
    conn = connect(tmp_path / "test.db")
    migrate(conn)
    ingest_folder(conn, config, registry, str(folder))

    path.write_text("email,phone\na@x.com,555\n")
    os.utime(path, (path.stat().st_atime, path.stat().st_mtime + 10))
    stats = ingest_folder(conn, config, registry, str(folder))
    assert stats["indexed"] == 1 and stats["pruned"] == 1

    bodies = [r["body"] for r in conn.execute("SELECT body FROM items")]
    assert len(bodies) == 1 and "email" in bodies[0] and "first_name" not in bodies[0]


def test_document_text_vectors_land_in_vec_table(tmp_path):
    pytest.importorskip("sqlite_vec", reason="requires sqlite-vec")
    folder = tmp_path / "docs"
    folder.mkdir()
    (folder / "cols.csv").write_text("project,owner,deadline\nx,y,z\n")

    config = make_config(tmp_path, str(folder), with_ocr=False, with_text_embed=True)
    registry = fake_registry(config)
    conn = connect(tmp_path / "test.db")
    migrate(conn)
    from image_search.store.db import load_vec_extension

    load_vec_extension(conn)

    ingest_folder(conn, config, registry, str(folder))
    n = conn.execute("SELECT COUNT(*) AS n FROM vec_map").fetchone()["n"]
    assert n == 1


def test_document_suffixes_are_phase_one(tmp_path):
    config = make_config(tmp_path, str(tmp_path), with_caption=True)
    folder = config.folders[str(tmp_path)]
    for name in ("a.pdf", "b.docx", "c.csv", "d.xlsx", "e.md", "f.links"):
        assert _ingest_phase(folder, Path(name)) == 1
    assert _ingest_phase(folder, Path("g.png")) == 0  # caption-pipeline image
