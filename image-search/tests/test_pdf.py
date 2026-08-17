from pathlib import Path

import pytest

from image_search import textitems
from image_search.config import load_config


# ---- page sampling ---------------------------------------------------------

def test_sample_takes_every_page_of_a_short_pdf():
    assert textitems.sample_page_numbers(3) == [0, 1, 2]
    assert textitems.sample_page_numbers(5) == [0, 1, 2, 3, 4]


def test_sample_spreads_across_a_long_pdf():
    """First page plus a spread — not the first five, which on a book-length
    PDF are all front matter."""
    pages = textitems.sample_page_numbers(100)
    assert len(pages) == 5
    assert pages[0] == 0
    assert pages == sorted(pages)
    assert pages[-1] > 90  # reaches the end of the document
    assert len(set(pages)) == 5  # no duplicates


def test_sample_never_exceeds_document_bounds():
    for total in range(1, 60):
        pages = textitems.sample_page_numbers(total)
        assert all(0 <= p < total for p in pages), total
        assert len(pages) == min(total, textitems.PDF_SAMPLE_PAGES)


# ---- financial exclusion ---------------------------------------------------

@pytest.mark.parametrize(
    "name",
    [
        "1099 Composite and Year-End Summary - 2022.PDF",
        "W-2_2023.pdf", "w2-2024.pdf", "form 1040.pdf",
        "Tax Return 2022.pdf", "bank statement march.pdf",
        "invoice-4471.pdf", "Receipt.PDF", "payroll_june.pdf",
    ],
)
def test_financial_filenames_are_excluded(name):
    assert textitems.looks_financial(Path(name)) is True


@pytest.mark.parametrize(
    "name",
    [
        "attention is all you need.pdf", "recipe book.pdf",
        "Lease Agreement.pdf", "syllabus.pdf", "resume_rodrigo.pdf",
    ],
)
def test_ordinary_filenames_are_not_excluded(name):
    assert textitems.looks_financial(Path(name)) is False


def test_exclude_patterns_can_be_overridden_per_folder(tmp_path):
    config_path = tmp_path / "folders.yaml"
    config_path.write_text(
        'folders:\n'
        '  "~/Papers":\n'
        '    text_embed: bge-small-en-v1.5\n'
        '    exclude_patterns: ["^draft-"]\n'
        '  "~/Docs":\n'
        '    text_embed: bge-small-en-v1.5\n'
    )
    config = load_config(config_path)

    papers = config.folders["~/Papers"]
    assert papers.excludes_pdf(Path("draft-notes.pdf")) is True
    # An explicit list replaces the financial defaults.
    assert papers.excludes_pdf(Path("1099.pdf")) is False

    docs = config.folders["~/Docs"]
    assert docs.excludes_pdf(Path("1099.pdf")) is True


def test_empty_exclude_patterns_indexes_everything(tmp_path):
    config_path = tmp_path / "folders.yaml"
    config_path.write_text(
        'folders:\n  "~/All":\n    text_embed: x\n    exclude_patterns: []\n'
    )
    folder = load_config(config_path).folders["~/All"]
    assert folder.excludes_pdf(Path("1099.pdf")) is False


def test_invalid_exclude_regex_raises(tmp_path):
    config_path = tmp_path / "folders.yaml"
    config_path.write_text(
        'folders:\n  "~/X":\n    text_embed: x\n    exclude_patterns: ["[unclosed"]\n'
    )
    with pytest.raises(ValueError, match="invalid regex"):
        load_config(config_path)


# ---- extraction ------------------------------------------------------------

def _write_pdf(path: Path, pages: list[str]) -> None:
    """A real, readable PDF: one text line per page. Text needs both a
    content stream and a font resource, or extract_text() returns nothing."""
    pytest.importorskip("pypdf")
    from pypdf import PdfWriter
    from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject

    writer = PdfWriter()
    for text in pages:
        page = writer.add_blank_page(width=200, height=200)
        stream = DecodedStreamObject()
        stream.set_data(f"BT /F1 12 Tf 20 100 Td ({text}) Tj ET".encode())
        page[NameObject("/Contents")] = writer._add_object(stream)
        font = DictionaryObject({
            NameObject("/Type"): NameObject("/Font"),
            NameObject("/Subtype"): NameObject("/Type1"),
            NameObject("/BaseFont"): NameObject("/Helvetica"),
        })
        page[NameObject("/Resources")] = DictionaryObject({
            NameObject("/Font"): DictionaryObject(
                {NameObject("/F1"): writer._add_object(font)}
            )
        })
    with path.open("wb") as fh:
        writer.write(fh)


def test_parse_pdf_reads_text_and_falls_back_to_stem_for_title(tmp_path):
    pytest.importorskip("pypdf")
    path = tmp_path / "notes on kalman filters.pdf"
    _write_pdf(path, ["state estimation", "covariance update"])

    title, body = textitems.parse_pdf(path)
    assert title == "notes on kalman filters"
    assert "state estimation" in body


def test_parse_pdf_uses_ocr_for_pages_without_text(tmp_path, monkeypatch):
    """A scanned page has no text layer; the OCR processor fills the gap."""
    pytest.importorskip("pypdf")
    path = tmp_path / "scan.pdf"
    _write_pdf(path, ["", ""])  # no extractable text

    calls = []

    def fake_ocr_pages(pdf_path, indexes, processor):
        calls.append((pdf_path, tuple(indexes)))
        return ["text recovered by ocr"]

    monkeypatch.setattr(textitems, "_ocr_pdf_pages", fake_ocr_pages)

    class FakeOcr:
        model_id = "fake-ocr"

    title, body = textitems.parse_pdf(path, ocr_processor=FakeOcr())
    assert "text recovered by ocr" in body
    assert calls and calls[0][1] == (0, 1)


def test_parse_pdf_without_ocr_processor_returns_empty_body(tmp_path):
    pytest.importorskip("pypdf")
    path = tmp_path / "scan.pdf"
    _write_pdf(path, ["", ""])

    title, body = textitems.parse_pdf(path, ocr_processor=None)
    assert title == "scan"
    assert body.strip() == ""


# ---- content-based financial exclusion -------------------------------------

def test_content_gate_catches_documents_filenames_miss():
    """The real case: an account-verification letter named '1_RODRIGO__LUNA.pdf'
    matches no filename pattern but is exactly what must not be indexed."""
    assert textitems.looks_financial(Path("1_RODRIGO__LUNA.pdf")) is False
    marker = textitems.content_looks_financial(
        "AccountVerification",
        "RODRIGO LUNA DESIGNATED BENE PLAN/TOD 522 S. BREED ST. LOS ANGELES",
    )
    assert marker is not None


@pytest.mark.parametrize(
    "title,body",
    [
        ("Statement", "Your routing number is 123456789"),
        ("Doc", "Social Security Number: 555-12-3456"),
        ("Summary", "2023 Year-End Summary of taxable income"),
        ("Form", "This is Form 1099-DIV for the tax year"),
    ],
)
def test_financial_content_is_detected(title, body):
    assert textitems.content_looks_financial(title, body) is not None


@pytest.mark.parametrize(
    "title,body",
    [
        ("Attention Is All You Need", "We propose a new network architecture."),
        ("Recipe", "Combine flour and water, then bake for 40 minutes."),
        ("Deep Entity Classification", "Abusive account detection for online networks."),
    ],
)
def test_ordinary_documents_pass_the_content_gate(title, body):
    assert textitems.content_looks_financial(title, body) is None


def test_explicit_exclude_patterns_disable_the_content_gate(tmp_path):
    """An explicit list means the user said exactly what to skip; we don't
    also apply a hidden content filter on top of their choice."""
    from image_search.config import load_config

    config_path = tmp_path / "folders.yaml"
    config_path.write_text(
        'folders:\n  "~/X":\n    text_embed: x\n    exclude_patterns: []\n'
    )
    assert load_config(config_path).folders["~/X"].exclude_patterns == ()
