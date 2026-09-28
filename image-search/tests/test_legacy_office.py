"""Legacy Word/PowerPoint (.doc/.ppt) and .pptx extraction. The binary
formats are exercised through synthetic streams built per [MS-DOC] and
[MS-PPT], so no real documents live in the repo."""

import struct
from pathlib import Path

import pytest

from image_search import textitems


def _word97_stream(pieces, ccp_text):
    """A minimal Word 97 WordDocument stream plus its 1Table stream.
    `pieces`: [(text, compressed)] laid out one after another."""
    csw, cslw, fc_pairs = 14, 22, 93
    rglw = 34 + csw * 2 + 2  # FibBase + csw + fibRgW + cslw
    blob = rglw + cslw * 4 + 2
    fib = bytearray(blob + fc_pairs * 8)
    struct.pack_into("<HH", fib, 0, 0xA5EC, 0x00C1)
    struct.pack_into("<H", fib, 0x0A, 0x0200)  # fWhichTblStm -> 1Table
    struct.pack_into("<H", fib, 32, csw)
    struct.pack_into("<H", fib, 34 + csw * 2, cslw)
    struct.pack_into("<I", fib, rglw + 12, ccp_text)
    struct.pack_into("<H", fib, rglw + cslw * 4, fc_pairs)

    word = bytearray(fib)
    cps, pcds, cp = [0], [], 0
    for text, compressed in pieces:
        fc = len(word)
        if compressed:
            word += text.encode("cp1252")
            fc_raw = (fc * 2) | 0x40000000
        else:
            word += text.encode("utf-16-le")
            fc_raw = fc
        cp += len(text)
        cps.append(cp)
        pcds.append(struct.pack("<HIH", 0, fc_raw, 0))
    plc = struct.pack(f"<{len(cps)}I", *cps) + b"".join(pcds)
    table = b"\x02" + struct.pack("<I", len(plc)) + plc
    struct.pack_into("<II", word, blob + 33 * 8, 0, len(table))
    return bytes(word), {"1Table": table}


def test_word97_piece_table_mixes_8bit_and_utf16_pieces():
    word, streams = _word97_stream([("Hola año ", True), ("señor ƒ\r", False)], ccp_text=17)
    text = textitems._word97_text(word, streams.__getitem__)
    assert text == "Hola año señor ƒ\r"


def test_word97_stops_at_the_main_document():
    # Pieces past ccpText belong to footnotes/headers.
    word, streams = _word97_stream([("body\r", True), ("footnote\r", True)], ccp_text=5)
    assert textitems._word97_text(word, streams.__getitem__) == "body\r"


def test_word95_text_is_contiguous():
    word = bytearray(0x40)
    struct.pack_into("<HH", word, 0, 0xA5DC, 104)
    body = "Canción de 1996\r".encode("cp1252")
    struct.pack_into("<II", word, 0x18, len(word), len(word) + len(body))
    word += body
    assert textitems._word97_text(bytes(word), {}.__getitem__) == "Canción de 1996\r"


def test_encrypted_word_document_raises():
    word = bytearray(0x40)
    struct.pack_into("<HH", word, 0, 0xA5EC, 0x00C1)
    struct.pack_into("<H", word, 0x0A, 0x0100)
    with pytest.raises(ValueError, match="encrypted"):
        textitems._word97_text(bytes(word), {}.__getitem__)


def test_clean_office_text_drops_field_codes_and_control_marks():
    raw = 'See \x13 HYPERLINK "http://x" \x14the site\x15 now\rCell\x07next\x0cpage\x01'
    assert textitems._clean_office_text(raw) == "See the site now\nCell\nnext\npage"


def _record(rtype, body, container=False):
    return struct.pack("<HHI", 0x000F if container else 0x0000, rtype, len(body)) + body


def test_ppt_text_atoms_in_nested_containers_keep_order():
    slide = _record(0x03F8, _record(0x0FA0, "Título".encode("utf-16-le"))
                    + _record(0x0FA8, "second line".encode("latin-1")), container=True)
    master = _record(0x0FA8, b"Click to edit Master title style")
    stream = _record(0x03E8, master + slide, container=True) + _record(0x0FA8, b"after")
    assert textitems._ppt_text(stream) == ["Título", "second line", "after"]


def test_parse_pptx_reads_slides(tmp_path):
    pptx = pytest.importorskip("pptx")
    prs = pptx.Presentation()
    slide = prs.slides.add_slide(prs.slide_layouts[1])
    slide.shapes.title.text = "Quarterly Review"
    slide.placeholders[1].text = "Revenue grew in Q3"
    path = tmp_path / "deck.pptx"
    prs.save(str(path))
    title, body = textitems.parse_pptx(path)
    assert title == "Quarterly Review"
    assert "Revenue grew in Q3" in body


def test_doc_that_is_really_rtf_or_text(tmp_path):
    pytest.importorskip("olefile")
    rtf = tmp_path / "carta.doc"
    rtf.write_bytes(rb"{\rtf1\ansi {\b Querida} Mar\'eda,\par nos vemos.}")
    title, body = textitems.parse_doc(rtf)
    assert title == "carta"
    assert "Querida María," in body and "nos vemos." in body
    plain = tmp_path / "readme.doc"
    plain.write_bytes("Instalación rápida".encode("cp1252"))
    assert textitems.parse_doc(plain)[1] == "Instalación rápida"


@pytest.mark.parametrize("name", ["a.doc", "b.ppt", "c.pptx"])
def test_legacy_office_files_are_walked_as_documents(tmp_path, name):
    from image_search.store import images as images_store

    (tmp_path / name).write_bytes(b"x")
    assert [p.name for p, _ in images_store.walk_candidates(tmp_path)] == [name]
    assert Path(name).suffix in textitems.DOCUMENT_KINDS
