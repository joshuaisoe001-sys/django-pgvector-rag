import pytest

from knowledge.services.extraction import ExtractionError, extract_text, normalise
from tests.helpers import make_pdf


def test_text_with_utf8_bom_and_windows_line_endings():
    result = extract_text("notes.txt", "﻿Line one\r\nLine two\r\n".encode())
    assert result.text == "Line one\nLine two"
    assert result.content_type == "text/plain"


def test_utf16_needs_a_byte_order_mark():
    assert extract_text("notes.txt", "Habari".encode("utf-16")).text == "Habari"


def test_invalid_utf8_is_rejected():
    with pytest.raises(ExtractionError, match="UTF-8"):
        extract_text("notes.txt", b"\x80\x81 not text")


def test_markdown_is_read_as_text():
    result = extract_text("README.MD", b"# Title\n\nBody")
    assert (result.text, result.content_type) == ("# Title\n\nBody", "text/markdown")


def test_pdf_text_is_extracted():
    result = extract_text("policy.pdf", make_pdf("Leave is 21 working days per year."))
    assert "21 working days" in result.text
    assert result.content_type == "application/pdf"


def test_corrupt_pdf_is_rejected():
    with pytest.raises(ExtractionError, match="PDF"):
        extract_text("broken.pdf", b"%PDF-1.4 this is not really a pdf")


@pytest.mark.parametrize("filename", ["photo.png", "script.exe", "no_extension"])
def test_unsupported_types_are_rejected(filename):
    with pytest.raises(ExtractionError, match="Unsupported file type"):
        extract_text(filename, b"data")


def test_normalise_collapses_blank_lines_and_strips_nuls():
    assert normalise("a  \n\n\n\nb\x00\n\n") == "a\n\nb"
