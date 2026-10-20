"""Turns an uploaded file into plain text. Supports .txt, .md and text-based PDFs."""

import io
from dataclasses import dataclass
from pathlib import PurePath

from pypdf import PdfReader
from pypdf.errors import PdfReadError

TEXT_TYPES = {".txt": "text/plain", ".md": "text/markdown"}
PDF_TYPE = "application/pdf"
SUPPORTED_EXTENSIONS = sorted([*TEXT_TYPES, ".pdf"])


class ExtractionError(ValueError):
    """The file cannot be read. The message is safe to show to the user."""


@dataclass(frozen=True)
class Extracted:
    text: str
    content_type: str


def extract_text(filename: str, data: bytes) -> Extracted:
    extension = PurePath(filename).suffix.lower()
    if extension in TEXT_TYPES:
        return Extracted(_decode(data), TEXT_TYPES[extension])
    if extension == ".pdf":
        return Extracted(_pdf_text(data), PDF_TYPE)
    raise ExtractionError(f"Unsupported file type '{extension or 'none'}'. Upload one of: {', '.join(SUPPORTED_EXTENSIONS)}.")


def _decode(data: bytes) -> str:
    # UTF-16 only with a byte-order mark: almost any even-length bytes "decode" as UTF-16.
    encoding = "utf-16" if data.startswith((b"\xff\xfe", b"\xfe\xff")) else "utf-8-sig"
    try:
        return normalise(data.decode(encoding))
    except UnicodeDecodeError as exc:
        raise ExtractionError("The file is not valid UTF-8 text.") from exc


def _pdf_text(data: bytes) -> str:
    try:
        reader = PdfReader(io.BytesIO(data))
        if reader.is_encrypted:
            raise ExtractionError("The PDF is password-protected.")
        pages = [page.extract_text() or "" for page in reader.pages]
    except PdfReadError as exc:
        raise ExtractionError("The PDF could not be read.") from exc
    return normalise("\n\n".join(pages))


def normalise(text: str) -> str:
    """Unifies line endings, strips NULs and trailing spaces, and collapses runs of blank lines."""
    lines = [line.rstrip() for line in text.replace("\r\n", "\n").replace("\r", "\n").replace("\x00", "").split("\n")]
    out: list[str] = []
    for line in lines:
        if line or (out and out[-1]):
            out.append(line)
    return "\n".join(out).strip()
