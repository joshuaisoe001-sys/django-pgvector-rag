from django.core.files.uploadedfile import SimpleUploadedFile

LEAVE_POLICY = """Annual leave

Every full-time employee is entitled to 21 working days of paid annual leave per year. Leave must be requested through the HR portal at least two weeks in advance.

Unused leave of up to five days may be carried over into the first quarter of the next year. Any remaining days are forfeited."""

TRAVEL_POLICY = """Travel and expenses

Staff travelling to Mombasa or Kisumu for work are reimbursed for economy flights or standard rail tickets. Hotel stays are capped at KES 12,000 per night.

Expense claims must be submitted with receipts within 30 days of returning from the trip."""

IT_POLICY = """Laptops and passwords

Company laptops must use full-disk encryption. Passwords must be at least 14 characters long and must never be shared, including with the IT helpdesk.

Report a lost or stolen laptop to the IT security team within 24 hours."""


def text_file(name: str, text: str) -> SimpleUploadedFile:
    return SimpleUploadedFile(name, text.encode("utf-8"), content_type="text/plain")


def make_pdf(text: str) -> bytes:
    """Builds a minimal one-page PDF with the given single line of text."""
    escaped = text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
    stream = f"BT /F1 12 Tf 72 720 Td ({escaped}) Tj ET".encode("latin-1")
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>",
        b"<< /Length " + str(len(stream)).encode() + b" >>\nstream\n" + stream + b"\nendstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for number, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += f"{number} 0 obj\n".encode() + body + b"\nendobj\n"
    xref = len(out)
    out += f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode()
    out += b"".join(f"{offset:010d} 00000 n \n".encode() for offset in offsets)
    out += f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode()
    return bytes(out)
