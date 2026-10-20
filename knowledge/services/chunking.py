"""Splits text into overlapping chunks that respect paragraph and sentence boundaries.

Each chunk records its character offsets in the source text, so a citation can be
traced back to the exact place in the original document.
"""

import re
from dataclasses import dataclass

_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


@dataclass(frozen=True)
class TextChunk:
    index: int
    text: str
    start_char: int
    end_char: int


def chunk_text(text: str, max_chars: int = 900, overlap: int = 150) -> list[TextChunk]:
    if max_chars <= 0 or not 0 <= overlap < max_chars:
        raise ValueError("Need max_chars > 0 and 0 <= overlap < max_chars.")
    spans = _units(text, max_chars)
    chunks: list[TextChunk] = []
    i = 0
    while i < len(spans):
        start = spans[i][0]
        j = i
        # Grow the chunk unit by unit while it fits.
        while j + 1 < len(spans) and spans[j + 1][1] - start <= max_chars:
            j += 1
        end = spans[j][1]
        chunks.append(TextChunk(len(chunks), text[start:end], start, end))
        if j + 1 >= len(spans):
            break
        # Step back over units that fall inside the overlap window, but always advance.
        next_i = j + 1
        while next_i - 1 > i and end - spans[next_i - 1][0] <= overlap:
            next_i -= 1
        i = next_i
    return chunks


def _units(text: str, max_chars: int) -> list[tuple[int, int]]:
    """Returns (start, end) spans of paragraphs, split into sentences or hard-wrapped when too long."""
    spans: list[tuple[int, int]] = []
    for para in re.finditer(r"\S(?:.*?\S)?(?=\n\s*\n|\s*\Z)", text, flags=re.DOTALL):
        if para.end() - para.start() <= max_chars:
            spans.append((para.start(), para.end()))
            continue
        cursor = para.start()
        for piece in _SENTENCE_END.split(para.group()):
            start = text.index(piece, cursor)
            end = start + len(piece)
            cursor = end
            # A single sentence longer than a chunk is cut into fixed-size pieces.
            for offset in range(start, end, max_chars):
                spans.append((offset, min(offset + max_chars, end)))
    return spans
