import pytest

from knowledge.services.chunking import chunk_text
from tests.helpers import LEAVE_POLICY


def test_short_text_is_one_chunk_with_exact_offsets():
    [chunk] = chunk_text("One paragraph only.", max_chars=100, overlap=10)
    assert (chunk.index, chunk.text, chunk.start_char, chunk.end_char) == (0, "One paragraph only.", 0, 19)


def test_every_chunk_is_an_exact_slice_of_the_source_and_within_the_limit():
    text = "\n\n".join(f"Paragraph {n}. " + "Some words here. " * 6 for n in range(30))
    chunks = chunk_text(text, max_chars=300, overlap=60)
    assert len(chunks) > 5
    for c in chunks:
        assert text[c.start_char : c.end_char] == c.text
        assert len(c.text) <= 300
    assert [c.index for c in chunks] == list(range(len(chunks)))


def test_chunks_cover_all_paragraphs_in_order():
    paragraphs = [f"Item {n} has a distinct marker M{n}." for n in range(40)]
    chunks = chunk_text("\n\n".join(paragraphs), max_chars=200, overlap=0)
    joined = " ".join(c.text for c in chunks)
    positions = [joined.index(f"M{n}.") for n in range(40)]
    assert positions == sorted(positions)
    starts = [c.start_char for c in chunks]
    assert starts == sorted(set(starts))


def test_consecutive_chunks_overlap_when_paragraphs_fit_in_the_window():
    text = "\n\n".join(f"Short paragraph number {n}." for n in range(20))
    chunks = chunk_text(text, max_chars=120, overlap=60)
    assert all(b.start_char < a.end_char for a, b in zip(chunks, chunks[1:], strict=False))


def test_large_overlap_still_makes_progress():
    text = "\n\n".join(f"P{n} " + "x" * 40 for n in range(10))
    chunks = chunk_text(text, max_chars=100, overlap=99)
    assert chunks[-1].end_char == len(text)
    assert len(chunks) < 20


def test_long_paragraph_is_split_at_sentences_and_overlong_sentences_are_cut():
    sentence = "This sentence is about leave. "
    text = sentence * 20 + "A" * 500
    chunks = chunk_text(text, max_chars=200, overlap=0)
    assert all(len(c.text) <= 200 for c in chunks)
    assert chunks[0].text.endswith("leave.")
    assert "".join(c.text for c in chunks if set(c.text) == {"A"}) == "A" * 500


def test_real_policy_text_keeps_paragraphs_whole():
    chunks = chunk_text(LEAVE_POLICY, max_chars=900, overlap=150)
    assert len(chunks) == 1
    assert "21 working days" in chunks[0].text


@pytest.mark.parametrize("max_chars, overlap", [(0, 0), (100, 100), (100, -1)])
def test_invalid_sizes_are_rejected(max_chars, overlap):
    with pytest.raises(ValueError):
        chunk_text("text", max_chars=max_chars, overlap=overlap)
