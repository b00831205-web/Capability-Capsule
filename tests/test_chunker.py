import pytest

from capability_capsule.manifest import SourceType
from capability_capsule.rag.chunker import chunk_document
from capability_capsule.scanner.documents import RepositoryDocument


def _document(text: str) -> RepositoryDocument:
    return RepositoryDocument(
        relative_path="notes/design.md",
        text=text,
        size_bytes=len(text.encode("utf-8")),
    )


def test_chunks_preserve_text_offsets_and_source() -> None:
    document = _document("abcdefghijk")

    chunks = chunk_document(document, chunk_size_chars=5, overlap_chars=2)

    assert [chunk.text for chunk in chunks] == ["abcde", "defgh", "ghijk"]
    assert [(chunk.start_char, chunk.end_char) for chunk in chunks] == [
        (0, 5), (3, 8), (6, 11),
    ]
    assert [chunk.chunk_index for chunk in chunks] == [0, 1, 2]
    for chunk in chunks:
        assert chunk.relative_path == document.relative_path
        assert chunk.source_type is SourceType.REPO
        assert chunk.text == document.text[chunk.start_char:chunk.end_char]
    assert chunks == chunk_document(document, chunk_size_chars=5, overlap_chars=2)


def test_zero_overlap_preserves_entire_text_and_short_tail() -> None:
    document = _document("abcdefghijk")

    chunks = chunk_document(document, chunk_size_chars=4, overlap_chars=0)

    assert [chunk.text for chunk in chunks] == ["abcd", "efgh", "ijk"]
    assert "".join(chunk.text for chunk in chunks) == document.text


@pytest.mark.parametrize("text", ["abc", "abcde"])
def test_short_or_exact_size_document_produces_one_chunk(text: str) -> None:
    chunks = chunk_document(_document(text), chunk_size_chars=5, overlap_chars=2)

    assert len(chunks) == 1
    assert chunks[0].text == text
    assert chunks[0].start_char == 0
    assert chunks[0].end_char == len(text)


def test_empty_document_produces_no_chunks() -> None:
    assert chunk_document(_document("")) == ()


def test_unicode_and_whitespace_use_character_offsets() -> None:
    document = _document("甲乙\n🙂 丙")

    chunks = chunk_document(document, chunk_size_chars=3, overlap_chars=1)

    assert [chunk.text for chunk in chunks] == ["甲乙\n", "\n🙂 ", " 丙"]
    assert [(chunk.start_char, chunk.end_char) for chunk in chunks] == [
        (0, 3), (2, 5), (4, 6),
    ]


@pytest.mark.parametrize(
    ("chunk_size_chars", "overlap_chars"),
    [
        (0, 0),
        (-1, 0),
        (5, -1),
        (5, 5),
        (5, 6),
    ],
)
def test_invalid_parameters_fail_even_for_empty_document(
    chunk_size_chars: int, overlap_chars: int,
) -> None:
    with pytest.raises(ValueError):
        chunk_document(
            _document(""),
            chunk_size_chars=chunk_size_chars,
            overlap_chars=overlap_chars,
        )
