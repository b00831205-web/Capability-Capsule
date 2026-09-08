"""Deterministic character-based text chunking"""

from pydantic import BaseModel, ConfigDict, Field

from capability_capsule.manifest import SourceType
from capability_capsule.scanner.documents import RepositoryDocument


class TextChunk(BaseModel):
    """A text slice with its source and character offsets"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    relative_path: str = Field(min_length=1)
    source_type: SourceType
    chunk_index: int = Field(ge=0)
    start_char: int = Field(ge=0)
    end_char: int = Field(gt=0)
    text: str = Field(min_length=1)


def chunk_document(
    document: RepositoryDocument, *, chunk_size_chars: int = 1_000, overlap_chars: int = 200
) -> tuple[TextChunk, ...]:
    """Split text into overlapping chunks, preserving source offsets"""

    if chunk_size_chars <= 0:
        raise ValueError("chunk_size_chars must be positive")

    if overlap_chars < 0 or overlap_chars >= chunk_size_chars:
        raise ValueError("overlap_chars must be non-negative and smaller than chunk-size-chars")
    chunks: list[TextChunk] = []
    text_length = len(document.text)
    start = 0
    while start < text_length:
        end = min(start + chunk_size_chars, text_length)
        chunks.append(
            TextChunk(
                relative_path=document.relative_path,
                source_type=document.source_type,
                chunk_index=len(chunks),
                start_char=start,
                end_char=end,
                text=document.text[start:end],
            )
        )

        if end == text_length:
            break

        start = end - overlap_chars

    return tuple(chunks)
