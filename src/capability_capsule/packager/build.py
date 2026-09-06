"""Build a searchable index from repository text files"""

from pathlib import Path

import httpx
from pydantic import BaseModel, ConfigDict, Field

from capability_capsule.config import Settings
from capability_capsule.rag.chunker import chunk_document
from capability_capsule.rag.embeddings import embed_texts
from capability_capsule.rag.storage import save_index
from capability_capsule.scanner.documents import read_repository_documents
from capability_capsule.scanner.repo import scan_repository

class IndexBuildResult(BaseModel):
    """Summary of a completed index build"""

    model_config = ConfigDict(extra = "forbid", frozen = True)

    output_path : Path
    document_count: int = Field(gt = 0)
    chunk_count: int = Field(gt = 0)
    vector_dimensions: int = Field(gt = 0)
    size_bytes: int = Field(gt = 0)

def build_index(
        root: Path,
        output_path: Path,
        settings: Settings,
        *,
        chunk_size_chars: int = 1_000,
        overlap_chars: int = 200,
        batch_size: int = 32,
        max_file_size_bytes: int = 1_000_000,
        transport: httpx.BaseTransport | None = None,
) -> IndexBuildResult:
    """Scan, chunk, embed, and save repository text as an index."""

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    if chunk_size_chars <= 0:
        raise ValueError("chunk_size_chars must be positive")

    if overlap_chars < 0 or overlap_chars >= chunk_size_chars:
        raise ValueError("overlap_chars must be smaller than chunk_size_chars")

    if max_file_size_bytes <= 0:
        raise ValueError("max_file_size_bytes must be positive")

    if output_path.exists() or output_path.is_symlink():
        raise FileExistsError(output_path)

    destination = output_path.resolve()

    if not destination.parent.is_dir():
        raise NotADirectoryError(destination.parent)

    files = scan_repository(root)
    documents = read_repository_documents(
        root,
        files,
        max_file_size_bytes=max_file_size_bytes,
    )

    # Empty documents cannot contribute any chunks.
    nonempty_documents = tuple(
        document for document in documents if document.text
    )

    chunks = tuple(
        chunk for document in nonempty_documents
        for chunk in chunk_document(
            document,
            chunk_size_chars= chunk_size_chars,
            overlap_chars= overlap_chars,
        )
    )

    if not chunks:
        raise ValueError("No text chunks are available to index")

    vectors: list[tuple[float, ...]] = []
    dimensions: int | None = None

    for start in range(0, len(chunks), batch_size):
        batch = chunks[start : start + batch_size]
        batch_vectors = embed_texts(
            [chunk.text for chunk in batch],
            settings,
            transport = transport,
        )

        batch_dimensions = len(batch_vectors[0])

        if dimensions is None:
            dimensions = batch_dimensions
        elif dimensions != batch_dimensions:
            raise ValueError("Embedding dimensions changed between batches")

        vectors.extend(batch_vectors)

    save_index(
        destination,
        chunks,
        vectors,
        embedding_model = settings.ollama.embedding_model,
    )

    return IndexBuildResult(
        output_path = destination,
        document_count= len(nonempty_documents),
        chunk_count=len(chunks),
        vector_dimensions=len(vectors[0]),
        size_bytes= destination.stat().st_size
    )