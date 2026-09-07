"""Save and load vector indexs with versioned metadata"""

from collections.abc import Sequence
from pathlib import Path
from typing import Literal

import numpy as np
from numpy.lib.npyio import NpzFile
from pydantic import BaseModel, ConfigDict, Field

from capability_capsule.rag.chunker import TextChunk
from capability_capsule.rag.index import VectorIndex


class IndexMetadata(BaseModel):
    """Text chunks and model information stored alongside vectors"""

    model_config = ConfigDict(extra = "forbid", frozen = True)

    schema_version: Literal["0.1"] = "0.1"
    embedding_model: str = Field(min_length = 1)
    chunks: tuple[TextChunk, ...]

def save_index(
        path: Path,
        chunks: Sequence[TextChunk],
        vectors: Sequence[Sequence[float]],
        *,
        embedding_model: str
) -> None:
    """Validate and save an index without overwriting an existing file."""

    metadata = IndexMetadata(
        embedding_model = embedding_model,
        chunks = tuple(chunks),
    )

    matrix = np.array(vectors, dtype=np.float64, copy = True)

    VectorIndex(metadata.chunks, matrix.tolist())
    metadata_array = np.array(metadata.model_dump_json())

    with path.open("xb") as stream:
        np.savez_compressed(
            stream,
            metadata = metadata_array,
            vectors = matrix,
            allow_pickle = False,
        )

def load_index(
        path: Path,
        *,
        expected_embedding_model: str
) -> VectorIndex:
    """Load an index and verify its embedding model and vector data"""

    archive = np.load(path, allow_pickle = False)

    if not isinstance(archive, NpzFile):
        raise ValueError("Expected an NPZ index archive")

    with archive:
        if set(archive.files) != {"metadata", "vectors"}:
            raise ValueError("Index archive has unexpected contents")

        metadata_array = archive["metadata"]

        if metadata_array.ndim != 0 or metadata_array.dtype.kind != "U":
            raise ValueError("Metadata must be a scalar Unicode string")

        metadata = IndexMetadata.model_validate_json(
            str(metadata_array.item())
        )

        if metadata.embedding_model != expected_embedding_model:
            raise ValueError("Embedding model does not match the saved index")

        matrix = archive["vectors"]

        if matrix.ndim != 2 or matrix.dtype.kind != "f":
            raise ValueError("Vector must be a floating-point matrix")

        return VectorIndex(metadata.chunks, matrix.tolist())

    
