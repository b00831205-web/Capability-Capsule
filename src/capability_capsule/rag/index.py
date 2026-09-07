"""In-memory vector retrieval using cosine similarity."""

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict

from capability_capsule.rag.chunker import TextChunk


class SearchResult(BaseModel):
    """A retrieved chunk and its cosine similarity score."""

    model_config = ConfigDict(extra = "forbid", frozen = True)

    chunk: TextChunk
    score: float

def _normalize_rows(matrix: NDArray[np.float64]) -> NDArray[np.float64]:
    """Normalize non-zero rows without overflowing on large values"""

    if not np.isfinite(matrix).all():
        raise ValueError("Vector must contain only finite numbers")

    scales = np.max(np.abs(matrix), axis = 1, keepdims = True)
    if np.any(scales == 0):
        raise ValueError("Zero vectors cannot be used for cosine similarity")

    scaled = matrix / scales
    lengths = np.sqrt(np.sum(scaled * scaled, axis = 1, keepdims = True))
    normalized: NDArray[np.float64] = scaled / lengths
    return normalized

class VectorIndex:
    """Store chunks and search their normalized embedding vectors"""

    def __init__(self,
                 chunks: Sequence[TextChunk],
                 vectors: Sequence[Sequence[float]]
                 ) -> None:
        if not chunks:
            raise ValueError("An index requires ata= least one chunk")

        matrix = np.array(vectors, dtype = np.float64, copy = True)

        if matrix.ndim != 2:
            raise ValueError("Vector must form a two-dimensional matrix")

        if matrix.shape[0] != len(chunks):
            raise ValueError("Vector count must match chunk count")

        if matrix.shape[1] == 0:
            raise ValueError("Vectors must have at least one dimention")

        self._chunks = tuple(chunks)
        self._vectors = _normalize_rows(matrix)
        self._dimensions = matrix.shape[1]

    def search(
            self,
            query_vector: Sequence[float],
            *,
            top_k: int = 5
    )->tuple[SearchResult, ...]:
        """Return up to top_k chunks, ordered by descending similarity"""

        if top_k <= 0:
            raise ValueError("top_k must be positive")

        query = np.array(query_vector, dtype = np.float64, copy = True)

        if query.ndim !=1 or query.size != self._dimensions:
            raise ValueError("Query dimensions must match index dimensions")

        normalized_query = _normalize_rows(query.reshape(1, -1))[0]
        scores = self._vectors @ normalized_query

        scores = np.clip(scores, -1.0, 1.0)

        positions = np.argsort(-scores, kind="stable")[:top_k]

        return tuple(SearchResult(
            chunk = self._chunks[int(position)],
            score = float(scores[position])
        ) for position in positions)
    