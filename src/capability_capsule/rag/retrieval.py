"""Retrieve relevent text chunks from a saved vector index."""

from pathlib import Path
import httpx

from capability_capsule.config import Settings
from capability_capsule.rag.embeddings import embed_texts
from capability_capsule.rag.index import SearchResult
from capability_capsule.rag.storage import load_index

def retrieve(
        question: str,
        index_path: Path,
        settings: Settings,
        *,
        top_k: int = 5,
        transport: httpx.BaseTransport | None = None,
) -> tuple[SearchResult, ...]:
    """Embed a question a retrieve the most similar stored chunks."""

    if not question.strip():
        raise ValueError("Question must not be blank")

    if top_k <= 0:
        raise ValueError("top_k must be positive")

    index = load_index(
        index_path,
        expected_embedding_model= settings.ollama.embedding_model,
    )

    query_vectors = embed_texts(
        [question],
        settings,
        transport=transport
    )

    return index.search(
        query_vectors[0],
        top_k = top_k
    )