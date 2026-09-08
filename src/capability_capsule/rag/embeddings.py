"""Batch text embeddings through the local Ollama API"""

from collections.abc import Sequence
from math import isfinite

import httpx

from capability_capsule.config import Settings


def embed_texts(
    texts: Sequence[str],
    settings: Settings,
    *,
    transport: httpx.BaseTransport | None = None,
) -> tuple[tuple[float, ...], ...]:
    """Return one embedding per input text, preserving input order."""

    if not texts:
        return ()

    with httpx.Client(
        base_url=str(settings.ollama.base_url),
        trust_env=settings.http.trust_env,
        timeout=120.0,
        transport=transport,
    ) as client:
        response = client.post(
            "/api/embed",
            json={
                "model": settings.ollama.embedding_model,
                "input": list(texts),
                "truncate": False,
            },
        )
        response.raise_for_status()
        payload = response.json()

    if not isinstance(payload, dict):
        raise ValueError("Expected an embedding response object")

    raw_vectors = payload.get("embeddings")
    if not isinstance(raw_vectors, list):
        raise ValueError("Response must contain an embedding list")

    if len(raw_vectors) != len(texts):
        raise ValueError("Embedding count does not match input count")

    vectors: list[tuple[float, ...]] = []
    dimensions: int | None = None

    for raw_vector in raw_vectors:
        if not isinstance(raw_vector, list) or not raw_vector:
            raise ValueError("Each embedding must be a non-empty list")

        if dimensions is None:
            dimensions = len(raw_vector)

        elif len(raw_vector) != dimensions:
            raise ValueError("Embedding dimensions must be consistent")

        values: list[float] = []
        for value in raw_vector:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError("Embedding values must be numbers")

            number = float(value)
            if not isfinite(number):
                raise ValueError("Embedding values must be finite")

            values.append(number)
        vectors.append(tuple(values))

    return tuple(vectors)
