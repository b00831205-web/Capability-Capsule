import json
from typing import Any

import httpx
import pytest

from capability_capsule.config import Settings
from capability_capsule.rag.embeddings import embed_texts


def test_batch_request_preserves_order_and_uses_config() -> None:
    settings = Settings()
    settings.ollama.embedding_model = "test-embedding"
    requests: list[httpx.Request] = []

    def handle(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        assert request.method == "POST"
        assert request.url.path == "/api/embed"
        assert str(request.url).startswith(str(settings.ollama.base_url))
        payload = json.loads(request.content)
        assert payload["model"] == "test-embedding"
        assert payload["input"] == ["first", "second"]
        assert payload["truncate"] is False
        return httpx.Response(200, json={"embeddings": [[1, 0.5], [-1, 2]]})

    vectors = embed_texts(
        ["first", "second"],
        settings,
        transport=httpx.MockTransport(handle),
    )

    assert len(requests) == 1
    assert vectors == ((1.0, 0.5), (-1.0, 2.0))


def test_empty_input_makes_no_request() -> None:
    def handle(request: httpx.Request) -> httpx.Response:
        pytest.fail("Empty input must not make an HTTP request")

    assert embed_texts([], Settings(), transport=httpx.MockTransport(handle)) == ()


@pytest.mark.parametrize("status", [400, 404, 500])
def test_http_errors_are_propagated(status: int) -> None:
    transport = httpx.MockTransport(
        lambda request: httpx.Response(status, json={"error": "service error"})
    )
    with pytest.raises(httpx.HTTPStatusError):
        embed_texts(["text"], Settings(), transport=transport)


def test_timeout_is_propagated() -> None:
    def handle(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("timeout", request=request)

    with pytest.raises(httpx.ReadTimeout):
        embed_texts(["text"], Settings(), transport=httpx.MockTransport(handle))


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"embeddings": []},
        {"embeddings": [[1.0]]},
        {"embeddings": [[], []]},
        {"embeddings": [[1.0], [1.0, 2.0]]},
        {"embeddings": [["bad"], [1.0]]},
        {"embeddings": [[True], [1.0]]},
    ],
)
def test_invalid_vectors_are_rejected(payload: dict[str, Any]) -> None:
    transport = httpx.MockTransport(lambda request: httpx.Response(200, json=payload))
    with pytest.raises(ValueError):
        embed_texts(["first", "second"], Settings(), transport=transport)


@pytest.mark.parametrize("trust_env", [False, True])
def test_proxy_setting_is_passed_to_http_client(
    monkeypatch: pytest.MonkeyPatch,
    trust_env: bool,
) -> None:
    settings = Settings()
    assert settings.http.trust_env is False
    settings.http.trust_env = trust_env
    original_client = httpx.Client
    captured: list[bool] = []

    def make_client(**kwargs: Any) -> httpx.Client:
        captured.append(kwargs["trust_env"])
        return original_client(**kwargs)

    monkeypatch.setattr(httpx, "Client", make_client)
    transport = httpx.MockTransport(
        lambda request: httpx.Response(200, json={"embeddings": [[1.0]]})
    )
    embed_texts(["text"], settings, transport=transport)
    assert captured == [trust_env]
