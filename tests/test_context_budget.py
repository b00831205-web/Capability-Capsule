import json
from pathlib import Path
from typing import Any

import httpx
import pytest

import capability_capsule.runtime.ollama as ollama
from capability_capsule.config import Settings
from capability_capsule.manifest import SourceType
from capability_capsule.rag.chunker import TextChunk
from capability_capsule.rag.index import SearchResult


def _source(text: str, number: int) -> SearchResult:
    return SearchResult(
        chunk=TextChunk(
            relative_path=f"notes/{number}.md",
            source_type=SourceType.REPO,
            chunk_index=number,
            start_char=10,
            end_char=10 + len(text),
            text=text,
        ),
        score=1.0 / (number + 1),
    )


@pytest.mark.parametrize(
    ("budget", "expected_texts"),
    [
        (100, ["abcdef", "ghijkl", "mnopqr"]),
        (12, ["abcdef", "ghijkl"]),
        (8, ["abcdef", "gh"]),
        (3, ["abc"]),
    ],
)
def test_context_budget_matches_sent_and_returned_sources(
    monkeypatch: pytest.MonkeyPatch,
    budget: int,
    expected_texts: list[str],
) -> None:
    original = tuple(_source(text, i) for i, text in enumerate(["abcdef", "ghijkl", "mnopqr"]))

    def fake_retrieve(*args: Any, **kwargs: Any) -> tuple[SearchResult, ...]:
        return original

    calls: list[httpx.Request] = []

    def handle(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        assert request.url.path == "/api/chat"
        user_data = json.loads(json.loads(request.content)["messages"][1]["content"])
        sent = user_data["sources"]
        assert [source["text"] for source in sent] == expected_texts
        assert [source["id"] for source in sent] == list(range(1, len(sent) + 1))
        assert sum(len(source["text"]) for source in sent) <= budget
        for source in sent:
            assert source["end_char"] - source["start_char"] == len(source["text"])
        return httpx.Response(
            200,
            json={
                "done": True,
                "message": {"role": "assistant", "content": "answer"},
            },
        )

    monkeypatch.setattr(ollama, "retrieve", fake_retrieve)
    result = ollama.answer_question(
        "question",
        Path("unused.npz"),
        Settings(),
        max_context_chars=budget,
        transport=httpx.MockTransport(handle),
    )
    assert len(calls) == 1
    assert [source.chunk.text for source in result.sources] == expected_texts
    for number, source in enumerate(result.sources):
        assert source.chunk.relative_path == original[number].chunk.relative_path
        assert source.chunk.start_char == 10
        assert source.chunk.end_char == 10 + len(expected_texts[number])
        assert source.score == original[number].score
    assert [source.chunk.text for source in original] == ["abcdef", "ghijkl", "mnopqr"]
    assert all(source.chunk.end_char == 16 for source in original)


@pytest.mark.parametrize("budget", [0, -1])
def test_invalid_budget_fails_before_retrieval(
    monkeypatch: pytest.MonkeyPatch,
    budget: int,
) -> None:
    def forbidden(*args: Any, **kwargs: Any) -> tuple[SearchResult, ...]:
        pytest.fail("Invalid budget must be rejected before retrieval")

    monkeypatch.setattr(ollama, "retrieve", forbidden)
    with pytest.raises(ValueError):
        ollama.answer_question("question", Path("unused.npz"), Settings(), max_context_chars=budget)


def test_unicode_budget_counts_characters(monkeypatch: pytest.MonkeyPatch) -> None:
    original = (_source("甲🙂\n乙", 0),)

    def fake_retrieve(*args: Any, **kwargs: Any) -> tuple[SearchResult, ...]:
        return original

    def handle(request: httpx.Request) -> httpx.Response:
        user_data = json.loads(json.loads(request.content)["messages"][1]["content"])
        assert user_data["sources"][0]["text"] == "甲🙂\n"
        return httpx.Response(
            200,
            json={
                "done": True,
                "message": {"role": "assistant", "content": "answer"},
            },
        )

    monkeypatch.setattr(ollama, "retrieve", fake_retrieve)
    result = ollama.answer_question(
        "question",
        Path("unused.npz"),
        Settings(),
        max_context_chars=3,
        transport=httpx.MockTransport(handle),
    )
    assert result.sources[0].chunk.text == "甲🙂\n"
    assert result.sources[0].chunk.end_char == 13


@pytest.mark.parametrize(
    ("configured", "explicit", "expected_length"),
    [
        (None, {}, 12_000),
        (3, {}, 3),
        (3, {"max_context_chars": None}, 3),
        (3, {"max_context_chars": 8}, 8),
    ],
)
def test_runtime_context_budget_precedence(
    monkeypatch: pytest.MonkeyPatch,
    configured: int | None,
    explicit: dict[str, Any],
    expected_length: int,
) -> None:
    settings = Settings.model_validate(
        {} if configured is None else {"rag": {"max_context_chars": configured}}
    )
    original = (_source("a" * 12_010, 0),)

    def fake_retrieve(*args: Any, **kwargs: Any) -> tuple[SearchResult, ...]:
        return original

    sent_texts: list[str] = []

    def handle(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/chat"
        data = json.loads(json.loads(request.content)["messages"][1]["content"])
        sent_texts.extend(source["text"] for source in data["sources"])
        return httpx.Response(
            200,
            json={
                "done": True,
                "message": {"role": "assistant", "content": "answer"},
            },
        )

    monkeypatch.setattr(ollama, "retrieve", fake_retrieve)
    result = ollama.answer_question(
        "question",
        Path("unused.npz"),
        settings,
        transport=httpx.MockTransport(handle),
        **explicit,
    )
    assert sent_texts == ["a" * expected_length]
    assert [source.chunk.text for source in result.sources] == sent_texts
    assert result.sources[0].chunk.end_char == 10 + expected_length
    assert len(original[0].chunk.text) == 12_010
    assert settings.rag.max_context_chars == (12_000 if configured is None else configured)
