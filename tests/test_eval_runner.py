import importlib
import json
from pathlib import Path
from typing import Any

import httpx
import pytest
from pydantic import ValidationError

from capability_capsule.config import Settings
from capability_capsule.manifest import SourceType
from capability_capsule.rag.chunker import TextChunk
from capability_capsule.rag.storage import save_index


@pytest.mark.parametrize("top_k", [1, 2])
def test_evaluate_retrieval_uses_embedding_and_aggregates(tmp_path: Path, top_k: int) -> None:
    module = importlib.import_module("capability_capsule.eval.runner")
    index = tmp_path / "index.npz"
    chunks = tuple(
        TextChunk(
            relative_path=path,
            source_type=SourceType.REPO,
            chunk_index=0,
            start_char=0,
            end_char=4,
            text="text",
        )
        for path in ("a.md", "b.md")
    )
    save_index(index, chunks, [[1.0, 0.0], [0.0, 1.0]], embedding_model="nomic-embed-text")
    before = index.read_bytes()
    cases = (
        module.RetrievalCase(question="first", expected_paths=("b.md",)),
        module.RetrievalCase(question="second", expected_paths=("missing.md",)),
    )
    calls: list[list[str]] = []

    def handle(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/embed"
        payload = json.loads(request.content)
        assert payload["model"] == "nomic-embed-text"
        calls.append(payload["input"])
        return httpx.Response(
            200,
            json={
                "embeddings": [[1.0, 0.0] for _ in payload["input"]],
            },
        )

    report = module.evaluate_retrieval(
        cases,
        index,
        Settings(),
        top_k=top_k,
        transport=httpx.MockTransport(handle),
    )
    assert calls == [["first", "second"]]
    assert report.top_k == top_k
    assert tuple(row.case for row in report.cases) == cases
    assert all(len(row.sources) == top_k for row in report.cases)
    assert report.cases[0].metrics.hit is (top_k == 2)
    assert report.cases[1].metrics.hit is False
    assert report.summary.query_count == 2
    assert report.summary.hit_rate == pytest.approx(0.5 if top_k == 2 else 0)
    assert report.summary.mean_recall == pytest.approx(0.5 if top_k == 2 else 0)
    assert report.summary.mrr == pytest.approx(0.25 if top_k == 2 else 0)
    assert index.read_bytes() == before


@pytest.mark.parametrize(
    "data",
    [
        {"question": "", "expected_paths": ["a.md"]},
        {"question": " \n", "expected_paths": ["a.md"]},
        {"question": "q", "expected_paths": []},
        {"question": "q", "expected_paths": [" "]},
        {"question": "q", "expected_paths": ["a.md"], "unknown": True},
    ],
)
def test_retrieval_case_rejects_invalid_input(data: dict[str, Any]) -> None:
    module = importlib.import_module("capability_capsule.eval.runner")
    with pytest.raises(ValidationError):
        module.RetrievalCase.model_validate(data)


@pytest.mark.parametrize(("empty", "top_k"), [(True, 5), (False, 0), (False, -1)])
def test_invalid_run_is_rejected_before_retrieval(
    monkeypatch: pytest.MonkeyPatch,
    empty: bool,
    top_k: int,
) -> None:
    module = importlib.import_module("capability_capsule.eval.runner")

    def forbidden(*args: Any, **kwargs: Any) -> None:
        pytest.fail("Invalid run must not start retrieval")

    monkeypatch.setattr(module, "retrieve_many", forbidden, raising=False)
    cases = () if empty else (module.RetrievalCase(question="q", expected_paths=("a.md",)),)
    with pytest.raises(ValueError):
        module.evaluate_retrieval(cases, Path("unused.npz"), Settings(), top_k=top_k)


def test_retrieval_failure_is_not_reported_as_a_miss(monkeypatch: pytest.MonkeyPatch) -> None:
    module = importlib.import_module("capability_capsule.eval.runner")
    failure = httpx.ConnectError("offline")
    calls: list[tuple[str, ...]] = []

    def fail(questions: tuple[str, ...], *args: Any, **kwargs: Any) -> None:
        calls.append(questions)
        raise failure

    monkeypatch.setattr(module, "retrieve_many", fail, raising=False)
    cases = tuple(module.RetrievalCase(question=q, expected_paths=("a.md",)) for q in ("q1", "q2"))
    with pytest.raises(httpx.ConnectError) as error:
        module.evaluate_retrieval(cases, Path("unused.npz"), Settings())
    assert error.value is failure
    assert calls == [("q1", "q2")]
