import json
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

import capability_capsule.cli as cli
from capability_capsule.manifest import SourceType
from capability_capsule.rag.chunker import TextChunk
from capability_capsule.rag.index import SearchResult
from capability_capsule.runtime.ollama import RagAnswer


def _answer() -> RagAnswer:
    text = "offline source"
    chunk = TextChunk(
        relative_path="docs/offline.md",
        source_type=SourceType.REPO,
        chunk_index=0,
        start_char=0,
        end_char=len(text),
        text=text,
    )
    return RagAnswer(
        answer="Offline answer [1].",
        generation_model="capsule-generator",
        sources=(SearchResult(chunk=chunk, score=0.75),),
    )


@pytest.mark.parametrize("json_output", [False, True])
def test_run_command_answers_from_capsule(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    json_output: bool,
) -> None:
    capsule_path = tmp_path / "flight.zip"
    capsule_path.write_bytes(b"placeholder")
    calls: list[tuple[Any, ...]] = []

    def answer_from_capsule(*args: Any, **kwargs: Any) -> RagAnswer:
        calls.append((args, kwargs))
        return _answer()

    monkeypatch.setattr(cli, "answer_from_capsule", answer_from_capsule)
    arguments = [
        "run",
        str(capsule_path),
        "What works offline?",
        "--top-k",
        "2",
        "--max-context-chars",
        "100",
    ]
    if json_output:
        arguments.append("--json")

    result = CliRunner().invoke(cli.app, arguments)

    assert result.exit_code == 0
    assert calls[0][0] == (capsule_path, "What works offline?")
    assert calls[0][1]["top_k"] == 2
    assert calls[0][1]["max_context_chars"] == 100
    if json_output:
        payload = json.loads(result.stdout)
        assert payload["answer"] == "Offline answer [1]."
        assert payload["sources"][0]["chunk"]["relative_path"] == "docs/offline.md"
    else:
        assert "Offline answer [1]." in result.stdout
        assert "docs/offline.md" in result.stdout


def test_run_command_reports_capsule_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    capsule_path = tmp_path / "broken.zip"
    capsule_path.write_bytes(b"broken")

    def fail(*args: Any, **kwargs: Any) -> RagAnswer:
        raise ValueError("invalid capsule")

    monkeypatch.setattr(cli, "answer_from_capsule", fail)
    result = CliRunner().invoke(cli.app, ["run", str(capsule_path), "question"])

    assert result.exit_code != 0
    assert "invalid capsule" in result.output
