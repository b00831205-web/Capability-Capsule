import json
from pathlib import Path
from typing import Any

import httpx
import pytest
from typer.testing import CliRunner

import capability_capsule.cli as cli
from capability_capsule.config import Settings
from capability_capsule.manifest import SourceType
from capability_capsule.packager.build import IndexBuildResult
from capability_capsule.rag.chunker import TextChunk
from capability_capsule.rag.index import SearchResult
from capability_capsule.runtime.ollama import RagAnswer


@pytest.mark.parametrize("with_sources", [True, False])
def test_ask_json_outputs_complete_answer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, with_sources: bool,
) -> None:
    index = tmp_path / "index.npz"
    index.touch()
    chunk = TextChunk(
        relative_path="notes/说明.md", source_type=SourceType.REPO,
        chunk_index=0, start_char=10, end_char=14, text="本地资料",
    )
    answer = RagAnswer(
        answer='回答含有 "引号" 和换行\n第二行。', generation_model="test-model",
        sources=(SearchResult(chunk=chunk, score=0.875),) if with_sources else (),
    )
    calls: list[bool] = []

    def fake_answer(*args: Any, **kwargs: Any) -> RagAnswer:
        calls.append(True)
        assert kwargs["max_context_chars"] == 100
        return answer

    monkeypatch.setattr(cli, "answer_question", fake_answer)
    result = CliRunner().invoke(cli.app, [
        "ask", str(index), "问题", "--json", "--max-context-chars", "100",
    ])
    assert result.exit_code == 0, result.output
    assert calls == [True]
    assert json.loads(result.stdout) == answer.model_dump(mode="json")
    assert not result.stderr.strip()


@pytest.mark.parametrize("failure", [ValueError("invalid"), httpx.ConnectError("offline")])
def test_ask_json_failure_keeps_stdout_empty(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: Exception,
) -> None:
    index = tmp_path / "index.npz"
    index.touch()
    calls: list[bool] = []

    def fake_answer(*args: Any, **kwargs: Any) -> RagAnswer:
        calls.append(True)
        raise failure

    monkeypatch.setattr(cli, "answer_question", fake_answer)
    result = CliRunner().invoke(cli.app, ["ask", str(index), "question", "--json"])
    assert calls == [True]
    assert result.exit_code == 1
    assert isinstance(result.exception, SystemExit)
    assert not result.stdout.strip()
    assert result.stderr.strip()
    assert "Traceback" not in result.stderr


def test_build_json_outputs_complete_summary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "知识 index.npz"
    config = tmp_path / "config.toml"
    config.write_text('[ollama]\nembedding_model = "test-embed"\n', encoding="utf-8")
    summary = IndexBuildResult(
        output_path=output, document_count=2, chunk_count=3,
        vector_dimensions=768, size_bytes=1234,
    )
    calls: list[bool] = []

    def fake_build(root: Path, output_path: Path, settings: Settings) -> IndexBuildResult:
        assert root == tmp_path
        assert output_path == output
        assert settings.ollama.embedding_model == "test-embed"
        calls.append(True)
        return summary

    monkeypatch.setattr(cli, "build_index", fake_build)
    result = CliRunner().invoke(cli.app, [
        "build", "--repo", str(tmp_path), "--output", str(output),
        "--config", str(config), "--json",
    ])
    assert result.exit_code == 0, result.output
    assert calls == [True]
    assert json.loads(result.stdout) == summary.model_dump(mode="json")
    assert not result.stderr.strip()


@pytest.mark.parametrize(
    "failure", [FileExistsError("exists"), ValueError("budget"), httpx.ConnectError("offline")],
)
def test_build_json_failure_keeps_stdout_empty(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: Exception,
) -> None:
    calls: list[bool] = []

    def fake_build(*args: Any, **kwargs: Any) -> IndexBuildResult:
        calls.append(True)
        raise failure

    monkeypatch.setattr(cli, "build_index", fake_build)
    result = CliRunner().invoke(cli.app, [
        "build", "--repo", str(tmp_path), "--output", str(tmp_path / "index.npz"), "--json",
    ])
    assert calls == [True]
    assert result.exit_code == 1
    assert isinstance(result.exception, SystemExit)
    assert not result.stdout.strip()
    assert result.stderr.strip()
    assert "Traceback" not in result.stderr
