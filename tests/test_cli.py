from pathlib import Path
from typing import Any

import httpx
import pytest
from typer.testing import CliRunner

import capability_capsule.cli as cli
from capability_capsule import __version__
from capability_capsule.config import Settings
from capability_capsule.manifest import SourceType
from capability_capsule.packager.build import IndexBuildResult
from capability_capsule.rag.chunker import TextChunk
from capability_capsule.rag.index import SearchResult
from capability_capsule.runtime.ollama import RagAnswer

runner = CliRunner()


def test_help_and_version() -> None:
    help_result = runner.invoke(cli.app, ["--help"])
    assert help_result.exit_code == 0
    assert "build" in help_result.output
    assert "ask" in help_result.output
    version_result = runner.invoke(cli.app, ["--version"])
    assert version_result.exit_code == 0
    assert __version__ in version_result.output


def test_build_passes_paths_and_configuration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "index.npz"
    config = tmp_path / "custom.toml"
    config.write_text('[ollama]\nembedding_model = "custom-embed"\n', encoding="utf-8")
    called: list[bool] = []

    def fake_build(root: Path, output_path: Path, settings: Settings) -> IndexBuildResult:
        assert root.resolve() == tmp_path.resolve()
        assert output_path == output
        assert settings.ollama.embedding_model == "custom-embed"
        assert settings.http.trust_env is False
        called.append(True)
        return IndexBuildResult(
            output_path=output, document_count=2, chunk_count=3,
            vector_dimensions=768, size_bytes=1234,
        )

    monkeypatch.setattr(cli, "build_index", fake_build, raising=False)
    result = runner.invoke(cli.app, [
        "build", "--repo", str(tmp_path), "--output", str(output), "--config", str(config),
    ])
    assert result.exit_code == 0, result.output
    assert called == [True]
    assert "index.npz" in result.output
    assert "768" in result.output


def test_ask_passes_question_and_displays_answer_and_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    index = tmp_path / "index.npz"
    index.touch()

    def fake_answer(
        question: str, index_path: Path, settings: Settings, *, top_k: int,
    ) -> RagAnswer:
        assert question == "Where is the index?"
        assert index_path == index
        assert top_k == 2
        assert settings.ollama.generation_model == "qwen3.5:4b"
        chunk = TextChunk(
            relative_path="notes/design.md", source_type=SourceType.REPO,
            chunk_index=0, start_char=0, end_char=5, text="local",
        )
        return RagAnswer(
            answer="Stored locally [1].", generation_model="qwen3.5:4b",
            sources=(SearchResult(chunk=chunk, score=0.9),),
        )

    monkeypatch.setattr(cli, "answer_question", fake_answer, raising=False)
    result = runner.invoke(cli.app, ["ask", str(index), "Where is the index?", "--top-k", "2"])
    assert result.exit_code == 0, result.output
    assert "Stored locally [1]." in result.output
    assert "notes/design.md" in result.output


@pytest.mark.parametrize("failure", [FileExistsError("exists"), httpx.ConnectError("offline")])
def test_build_failure_has_nonzero_exit_without_traceback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: Exception,
) -> None:
    def fail(*args: Any, **kwargs: Any) -> IndexBuildResult:
        raise failure

    monkeypatch.setattr(cli, "build_index", fail, raising=False)
    result = runner.invoke(cli.app, [
        "build", "--repo", str(tmp_path), "--output", str(tmp_path / "index.npz"),
    ])
    assert result.exit_code == 1
    assert isinstance(result.exception, SystemExit)
    assert result.output.strip()
    assert "Traceback" not in result.output


def test_ask_failure_has_nonzero_exit_without_traceback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    index = tmp_path / "index.npz"
    index.touch()

    def fail(*args: Any, **kwargs: Any) -> RagAnswer:
        raise ValueError("model mismatch")

    monkeypatch.setattr(cli, "answer_question", fail, raising=False)
    result = runner.invoke(cli.app, ["ask", str(index), "question"])
    assert result.exit_code == 1
    assert isinstance(result.exception, SystemExit)
    assert result.output.strip()
    assert "Traceback" not in result.output


def test_bad_config_prevents_build(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config = tmp_path / "bad.toml"
    config.write_text("broken = [", encoding="utf-8")
    called: list[bool] = []

    def fail(*args: Any, **kwargs: Any) -> IndexBuildResult:
        called.append(True)
        raise AssertionError("Build must not run with invalid config")

    monkeypatch.setattr(cli, "build_index", fail, raising=False)
    result = runner.invoke(cli.app, [
        "build", "--repo", str(tmp_path), "--output", str(tmp_path / "index.npz"),
        "--config", str(config),
    ])
    assert result.exit_code == 1
    assert isinstance(result.exception, SystemExit)
    assert called == []
