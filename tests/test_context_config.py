from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError
from typer.testing import CliRunner

import capability_capsule.cli as cli
from capability_capsule.config import Settings
from capability_capsule.runtime.ollama import RagAnswer


def test_default_context_budget() -> None:
    assert Settings().model_dump()["rag"]["max_context_chars"] == 12_000


def test_context_budget_from_toml(tmp_path: Path) -> None:
    config = tmp_path / "config.toml"
    config.write_text("[rag]\nmax_context_chars = 2400\n", encoding="utf-8")
    assert Settings.from_toml(config).model_dump()["rag"]["max_context_chars"] == 2400


@pytest.mark.parametrize("budget", [0, -1])
def test_config_rejects_nonpositive_context_budget(budget: int) -> None:
    with pytest.raises(ValidationError):
        Settings.model_validate({"rag": {"max_context_chars": budget}})


@pytest.mark.parametrize(
    ("configured", "override", "expected"),
    [(None, None, 12_000), (2400, None, 2400), (2400, 800, 800), (None, 800, 800)],
)
def test_cli_context_budget_precedence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    configured: int | None,
    override: int | None,
    expected: int,
) -> None:
    index = tmp_path / "index.npz"
    index.touch()
    args = ["ask", str(index), "question"]
    if configured is not None:
        config = tmp_path / "config.toml"
        config.write_text(f"[rag]\nmax_context_chars = {configured}\n", encoding="utf-8")
        args.extend(["--config", str(config)])
    if override is not None:
        args.extend(["--max-context-chars", str(override)])
    received: list[int | None] = []

    def fake_answer(*args: Any, **kwargs: Any) -> RagAnswer:
        received.append(kwargs.get("max_context_chars"))
        return RagAnswer(answer="answer", generation_model="test", sources=())

    monkeypatch.setattr(cli, "answer_question", fake_answer)
    result = CliRunner().invoke(cli.app, args)
    assert result.exit_code == 0, result.output
    assert received == [expected]


@pytest.mark.parametrize("via_config", [False, True])
@pytest.mark.parametrize("budget", [0, -1])
def test_invalid_budget_prevents_answer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    via_config: bool,
    budget: int,
) -> None:
    index = tmp_path / "index.npz"
    index.touch()
    args = ["ask", str(index), "question"]
    if via_config:
        config = tmp_path / "config.toml"
        config.write_text(f"[rag]\nmax_context_chars = {budget}\n", encoding="utf-8")
        args.extend(["--config", str(config)])
    else:
        args.extend(["--max-context-chars", str(budget)])
    called: list[bool] = []

    def fake_answer(*args: Any, **kwargs: Any) -> RagAnswer:
        called.append(True)
        return RagAnswer(answer="answer", generation_model="test", sources=())

    monkeypatch.setattr(cli, "answer_question", fake_answer)
    result = CliRunner().invoke(cli.app, args)
    assert result.exit_code != 0
    assert isinstance(result.exception, SystemExit)
    assert called == []
