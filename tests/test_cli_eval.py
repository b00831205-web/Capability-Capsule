import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import httpx
import pytest
from typer.testing import CliRunner

import capability_capsule.cli as cli
from capability_capsule.config import Settings
from capability_capsule.eval.retrieval import RetrievalMetrics, summarize_retrieval
from capability_capsule.eval.runner import (
    RetrievalCase,
    RetrievalCaseResult,
    RetrievalEvaluationReport,
)


@pytest.mark.parametrize("as_json", [False, True])
def test_eval_cli_loads_inputs_and_prints_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, as_json: bool,
) -> None:
    index = tmp_path / "index.npz"
    index.touch()
    dataset = tmp_path / "cases.json"
    case = RetrievalCase(question="恢复码？", expected_paths=("recovery.md",))  # noqa: RUF001
    dataset.write_text(json.dumps([case.model_dump(mode="json")]), encoding="utf-8")
    config = tmp_path / "config.toml"
    config.write_text('[ollama]\nembedding_model = "test-embed"\n', encoding="utf-8")
    metrics = RetrievalMetrics(hit=False, recall=0, reciprocal_rank=0)
    report = RetrievalEvaluationReport(
        top_k=2, cases=(RetrievalCaseResult(case=case, sources=(), metrics=metrics),),
        summary=summarize_retrieval((metrics,)),
    )
    calls: list[bool] = []

    def fake_eval(
        cases: Sequence[RetrievalCase], index_path: Path, settings: Settings, *, top_k: int,
    ) -> RetrievalEvaluationReport:
        assert tuple(cases) == (case,)
        assert index_path == index
        assert settings.ollama.embedding_model == "test-embed"
        assert settings.http.trust_env is False
        assert top_k == 2
        calls.append(True)
        return report

    monkeypatch.setattr(cli, "evaluate_retrieval", fake_eval, raising=False)
    args = ["eval", str(index), "--cases", str(dataset), "--config", str(config), "--top-k", "2"]
    if as_json:
        args.append("--json")
    result = CliRunner().invoke(cli.app, args)
    assert result.exit_code == 0, result.output
    assert calls == [True]
    assert not result.stderr.strip()
    if as_json:
        assert json.loads(result.stdout) == report.model_dump(mode="json")
    else:
        assert result.stdout.strip()
        assert "1" in result.stdout
        assert "mrr" in result.stdout.lower()


@pytest.mark.parametrize("invalid", ["dataset", "config", "top_k"])
def test_bad_eval_input_prevents_evaluation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, invalid: str,
) -> None:
    index = tmp_path / "index.npz"
    index.touch()
    dataset = tmp_path / "cases.json"
    dataset.write_text(
        "[]" if invalid == "dataset" else '[{"question":"q","expected_paths":["a.md"]}]',
        encoding="utf-8",
    )
    config = tmp_path / "config.toml"
    config.write_text("broken = [" if invalid == "config" else "", encoding="utf-8")
    calls: list[bool] = []

    def forbidden(*args: Any, **kwargs: Any) -> None:
        calls.append(True)
        pytest.fail("Invalid inputs must be rejected before evaluation")

    monkeypatch.setattr(cli, "evaluate_retrieval", forbidden, raising=False)
    args = ["eval", str(index), "--cases", str(dataset), "--config", str(config), "--json"]
    if invalid == "top_k":
        args.extend(["--top-k", "0"])
    result = CliRunner().invoke(cli.app, args)
    assert result.exit_code != 0
    assert isinstance(result.exception, SystemExit)
    assert calls == []
    assert not result.stdout.strip()
    assert result.stderr.strip()


@pytest.mark.parametrize("failure", [ValueError("index"), httpx.ConnectError("offline")])
def test_eval_failure_keeps_stdout_empty(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: Exception,
) -> None:
    index = tmp_path / "index.npz"
    index.touch()
    dataset = tmp_path / "cases.json"
    dataset.write_text('[{"question":"q","expected_paths":["a.md"]}]', encoding="utf-8")
    calls: list[bool] = []

    def fail(*args: Any, **kwargs: Any) -> None:
        calls.append(True)
        raise failure

    monkeypatch.setattr(cli, "evaluate_retrieval", fail, raising=False)
    result = CliRunner().invoke(cli.app, ["eval", str(index), "--cases", str(dataset), "--json"])
    assert calls == [True]
    assert result.exit_code == 1
    assert isinstance(result.exception, SystemExit)
    assert not result.stdout.strip()
    assert result.stderr.strip()
    assert "Traceback" not in result.stderr
