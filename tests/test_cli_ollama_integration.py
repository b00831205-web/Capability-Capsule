"""Opt-in acceptance test of installed CLI commands against local Ollama."""

import json
import os
import subprocess
import sys
from pathlib import Path
from time import perf_counter

import httpx
import pytest

from capability_capsule.config import Settings
from capability_capsule.eval.runner import RetrievalEvaluationReport
from capability_capsule.packager.build import IndexBuildResult
from capability_capsule.runtime.ollama import RagAnswer


@pytest.mark.skipif(
    os.environ.get("CAPSULE_RUN_OLLAMA") != "1",
    reason="Set CAPSULE_RUN_OLLAMA=1 to use local Ollama",
)
def test_real_cli_build_eval_and_ask_json(tmp_path: Path) -> None:
    settings = Settings()
    with httpx.Client(
        base_url=str(settings.ollama.base_url), trust_env=False, timeout=10,
    ) as client:
        version = client.get("/api/version")
        version.raise_for_status()
        tags = client.get("/api/tags")
        tags.raise_for_status()
    names = {model["name"] for model in tags.json()["models"]}
    for required in (settings.ollama.generation_model, settings.ollama.embedding_model):
        assert required in names or f"{required}:latest" in names, required
    print(f"OLLAMA {version.json()['version']}", flush=True)

    repo = tmp_path / "sample repo"
    repo.mkdir()
    recovery = (
        "# Offline recovery\n"
        "The recovery code for the fictional Silver Finch project is CAPSULE-7319.\n"
        "Use this code when restoring an offline session.\n"
    )
    (repo / "recovery.md").write_text(recovery, encoding="utf-8")
    (repo / "theme.md").write_text(
        "# Visual theme\nThe interface uses blue buttons and a white background.\n",
        encoding="utf-8",
    )
    config = tmp_path / "config.toml"
    config.write_text(
        "[http]\ntrust_env = false\n[rag]\nmax_context_chars = 120\n",
        encoding="utf-8",
    )
    output = tmp_path / "knowledge index.npz"
    executable = Path(sys.executable).parent / "capsule"
    assert executable.is_file(), "Installed capsule command is missing"

    def run_cli(*args: str) -> str:
        started = perf_counter()
        process = subprocess.run(
            [str(executable), *args], cwd=tmp_path, capture_output=True,
            text=True, encoding="utf-8", timeout=240, check=False,
        )
        assert process.returncode == 0, process.stderr or process.stdout
        assert not process.stderr.strip(), process.stderr
        print(f"CLI {args[0]} elapsed_seconds={perf_counter() - started:.2f}", flush=True)
        return process.stdout

    build = IndexBuildResult.model_validate(json.loads(run_cli(
        "build", "--repo", str(repo), "--output", str(output),
        "--config", str(config), "--json",
    )))
    assert build.output_path == output
    assert build.document_count == 2
    assert build.chunk_count == 2
    assert build.size_bytes == output.stat().st_size
    print(f"BUILD {build.model_dump_json()}", flush=True)

    dataset = tmp_path / "evaluation cases.json"
    case_data = [
        {
            "question": "What is the recovery code for the Silver Finch project?",
            "expected_paths": ["recovery.md"],
        },
        {
            "question": "What colors are used for the interface buttons and background?",
            "expected_paths": ["theme.md"],
        },
    ]
    dataset.write_text(json.dumps(case_data, indent=2), encoding="utf-8")
    index_before = output.read_bytes()
    report = RetrievalEvaluationReport.model_validate(json.loads(run_cli(
        "eval", str(output), "--cases", str(dataset), "--top-k", "1",
        "--config", str(config), "--json",
    )))
    assert report.top_k == 1
    assert len(report.cases) == 2
    assert [row.case.model_dump(mode="json") for row in report.cases] == case_data
    for row in report.cases:
        assert len(row.sources) == 1
        assert row.sources[0].chunk.relative_path == row.case.expected_paths[0]
        assert row.metrics.hit is True
        assert row.metrics.recall == pytest.approx(1.0)
        assert row.metrics.reciprocal_rank == pytest.approx(1.0)
    assert report.summary.query_count == 2
    assert report.summary.hit_rate == pytest.approx(1.0)
    assert report.summary.mean_recall == pytest.approx(1.0)
    assert report.summary.mrr == pytest.approx(1.0)
    assert output.read_bytes() == index_before
    print(f"EVAL {report.summary.model_dump_json()}", flush=True)

    result = RagAnswer.model_validate(json.loads(run_cli(
        "ask", str(output),
        "What is the recovery code for the Silver Finch project? "
        "Answer briefly and cite the source.",
        "--top-k", "1", "--config", str(config), "--json",
    )))
    assert result.generation_model == settings.ollama.generation_model
    assert "CAPSULE-7319" in result.answer
    assert len(result.sources) == 1
    chunk = result.sources[0].chunk
    assert chunk.relative_path == "recovery.md"
    assert len(chunk.text) == 120
    assert chunk.text == recovery[chunk.start_char:chunk.end_char]
    print(f"ANSWER {result.answer}", flush=True)
    print(f"SOURCE {chunk.relative_path} chars={len(chunk.text)}", flush=True)
