"""Opt-in smoke test using the existing local Ollama models."""

import os
from pathlib import Path
from time import perf_counter

import pytest

from capability_capsule.config import Settings
from capability_capsule.packager.build import build_index
from capability_capsule.runtime.ollama import answer_question


@pytest.mark.skipif(
    os.environ.get("CAPSULE_RUN_OLLAMA") != "1",
    reason="Set CAPSULE_RUN_OLLAMA=1 to use local Ollama",
)
def test_real_build_and_answer(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "recovery.md").write_text(
        "# Offline recovery\n"
        "The recovery code for the fictional Silver Finch project is CAPSULE-7319.\n"
        "Use this code when restoring an offline session.\n",
        encoding="utf-8",
    )
    (repo / "theme.md").write_text(
        "# Visual theme\nThe interface uses blue buttons and a white background.\n",
        encoding="utf-8",
    )
    settings = Settings()
    assert settings.http.trust_env is False
    output = tmp_path / "index.npz"
    started = perf_counter()
    build = build_index(repo, output, settings)
    build_seconds = perf_counter() - started
    assert build.document_count == 2
    assert build.chunk_count == 2
    assert build.vector_dimensions > 0
    assert output.stat().st_size == build.size_bytes
    print(f"BUILD {build.model_dump_json()} elapsed_seconds={build_seconds:.2f}", flush=True)

    started = perf_counter()
    result = answer_question(
        "What is the recovery code for the Silver Finch project? "
        "Answer briefly and cite the source.",
        output,
        settings,
        top_k=1,
    )
    print(f"ANSWER {result.answer}", flush=True)
    print(f"ANSWER_SECONDS {perf_counter() - started:.2f}", flush=True)
    print(f"SOURCE {result.sources[0].chunk.relative_path}", flush=True)
    assert len(result.sources) == 1
    assert result.sources[0].chunk.relative_path == "recovery.md"
    # Check the fact, not the model's wording or citation formatting.
    assert "CAPSULE-7319" in result.answer
