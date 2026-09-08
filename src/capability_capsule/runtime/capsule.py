"""Run retrieval-augmented generation directly from a capsule archive"""

import contextlib
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter
from zipfile import ZipFile

import httpx

from capability_capsule.packager.capsule import inspect_capsule
from capability_capsule.runtime.ollama import RagAnswer, answer_question
from capability_capsule.telemetry.writer import write_run_telemetry


def answer_from_capsule(
    capsule_path: Path,
    question: str,
    *,
    top_k: int = 5,
    max_context_chars: int | None = None,
    transport: httpx.BaseTransport | None = None,
) -> RagAnswer:
    """Validate a capsule and answer a question using its embedded snapshot."""

    inspection = inspect_capsule(capsule_path)
    started_at = perf_counter()

    try:
        with ZipFile(inspection.path, mode="r") as archive:
            index_bytes = archive.read("index.npz")

        with TemporaryDirectory(prefix=".capsule-run-") as temporary_directory:
            index_path = Path(temporary_directory) / "index.npz"
            index_path.write_bytes(index_bytes)

            result = answer_question(
                question,
                index_path,
                inspection.settings,
                top_k=top_k,
                max_context_chars=max_context_chars,
                transport=transport,
            )

    except Exception as error:
        duration_ms = (perf_counter() - started_at) * 1_000

        with contextlib.suppress(OSError):
            write_run_telemetry(
                inspection.settings.telemetry,
                capsule_path=inspection.path,
                capsule_build_id=inspection.manifest.capsule_build_id,
                generation_model=inspection.manifest.generation_model,
                duration_ms=duration_ms,
                question_chars=len(question),
                source_count=0,
                status="error",
                error_type=type(error).__name__,
            )

        raise
    duration_ms = (perf_counter() - started_at) * 1_000

    with contextlib.suppress(OSError):
        write_run_telemetry(
            inspection.settings.telemetry,
            capsule_path=inspection.path,
            capsule_build_id=inspection.manifest.capsule_build_id,
            generation_model=result.generation_model,
            duration_ms=duration_ms,
            question_chars=len(question),
            source_count=len(result.sources),
            status="success",
        )

    return result
