"""Command-line interface for building index and asking questions"""

from pathlib import Path
from typing import Annotated

import httpx
import typer

from capability_capsule import __version__
from capability_capsule.config import Settings
from capability_capsule.eval.dataset import load_cases
from capability_capsule.eval.runner import evaluate_retrieval
from capability_capsule.packager.build import build_index
from capability_capsule.packager.capsule import build_capsule, inspect_capsule
from capability_capsule.runtime.agent import run_read_only_agent
from capability_capsule.runtime.capsule import answer_from_capsule
from capability_capsule.runtime.ollama import answer_question
from capability_capsule.runtime.policy_config import load_tool_policy
from capability_capsule.runtime.readiness import check_capsule_readiness
from capability_capsule.telemetry.report import summarize_telemetry

app = typer.Typer(
    name="capsule",
    help="build local knowledge indexs and answer questions with Ollama.",
    no_args_is_help=True,
)


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(__version__)
        raise typer.Exit


def _load_settings(config: Path | None) -> Settings:
    if config is None:
        return Settings()
    return Settings.from_toml(config)


@app.callback(invoke_without_command=True)
def main(
    version: Annotated[
        bool,
        typer.Option("--version", callback=_version_callback, is_eager=True, help="Show version."),
    ] = False,
) -> None:
    """Capability Capsule command-line interface"""


@app.command("build")
def build_command(
    repo: Annotated[
        Path,
        typer.Option("--repo", exists=True, file_okay=False, help="Repository directory to index"),
    ],
    output: Annotated[
        Path,
        typer.Option(
            "--output", dir_okay=False, help="New index file; ites parent directory must exists."
        ),
    ],
    config: Annotated[
        Path | None,
        typer.Option(
            "--config", exists=True, dir_okay=False, help="Optional TOML configuration file. "
        ),
    ] = None,
    json_output: Annotated[
        bool, typer.Option("--json", help="Output the index build summary as JSON.")
    ] = False,
) -> None:
    """Build and save a repository index."""

    try:
        settings = _load_settings(config)
        result = build_index(repo, output, settings)
    except (OSError, ValueError, httpx.HTTPError) as error:
        typer.echo(f"Build failed: {error}", err=True)
        raise typer.Exit(code=1) from error

    if json_output:
        typer.echo(result.model_dump_json(indent=2))
        return

    typer.echo(f"Index: {result.output_path}")
    typer.echo(f"Documents: {result.document_count}")
    typer.echo(f"Chunk: {result.chunk_count}")
    typer.echo(f"Vector dimensions: {result.vector_dimensions}")
    typer.echo(f"Size: {result.size_bytes} bytes")


@app.command("pack")
def pack_command(
    repo: Annotated[
        Path,
        typer.Option(
            "--repo", exists=True, file_okay=False, help="Repository directory to package."
        ),
    ],
    output: Annotated[
        Path,
        typer.Option("--output", dir_okay=False, help="New capsule ZIP file."),
    ],
    task: Annotated[
        str,
        typer.Option("--task", help="Future offline task supported by the capsule."),
    ],
    config: Annotated[
        Path | None,
        typer.Option(
            "--config", exists=True, dir_okay=False, help="Optional TOML configuration file."
        ),
    ] = None,
    json_output: Annotated[
        bool,
        typer.Option("--json", help="Output the complete capsule build result as JSON."),
    ] = False,
) -> None:
    """Build a portable offline capsule from a repository."""

    try:
        settings = _load_settings(config)
        result = build_capsule(
            repo,
            output,
            task,
            settings,
        )
    except (OSError, ValueError, httpx.HTTPError) as error:
        typer.echo(f"Capsule build failed: {error}", err=True)
        raise typer.Exit(code=1) from error

    if json_output:
        typer.echo(result.model_dump_json(indent=2))
        return
    typer.echo(f"Capsule: {result.output_path}")
    typer.echo(f"Build ID: {result.manifest.capsule_build_id}")
    typer.echo(f"Documents: {result.document_count}")
    typer.echo(f"Chunks: {result.chunk_count}")
    typer.echo(f"Vector dimentsions: {result.vector_dimensions}")
    typer.echo(f"Size: {result.size_bytes} bytes")


@app.command("inspect")
def inspect_command(
    capsule_path: Annotated[
        Path,
        typer.Argument(exists=True, dir_okay=False, help="Capsule ZIP file to validate."),
    ],
    json_output: Annotated[
        bool,
        typer.Option("--json", help="Output validated capsule metadata as JSON."),
    ] = False,
) -> None:
    """Validate and describe an existing capsule."""

    try:
        result = inspect_capsule(capsule_path)
    except (OSError, ValueError) as error:
        typer.echo(f"Capsule inspection failed: {error}", err=True)
        raise typer.Exit(code=1) from error

    if json_output:
        typer.echo(result.model_dump_json(indent=2))
        return

    manifest = result.manifest

    typer.echo(f"Capsule: {result.path}")
    typer.echo(f"Build ID: {manifest.capsule_build_id}")
    typer.echo(f"Task: {manifest.task}")
    typer.echo(f"Generation model: {manifest.generation_model}")
    typer.echo(f"Embedding model: {manifest.embedding_model}")
    typer.echo(f"Artifacts: {len(manifest.artifacts)}")
    typer.echo(f"Index size: {result.index_size_bytes} bytes")
    typer.echo(f"Package size: {result.package_size_bytes} bytes")
    typer.echo(f"Size budget: {manifest.size_budget_bytes} bytes")
    typer.echo(f"Offline duration: {manifest.offline_duration_hours:g} hours")


@app.command("doctor")
def doctor_command(
    capsule_path: Annotated[
        Path,
        typer.Argument(exists=True, dir_okay=False, help="Capsule ZIP file to check"),
    ],
    json_output: Annotated[
        bool,
        typer.Option("--json", help="Output the readiness report as JSON"),
    ] = False,
) -> None:
    """Check whether a capsule is ready for offline use."""

    try:
        report = check_capsule_readiness(capsule_path)
    except (OSError, ValueError, httpx.HTTPError) as error:
        typer.echo(f"Readiness check failed: {error}", err=True)
        raise typer.Exit(code=1) from error

    if json_output:
        typer.echo(report.model_dump_json(indent=2))

    else:
        typer.echo(f"Capsule: {report.capsule_path}")
        typer.echo(f"Build ID: {report.capsule_build_id}")
        typer.echo(f"Ollama version: {report.ollama_version}")

        for model in report.required_models:
            status = "missing" if model in report.missing_models else "available"
            typer.echo(f"Model: {model} ({status})")

        typer.echo(f"Ready: {'yes' if report.ready else 'no'}")
    if not report.ready:
        raise typer.Exit(code=1)


@app.command("run")
def run_command(
    capsule_path: Annotated[
        Path,
        typer.Argument(exists=True, dir_okay=False, help="Capsule ZIP file to run."),
    ],
    question: Annotated[str, typer.Argument(help="Question to answer from the capsule.")],
    top_k: Annotated[
        int,
        typer.Option("--top-k", min=1, help="Maximum number of source chunks to retrieve"),
    ] = 5,
    max_context_chars: Annotated[
        int | None,
        typer.Option(
            "--max-context-chars",
            min=1,
            help="Maximum source-text characters supplied to the model.",
        ),
    ] = None,
    json_output: Annotated[
        bool,
        typer.Option("--json", help="Output the answer and sources as JSON."),
    ] = False,
) -> None:
    """Answer a question directly from a portable capsule."""

    try:
        result = answer_from_capsule(
            capsule_path,
            question,
            top_k=top_k,
            max_context_chars=max_context_chars,
        )

    except (OSError, ValueError, httpx.HTTPError) as error:
        typer.echo(f"Capsule run failed: {error}", err=True)
        raise typer.Exit(code=1) from error

    if json_output:
        typer.echo(result.model_dump_json(indent=2))
        return

    typer.echo(result.answer)
    typer.echo()
    typer.echo("Retrieved sources:")

    for number, source in enumerate(result.sources, start=1):
        chunk = source.chunk
        typer.echo(
            f"[{number}] {chunk.relative_path} "
            f"chars {chunk.start_char}:{chunk.end_char} "
            f"score={source.score:.3f}"
        )


@app.command("ask")
def ask_command(
    index_path: Annotated[
        Path,
        typer.Argument(exists=True, dir_okay=False, help="Sved index file."),
    ],
    question: Annotated[str, typer.Argument(help="Question to answer from the index.")],
    top_k: Annotated[
        int,
        typer.Option("--top-k", min=1, help="Maximum number of source chunks to retrieve"),
    ] = 5,
    max_context_chars: Annotated[
        int | None,
        typer.Option(
            "--max-context-chars",
            min=1,
            help="Maximum source-text characters supplied to the model.",
        ),
    ] = None,
    config: Annotated[
        Path | None,
        typer.Option(
            "--config", exists=True, dir_okay=False, help="Optional TOML configuration file."
        ),
    ] = None,
    json_output: Annotated[
        bool,
        typer.Option("--json", help="Output the answer and sources as JSON."),
    ] = False,
) -> None:
    """Answer a question using a saved index and local Ollama"""

    try:
        settings = _load_settings(config)
        result = answer_question(
            question,
            index_path,
            settings,
            top_k=top_k,
            max_context_chars=(
                max_context_chars
                if max_context_chars is not None
                else settings.rag.max_context_chars
            ),
        )
    except (OSError, ValueError, httpx.HTTPError) as error:
        typer.echo(f"Question failed: {error}", err=True)
        raise typer.Exit(code=1) from error

    if json_output:
        typer.echo(result.model_dump_json(indent=2))
        return

    typer.echo(result.answer)
    typer.echo()
    typer.echo("Retrieve sources:")

    for number, source in enumerate(result.sources, start=1):
        chunk = source.chunk
        typer.echo(
            f"[{number}] {chunk.relative_path} "
            f"chars {chunk.start_char}:{chunk.end_char} "
            f"score={source.score:.3f}"
        )


@app.command("eval")
def eval_command(
    index_path: Annotated[
        Path, typer.Argument(exists=True, dir_okay=False, help="Saved index file to evaluate.")
    ],
    cases_path: Annotated[
        Path,
        typer.Option(
            "--cases",
            exists=True,
            dir_okay=False,
            help="JSON file containing evaluation cases.",
        ),
    ],
    top_k: Annotated[
        int,
        typer.Option("--top-k", min=1, help="Maximum number of chunks to retrieve per question."),
    ] = 5,
    config: Annotated[
        Path | None,
        typer.Option(
            "--config", exists=True, dir_okay=False, help="Optional TOML configuration file."
        ),
    ] = None,
    json_output: Annotated[
        bool,
        typer.Option("--json", help="Output the complete evaluation report as JSON."),
    ] = False,
) -> None:
    """Evaluate retrieval against expected source files."""

    try:
        settings = _load_settings(config)
        cases = load_cases(cases_path)
        report = evaluate_retrieval(cases, index_path, settings, top_k=top_k)
    except (OSError, ValueError, httpx.HTTPError) as error:
        typer.echo(f"Evaluation failed: {error}", err=True)
        raise typer.Exit(code=1) from error

    if json_output:
        typer.echo(report.model_dump_json(indent=2))
        return

    summary = report.summary
    typer.echo(f"Questions: {summary.query_count}")
    typer.echo(f"Top k: {report.top_k}")
    typer.echo(f"Hit rate: {summary.hit_rate:.3f}")
    typer.echo(f"Mean recall: {summary.mean_recall:.3f}")
    typer.echo(f"MRR: {summary.mrr:.3f}")


@app.command("report")
def report_command(
    output_dir: Annotated[
        Path,
        typer.Argument(
            exists=True,
            file_okay=False,
            help="Directory containing local telemetry events",
        ),
    ],
    json_output: Annotated[
        bool,
        typer.Option("--json", help="Output the telemetry summary as JSON."),
    ] = False,
) -> None:
    """Summarize local capsule runtime telemetry."""

    try:
        report = summarize_telemetry(output_dir)

    except (OSError, ValueError) as error:
        typer.echo(f"Telemetry report failed: {error}", err=True)
        raise typer.Exit(code=1) from error

    if json_output:
        typer.echo(report.model_dump_json(indent=2))
        return

    typer.echo(f"Telemetry directory: {report.output_dir}")
    typer.echo(f"Runs: {report.event_count}")
    typer.echo(f"Successful: {report.success_count}")
    typer.echo(f"Failed: {report.error_count}")
    typer.echo(f"Success rate: {report.success_rate:.1%}")
    typer.echo(f"Mean duration: {report.mean_duration_ms:.1f} ms")
    typer.echo(f"P95 duration: {report.p95_duration_ms:.1f} ms")
    typer.echo(f"Question characters: {report.total_question_chars}")
    typer.echo(f"Retrieved sources: {report.total_source_count}")

    if report.error_types:
        typer.echo("Error types:")

        for error_type, count in report.error_types.items():
            typer.echo(f"- {error_type}: {count}")


@app.command("agent")
def agent_command(
    task: Annotated[
        str,
        typer.Argument(help="Repository task for the read-only local agent"),
    ],
    workspace: Annotated[
        Path,
        typer.Option(
            "--workspace",
            exists=True,
            file_okay=False,
            help="Workspace directory the agent may inspect.",
        ),
    ],
    policy_path: Annotated[
        Path | None,
        typer.Option(
            "--policy",
            exists=True,
            dir_okay=False,
            help="Optional host-controlled TOML tool policy.",
        ),
    ] = None,
    config: Annotated[
        Path | None,
        typer.Option(
            "--config", exists=True, dir_okay=False, help="Optional TOML configuration file."
        ),
    ] = None,
    max_tool_rounds: Annotated[
        int,
        typer.Option("--max-tool-rounds", min=1, help="Maximum model-to-tool interaction rounds."),
    ] = 8,
    max_tool_calls: Annotated[
        int,
        typer.Option("--max-tool-calls", min=1, help="Maximum total tool calls."),
    ] = 16,
    json_output: Annotated[
        bool,
        typer.Option(
            "--json",
            help="Output the agent result as JSON",
        ),
    ] = False,
) -> None:
    """Run a bounded read-only local workspace agent."""

    try:
        settings = _load_settings(config)
        policy = load_tool_policy(policy_path) if policy_path is not None else None
        result = run_read_only_agent(
            task,
            workspace,
            settings,
            policy=policy,
            max_tool_rounds=max_tool_rounds,
            max_tool_calls=max_tool_calls,
        )

    except (
        OSError,
        ValueError,
        RuntimeError,
        httpx.HTTPError,
    ) as error:
        typer.echo(f"Agent failed: {error}", err=True)
        raise typer.Exit(code=1) from error

    if json_output:
        typer.echo(result.model_dump_json(indent=2))
        return

    typer.echo(result.answer)
    typer.echo()
    typer.echo(f"Tool calls: {result.tool_call_count}")

    if result.tool_names:
        typer.echo(f"Tools used: {', '.join(result.tool_names)}")
