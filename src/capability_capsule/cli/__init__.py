"""Command-line interface for building index and asking questions"""

from pathlib import Path
from typing import Annotated

import httpx
import typer

from capability_capsule import __version__
from capability_capsule.config import Settings
from capability_capsule.packager.build import build_index
from capability_capsule.runtime.ollama import answer_question

app = typer.Typer(
    name = "capsule",
    help = "build local knowledge indexs and answer questions with Ollama.",
    no_args_is_help = True,
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
        typer.Option(
            "--version",
            callback = _version_callback,
            is_eager = True,
            help = "Show version."
        ),
    ] = False,
) -> None:
    """Capability Capsule command-line interface"""

@app.command("build")
def build_command(
    repo: Annotated[
        Path,
        typer.Option(
            "--repo",
            exists = True,
            file_okay=False,
            help = "Repository directory to index"
        ),
    ],
    output: Annotated[
        Path,
        typer.Option(
            "--output",
            dir_okay = False,
            help = "New index file; ites parent directory must exists."
        ),
    ],
    config: Annotated[
        Path | None,
        typer.Option(
            "--config",
            exists = True,
            dir_okay = False,
            help = "Optional TOML configuration file. "
            ),
        ] = None,
) -> None:
    """Build and save a repository index."""

    try:
        settings = _load_settings(config)
        result = build_index(repo, output, settings)
    except (OSError, ValueError, httpx.HTTPError) as error:
        typer.echo(f"Build failed: {error}", err = True)
        raise typer.Exit(code = 1) from error

    typer.echo(f"Index: {result.output_path}")
    typer.echo(f"Documents: {result.document_count}")
    typer.echo(f"Chunk: {result.chunk_count}")
    typer.echo(f"Vector dimensions: {result.vector_dimensions}")
    typer.echo(f"Size: {result.size_bytes} bytes")

@app.command("ask")
def ask_command(
    index_path: Annotated[
        Path,
        typer.Argument(
            exists = True,
            dir_okay= False,
            help= "Sved index file."
        ),
    ],
    question: Annotated[
        str,
        typer.Argument(
            help = "Question to answer from the index."
            )
    ],
    top_k: Annotated[
        int, 
        typer.Option(
            "--top-k",
            min = 1,
            help = "Maximum number of source chunks to retrieve"
        ),
    ] = 5,
    config: Annotated[
        Path | None,
        typer.Option(
            "--config",
            exists = True,
            dir_okay = False,
            help = "Optional TOML configuration file."
        ),
    ] = None,
) -> None:
    """Answer a question using a saved index and local Ollama"""

    try:
        settings = _load_settings(config)
        result = answer_question(
            question,
            index_path,
            settings,
            top_k = top_k,
        )
    except ( OSError, ValueError, httpx.HTTPError) as error:
        typer.echo(f"Question failed: {error}", err = True)
        raise typer.Exit(code = 1) from error

    typer.echo(result.answer)
    typer.echo()
    typer.echo("Retrieve sources:")

    for number, source in enumerate(result.sources, start = 1):
        chunk = source.chunk
        typer.echo(
            f"[{number}] {chunk.relative_path} "
            f"chars {chunk.start_char}:{chunk.end_char} "
            f"score={source.score:.3f}"
        )
    