# Capability Capsule

Capability Capsule builds portable, task-specific knowledge packages for local work when network
access is unavailable. The current V0 flight MVP can scan a repository, build an Ollama-backed
vector index, package and validate a capsule, check local model readiness, answer questions from the
capsule, and summarize privacy-preserving local runtime telemetry.

## Current scope

The implemented offline chain is:

```text
repository -> pack -> inspect -> doctor -> run -> report
```

A capsule is a ZIP archive containing exactly:

- `manifest.json`: task, build identity, model requirements, budget, and artifact provenance.
- `settings.json`: the validated configuration snapshot used to build and run the capsule.
- `index.npz`: text chunks, vector-index metadata, and embedding vectors.

V0 does not include a cloud Teacher, model downloads, LoRA, a database, Docker, or a Web UI.

## Development environment

The supported development environment is Ubuntu on WSL2. The repository must be available at
`/mnt/e/capsule`. Python and dependencies are managed inside the project by uv; the Windows and WSL
global Python environments are not modified.

```bash
cd /mnt/e/capsule
uv sync --all-groups
uv run python --version
uv run capsule --help
```

The project targets Python 3.13 and uses Hatchling as its build backend.

## Ollama

Ollama runs on Windows and is accessed from WSL at `http://127.0.0.1:11434`. The default capsule
configuration uses:

- Generation: `qwen3.5:4b`
- Embedding: `nomic-embed-text`

WSL may inherit proxy variables. Keep `http.trust_env = false` so local Ollama requests bypass the
proxy. Capability Capsule never installs Ollama or downloads missing models. Use `capsule doctor` to
report missing requirements.

## Configuration

Copy `config.example.toml` to `config.toml` for local overrides. `config.toml` is ignored by Git.

```bash
cp config.example.toml config.toml
```

The selected configuration is embedded in the capsule during `pack`. The `run` and `doctor`
commands use that embedded snapshot rather than an external configuration file, keeping the runtime
models consistent with the stored vectors.

## Flight MVP workflow

Build a capsule from a repository:

```bash
uv run capsule pack \
  --repo /path/to/repository \
  --output /path/to/flight.zip \
  --task "answer repository questions during a flight" \
  --config config.toml
```

Validate its archive, metadata, index, and declared size budget:

```bash
uv run capsule inspect /path/to/flight.zip
```

Check Ollama connectivity and required local models without downloading anything:

```bash
uv run capsule doctor /path/to/flight.zip
```

Ask a question directly from the capsule:

```bash
uv run capsule run /path/to/flight.zip "Where is the main entry point?"
```

Limit retrieval or source text when needed:

```bash
uv run capsule run /path/to/flight.zip "How is configuration loaded?" \
  --top-k 3 \
  --max-context-chars 8000
```

Each of `pack`, `inspect`, `doctor`, and `run` supports `--json` for structured output.

## Local telemetry

When enabled, each `run` writes one JSON event under the configured telemetry directory. Relative
paths are resolved beside the capsule; the default is `.capsule/sessions`. Events include build ID,
model, status, duration, question character count, source count, and error type. They do not include
the question text or answer text and are never uploaded.

Summarize the local events with:

```bash
uv run capsule report /path/to/.capsule/sessions
uv run capsule report /path/to/.capsule/sessions --json
```

Set `telemetry.enabled = false` in the configuration before packing to disable event files.

## Index and retrieval evaluation

The lower-level `build` and `ask` commands operate on a standalone `index.npz`. Retrieval evaluation
uses a JSON case set and reports hit rate, mean recall, and mean reciprocal rank (MRR):

```bash
uv run capsule build --repo /path/to/repository --output /path/to/index.npz
uv run capsule ask /path/to/index.npz "How does this project work?"
uv run capsule eval /path/to/index.npz --cases /path/to/cases.json
```

Evaluation currently measures retrieval quality; it does not yet grade generated-answer correctness
or automatically reject a capsule below a quality threshold.

## Verification

Run the standard checks:

```bash
uv run pytest
uv run mypy src tests
uv run ruff check .
uv run ruff format --check .
```

Real Ollama integration tests are opt-in:

```bash
CAPSULE_RUN_OLLAMA=1 uv run pytest tests/test_flight_mvp_integration.py -q
```

The integration test uses pytest temporary directories and exercises the complete
`pack -> doctor -> run -> report` chain without downloading models.

## Package layout

All Python packages live under `src/capability_capsule`:

- `scanner`: repository and document discovery.
- `rag`: chunking, embeddings, vector search, retrieval, and index storage.
- `packager`: index building and capsule packaging/inspection.
- `runtime`: Ollama generation, capsule execution, and readiness checks.
- `telemetry`: privacy-preserving local event writing and aggregation.
- `eval`: retrieval datasets, metrics, and evaluation runner.
- `planner`: reserved for later planning behavior.
- `cli`: Typer command-line interface.
