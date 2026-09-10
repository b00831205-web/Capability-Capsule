# Capability Capsule

Capability Capsule is an Adapter-first system for building portable, task-specific capabilities for
small local models. A future lightweight capsule will combine a verified LoRA adapter with its base
model identity, runtime configuration, tool policy, evaluation evidence, and offline execution
harness. The base model remains installed on the host; changing tasks means changing capsules rather
than redistributing the full model.

The project is being migrated from a completed retrieval-based Flight MVP to this post-training
architecture. Existing RAG components remain available as an optional knowledge layer and as a
reproducible baseline, but they are no longer the product's central capability mechanism.

## Current status

Implemented and verified:

- Repository and document scanning with ignore, binary, size, and path-boundary handling.
- Ollama embeddings, local vector indexes, retrieval, and grounded question answering.
- Flight capsule packaging, inspection, readiness checks, execution, and local telemetry reports.
- Retrieval evaluation and real Ollama integration tests.
- A bounded read-only workspace agent with `read_file` and `search_text` tools.
- Host-controlled tool policy with allow, approval, explicit, and deny levels.
- CLI text and JSON output for the read-only agent.

Not implemented yet:

- Codex Teacher Skill and training-data generation workflow.
- Training-data validation, deduplication, splitting, and quality gates.
- LoRA or QLoRA training.
- Adapter conversion and quantized deployment validation.
- Writable tools, approval interaction, or Git mutation tools.
- Adapter-bearing capsule manifests and runtime activation.

No command in the current project downloads a model, modifies global Python, starts Docker, or runs
model training.

## Adapter-first target architecture

```text
Codex Teacher + project Skill
        |
        v
validated training trajectories
        |
        v
small non-thinking base model + LoRA
        |
        v
quantized base model + task adapter
        |
        v
lightweight capsule + controlled harness
```

The initial feasibility study will compare a 0.8B candidate with an approximately 1.5B/2B candidate.
The primary product metric is time-bounded task success, not token throughput alone. Quantized
deployment must remain within a pre-registered quality margin of the high-precision LoRA reference.

The experimental protocol, including cycle replay, device stratification, formulas, statistical
tests, and planned charts, is available at
[`docs/research/offline-capability-decay-experiment-protocol.pdf`](docs/research/offline-capability-decay-experiment-protocol.pdf).
The decisions that must remain visible across long Codex sessions are tracked in the editable
[`docs/research/experiment-decisions.md`](docs/research/experiment-decisions.md) record.

## Development environment

The supported development environment is Ubuntu on WSL2 with the repository mounted at
`/mnt/e/capsule`. Python and dependencies are managed by uv inside the project. Windows and WSL
global Python environments are not modified.

```bash
cd /mnt/e/capsule
uv sync --all-groups
uv run python --version
uv run capsule --help
```

The runtime project targets Python 3.13 and uses Hatchling. A future ML training environment may need
its own uv-managed Python version because PyTorch and trainer compatibility can differ from the
runtime stack; it must remain isolated from global Python.

## Ollama

Ollama runs on Windows and is accessed from WSL at `http://127.0.0.1:11434`. The current baseline
configuration uses:

- Generation: `qwen3.5:4b`
- Embedding: `nomic-embed-text`

WSL may inherit proxy variables. Keep `http.trust_env = false` so local Ollama requests bypass the
proxy. Capability Capsule does not install Ollama or automatically download missing models.

## Read-only agent preview

The current agent is a bounded Harness prototype. It can inspect a workspace but cannot modify files
or run commands.

```bash
uv run capsule agent \
  "Find the CLI entry point and explain how it is created." \
  --workspace /mnt/e/capsule
```

Use a host-controlled policy file and structured output when needed:

```bash
uv run capsule agent \
  "Locate the runtime model call." \
  --workspace /mnt/e/capsule \
  --policy /path/to/policy.toml \
  --max-tool-rounds 3 \
  --max-tool-calls 5 \
  --json
```

The policy is loaded from the host, never from a capsule, and can only tighten the built-in policy.

## Retrieval-based Flight MVP

The completed legacy baseline remains usable:

```text
repository -> pack -> inspect -> doctor -> run -> report
```

Build and validate a retrieval capsule:

```bash
uv run capsule pack \
  --repo /path/to/repository \
  --output /path/to/flight.zip \
  --task "answer repository questions during a flight" \
  --config config.toml

uv run capsule inspect /path/to/flight.zip
uv run capsule doctor /path/to/flight.zip
uv run capsule run /path/to/flight.zip "Where is the main entry point?"
uv run capsule report /path/to/.capsule/sessions
```

This legacy ZIP contains `manifest.json`, `settings.json`, and `index.npz`. It contains no adapter or
model weights. The format will not be silently treated as a post-training capsule.

## Configuration

Copy `config.example.toml` to the ignored local configuration file:

```bash
cp config.example.toml config.toml
```

Current settings reserve the Ollama endpoint, generation and embedding models, proxy inheritance,
capsule size budget, offline duration, RAG context limit, and telemetry output. Adapter identity,
training provenance, and experiment references will be added only after their schemas and validation
rules are implemented.

## Verification

Run the standard checks:

```bash
uv run pytest
uv run mypy src tests
uv run ruff check .
uv run ruff format --check .
```

Real Ollama integration tests are opt-in and never download models:

```bash
CAPSULE_RUN_OLLAMA=1 uv run pytest tests/test_flight_mvp_integration.py -q
```

## Package layout

All Python packages live under `src/capability_capsule`:

- `scanner`: bounded repository and document discovery.
- `rag`: optional chunking, embeddings, vector search, retrieval, and storage.
- `packager`: current Flight capsule building and inspection.
- `runtime`: Ollama calls, capsule execution, readiness, workspace tools, agent loop, and policy.
- `telemetry`: privacy-preserving local event writing and aggregation.
- `eval`: current retrieval evaluation and future post-training experiment analysis.
- `planner`: reserved for later planning behavior.
- `cli`: Typer command-line interface.

## Next milestones

1. Define Teacher trajectory, training dataset, experiment manifest, and per-case result schemas.
2. Create the Codex Teacher Skill and generate the first auditable dataset.
3. Lock independent validation fixtures and a short logical task cycle.
4. Measure untrained 0.8B and approximately 1.5B/2B controls.
5. Run small LoRA experiments in an isolated training environment.
6. Compare high-precision adapters with quantized deployment under a fixed total-memory budget.
7. Generate statistical summaries and experiment charts from immutable JSONL records.
8. Add adapter identity and activation to the lightweight capsule format.
