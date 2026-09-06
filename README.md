# Capability Capsule

Capability Capsule prepares task-specific local resources for meaningful work while offline. This
repository currently contains only the V0.1 engineering skeleton: packaging metadata, configuration,
module boundaries, a minimal CLI, and environment checks. Scanner, planner, RAG, runtime, telemetry,
evaluation, and packaging behavior are intentionally not implemented yet.

## Development environment

The supported development environment is Ubuntu on WSL2, with the repository mounted at
`/mnt/e/capsule`. Python is managed per project by uv; do not install dependencies into the Windows
or WSL system Python.

```bash
cd /mnt/e/capsule
uv sync --all-groups
uv run capsule --help
uv run pytest
uv run ruff check .
uv run mypy
```

The project targets Python 3.13. The `.python-version` file lets uv select or provision a compatible
managed interpreter without changing the global Python installation.

## Configuration

Copy `config.example.toml` to `config.toml` for local overrides. The local file is ignored by Git.
The example uses the Windows Ollama endpoint and models already available on the development host.

WSL may inherit proxy variables. Keep `http.trust_env = false` when connecting to the localhost
Ollama API so HTTP clients bypass those proxies. Future Ollama clients must pass this setting to
`httpx.Client` or `httpx.AsyncClient`; the current skeleton does not yet implement an Ollama client.

## Package layout

All Python packages live under `src/capability_capsule`. The `scanner`, `planner`, `rag`, `runtime`,
`telemetry`, `eval`, and `packager` packages are reserved boundaries for future V0.1 work. The `cli`
package currently exposes only application metadata and help.

Runtime session output will use `.capsule/` and is excluded from source control.
