# Pending Configuration Optimizations

This file records configuration work intentionally deferred until the first end-to-end MVP works.
Items here must not block the initial Teacher-to-capsule training loop unless they become a proven
correctness or safety issue.

## MVP rule

Use the smallest configuration that can complete one real trajectory, one SFT run, one evaluation,
one capsule build, and one harness-driven task. Move an item back into the MVP only when the current
pipeline cannot run or cannot produce trustworthy evidence without it.

## Deferred hardware-profile work

- Detect physical CPU cores under WSL instead of leaving the value `null`.
- Detect AMD integrated-GPU backends available to the actual WSL runtime.
- Distinguish reserved video memory from shared unified/system memory.
- Detect WSL memory and swap limits from both Linux and Windows configuration.
- Record runtime-specific instruction-set support and acceleration backends.
- Remove duplicate helper definitions and finish formatting/type cleanup in the hardware probe.
- Add explicit migration support for older hardware-profile schemas if profiles are published.

## Deferred base-model recommendation work

- Replace the two-model candidate list with a versioned model catalog.
- Verify GGUF artifact sizes and hashes instead of estimating quantized deployment feasibility.
- Pin the exact `llama.cpp` revision after the first local benchmark.
- Calibrate context, memory, throughput, and success thresholds from measured baselines.
- Add automatic rejection rules for unsupported prompt templates and tool-call formats.
- Compare Q4 variants and select quantization from measured quality, memory, and speed.
- Add separate recommendations for router, executor, coder, planner, retriever, and reviewer roles.
- Add a recommendation refresh policy when model revisions or deployment hardware change.

## Deferred training configuration work

- Automated batch-size and gradient-accumulation tuning.
- Automatic learning-rate search.
- Multiple LoRA ranks and target-module sweeps.
- QLoRA versus full-precision LoRA comparisons.
- Multiple random seeds and statistically powered repetitions.
- Automated early stopping based on learning-rate plateau metrics.
- Remote training-host discovery and scheduling.
- Deadline-aware selection between local training, remote GPU training, adapter reuse, and an
  untrained local-model fallback.
- Build-time estimation and a user-visible readiness deadline for travel or other upcoming offline
  periods.
- Resume/recovery across interrupted training jobs.
- Distributed and multi-GPU training.

## Deferred dataset optimization work

- Project-level Capsule recipes that let developers register fixture catalogs, allowed splits,
  validators, Student targets, and dataset destinations once.
- Consumer-facing fixture discovery so a user can request a smoke capsule without knowing fixture
  IDs, roots, trajectory IDs, or JSONL paths.
- Automatic selection of the smallest eligible train and validation fixture bundle for a requested
  project capability and offline deadline.
- One-time project authorization for executing registered tasks in disposable fixture copies while
  always excluding locked test data.
- Automatic difficulty scheduling for each Student parameter scale.
- Automatic task decomposition learned from Student failures.
- Coverage-aware Teacher sampling across the knowledge tree.
- Active learning driven by validation errors.
- Cross-Teacher comparison and trajectory arbitration.
- Semantic deduplication beyond the existing deterministic trajectory fingerprint.
- Automated data-volume selection from learning-curve saturation.
- Dataset balancing across every future Student role.

## Deferred Capsule and Codex App integration work

- Multi-model routing.
- Dynamic fallback from 0.8B to 2B.
- Speculative decoding.
- Long-context memory management beyond the MVP context budget.
- Automatic discovery and compatibility checks for Codex App, Codex CLI, Ollama, LM Studio, and
  other supported local providers.
- Versioned Codex App injection profiles with migration across App, CLI, and configuration-schema
  revisions.
- Verified switching of an already-running Codex App task between cloud and local Capsule models.
- App-native build progress, readiness, validation, and "switch to offline Capsule" controls.
- Automatic creation of a new offline handoff task when safe in-place model switching is unavailable.
- Preservation of task context, relevant files, current plan, validator state, and working-tree state
  in the offline handoff without copying secrets or unrelated user data.
- Stable app-server integration after its current experimental protocol is suitable for production
  use; the MVP uses verified CLI/App configuration injection instead.
- Optional DeepSeek or other existing harness adapters behind the same Capsule launch contract after
  the Codex App path works end to end.
- Local model-service background lifecycle management, health monitoring, crash recovery, and clean
  shutdown.
- Provider-specific adapter loading when supported, avoiding a merge when it is unnecessary.
- Automatic adapter merging and multiple quantization exports.
- Cross-platform packaging beyond the first WSL deployment target.
- Self-update and capsule migration mechanisms.

## Deferred consumer experience work

- A one-sentence request such as "prepare an offline Capsule for this project before my flight" that
  resolves the project recipe, deadline, build strategy, and Codex App handoff automatically.
- A single high-impact authorization screen covering cloud Teacher use, remote training, disposable
  fixture execution, and the exclusion of secrets and locked test data.
- Readiness estimates and explicit fallback reporting when the preferred trained adapter cannot be
  completed before the offline deadline.
- A pre-departure offline rehearsal that disables network access, opens the Capsule in Codex App,
  executes a representative tool-using validation task, and reports whether the package is ready.
- Recovery UX that keeps the best verified fallback usable when training, conversion, model serving,
  or Codex App injection fails.

## Promotion criteria

An item may leave this list only when at least one condition is true:

- it blocks the end-to-end MVP;
- a measured MVP failure identifies it as the root cause;
- it is required to preserve data integrity, safety, or reproducibility;
- the end-to-end MVP has passed and the next optimization experiment is explicitly authorized.

## Codex write boundary

- Without explicit user authorization, Codex may edit only files under `tests/`.
- For implementation or script changes without authorization, Codex must output complete code blocks
  for the user to apply manually.
- Read-only inspection and test execution are allowed when needed to diagnose or verify work.
- Writing generated artifacts, documentation, configuration, source, scripts, or repository metadata
  outside `tests/` requires explicit user authorization.
