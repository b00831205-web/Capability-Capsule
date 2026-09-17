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
- Expand the MVP runtime benchmark into a cross-device, cross-runtime performance catalog after the
  first measured harness-specific budget works.
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
- Optimize deadline-aware selection between local training, remote GPU training, adapter reuse, and
  an untrained local-model fallback after the MVP implements one conservative readiness estimate.
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
- Cross-harness shared representations or a single multi-harness adapter; the MVP keeps alignment
  publications separate unless compatibility is measured.
- Behavioral models of user reading, thinking, and manual editing time. MVP coverage safely sets user
  delay to zero instead of requiring personal behavior tracking.

## Deferred Capsule and Codex App integration work

- Multi-model routing.
- Dynamic fallback from 0.8B to 2B.
- Speculative decoding.
- Long-context memory management beyond the now-measured 32K MVP context budget.
- Broader automatic discovery and compatibility checks beyond the first supported Codex, Claude Code,
  and DeepSeek harness profiles and their selected local providers.
- Automatic migration of versioned harness profiles across future app, CLI, provider, and
  configuration-schema revisions.
- Verified switching of an already-running Codex App task between cloud and local Capsule models.
- App-native build progress, readiness, validation, and "switch to offline Capsule" controls.
- Automatic creation of a new offline handoff task when safe in-place model switching is unavailable.
- Preservation of task context, relevant files, current plan, validator state, and working-tree state
  in the offline handoff without copying secrets or unrelated user data.
- Stable app-server integration after its current experimental protocol is suitable for production
  use; the MVP uses verified CLI/App configuration injection instead.
- Additional existing harness adapters behind the same Capsule launch contract after the initial
  Codex, Claude Code, and DeepSeek targets establish the profile interface.
- Local model-service background lifecycle management, health monitoring, crash recovery, and clean
  shutdown.
- Provider-specific adapter loading when supported, avoiding a merge when it is unnecessary.
- Automatic adapter merging and multiple quantization exports.
- Cross-platform packaging beyond the first WSL deployment target.
- Self-update and capsule migration mechanisms.

### Promoted blocking integration work

The following is no longer a deferred optimization because the real harness run proved it blocks
correctness:

- Add an explicit CLI harness choice and versioned `HarnessProfile`; never hard-code Codex into the
  Teacher dataset or silently choose a detected app.
- Maintain separate harness-alignment publications containing the actual Student-visible tool schema,
  prompt envelope, shell semantics, command-result envelope, and multi-turn behavior.
- Preserve at least a 32K local-provider context for Codex; the earlier 8K smoke setting is too small
  for the complete tool and safety prompt.
- Add harness-aligned Teacher trajectories that use the actual Codex `exec_command` schema, Windows
  command-result envelope, and multi-turn prompt shape. Direct adapter generation is not a substitute
  for this end-to-end evidence.
- Decide from measured validation whether to keep Qwen3.5-2B as a narrowly scripted executor or move
  the primary coding capsule to the next larger model that fits the confirmed hardware budget.
- Benchmark the exact model, quantization, runtime, harness, and deployment device. Keep measured
  prefill throughput separate from decode throughput and include tool, validation, retry, memory, and
  sustained-performance costs.
- Produce two workload estimates: conservative-slow readiness ETA and fastest-sustainable offline
  consumption capacity with user delay set to zero.
- Derive the knowledge-tree boundary from maximum task cycles, task classes, project areas, tool
  calls, and validation demand. Output-token count is supporting evidence, not the boundary itself.
- Require runtime endurance, work-capacity, and knowledge-coverage evidence before claiming that a
  capsule is provisioned for the requested offline duration.

The first `HarnessProfile` implementation is intentionally narrow: one immutable Codex profile,
SHA-256 verification, one exact `exec_command` argument schema, and one result-envelope identifier.
Automatic installed-app discovery, profile migration, multiple shells, and schema negotiation remain
deferred until this contract passes and is connected to collection-plan validation.

### Resolved blocking configuration compatibility

- Qwen3.5 tokenizers whose chat template omits `{% generation %}` no longer depend on a native
  assistant mask for SFT export. The minimal fallback derives trainable spans from tokenized message
  prefixes; broader template-family compatibility and performance optimization remain deferred until
  another supported harness or model demonstrates a need.

## Deferred consumer experience work

- A one-sentence request such as "prepare an offline Capsule for this project before my flight" that
  resolves the project recipe, deadline, build strategy, and Codex App handoff automatically.
- A single high-impact authorization screen covering cloud Teacher use, remote training, disposable
  fixture execution, and the exclusion of secrets and locked test data.
- Readiness estimates and explicit fallback reporting when the preferred trained adapter cannot be
  completed before the offline deadline.
- A pre-departure offline rehearsal that disables network access, opens the Capsule in Codex App,
  executes a representative tool-using validation task, and reports whether the package is ready.
- Rich per-user behavior forecasting after the model-throughput upper-bound approach has been
  validated; personal editing-speed measurement is not required for MVP correctness.
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
