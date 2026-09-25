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
- Replace the one-off Stage 1 recovery override with a persisted, configurable evaluation cadence.
  On the CPU-only profile, validating the 2,167-token example after every optimization step restarted
  WSL after step 3; one validation pass after the final step completed the full 8-step run. Keep the
  per-step policy only for hardware that passes an explicit memory check.
- Distributed and multi-GPU training.

## Deferred dataset optimization work

- Materialize disposable fixtures directly from their pinned snapshots with byte-preserving line
  endings. The first schema `0.3` smoke collection exposed both a stale working-tree trailing LF and
  Windows Git CRLF conversion; collection recovered without touching the registered fixture, but
  larger batches should not require manual snapshot reconstruction or line-ending normalization.
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

- Add explicit CLI harness discovery and choice beyond the now-pinned Codex MVP profile; never
  silently choose a detected app.
- Maintain separate harness-alignment publications containing the actual Student-visible tool schema,
  prompt envelope, shell semantics, command-result envelope, and multi-turn behavior.
- Preserve at least a 32K local-provider context for Codex; the earlier 8K smoke setting is too small
  for the complete tool and safety prompt.
- Block promotion of `stage1-codex-qwen35-2b-001`. Its digest-pinned two-case evaluation scored
  `0/2`: the adapter repeatedly generated Unix `sed -i` and Bash heredoc commands for a Windows
  PowerShell harness, accumulated 1 and 3 invalid tool calls, and reached the four-round limit.
- The separately versioned `stage1-codex-powershell-002` increment is published as a combined
  16-train/2-validation dataset and exported without truncation as
  `stage1-codex-powershell-002-qwen35-2b-v1`. After restart, save-first recovery `004` trained 16
  steps, saved and independently reloaded its adapter, and measured validation loss `1.022`; it then
  scored `0/3` on fixed-plus-unseen suite digest
  `3e756195c1585c57c4dcc8a3fef40cb2653a67bc57820c6156602f357301b9bb`.
  Add a new data increment for inspect→PowerShell-edit→validate transitions, but never add that suite
  or its execution results to training. Do not loosen the evaluator to accept Unix commands merely to
  improve the score.
- Decide from measured validation whether to keep Qwen3.5-2B as a narrowly scripted executor or move
  the primary coding capsule to the next larger model that fits the confirmed hardware budget.
- Keep both contract checkpoints blocked from publication. The controlled 4-epoch / 32-step run held
  data, model, seed, learning rate, and LoRA configuration fixed and still scored `0/3` on unchanged
  v2. It produced zero tool calls because schema `0.2` encodes narration and its following tool call
  as separate assistant turns; the model learned the narration-only turn plus its end marker. Promote
  SFT turn normalization back into blocking work: coalesce adjacent assistant narration and tool-call
  records into one assistant turn, assert decoded trainable tokens contain narration followed by the
  tool call before one `<|im_end|>`, export a new immutable SFT version, and only then retrain. Do not
  collect more data, increase steps again, expose v2 fixtures, or change model size before this
  serialization defect is tested.
- Validate schema `0.3` assistant-turn normalization across all 8 `contract-004` trajectories.
  Every narration immediately followed by a tool call must decode as one assistant turn containing
  both narration and `<tool_call>` before a single `<|im_end|>`. Preserve user/tool-result boundaries,
  export to a new immutable SFT directory, and do not start another LoRA run until this invariant
  passes for the complete export.
- The schema `0.3` gate now passes for `stage1-codex-powershell-contract-004-qwen35-2b-v2`: all 8
  trajectories retain their 25 trainable tool calls, the decoded narration/tool-call boundaries are
  coalesced, and no example reaches the 4K limit. Next train a fresh adapter from this exact export
  and evaluate it on the unchanged v2 fixed-plus-unseen suite. Keep the old schema `0.2` checkpoints
  as comparison evidence rather than promoting them based on training loss.
- The schema `0.3` checkpoint `stage1-codex-qwen35-2b-006-turns-v2` completed 32 steps, reloaded
  independently, and scored `0/3` on unchanged v2. It now reads files through the tool on every case,
  so the assistant-turn boundary defect is resolved in generated behavior. The remaining blocker is
  edit-command policy and correct target text: two cases repeatedly used rejected here-string edits;
  the unseen case made a permitted but wrong edit. Before any further training, compare emitted edit
  commands with the accepted `.Replace` + `Set-Content` trajectory pattern and check whether the
  current constrained evaluator and training examples agree on executable command forms. Preserve
  the fixed suite and genuinely unseen case as evaluation-only evidence.
- The generic v3 command-policy intervention did not clear the gate. Its initial `0/3` run exposed a
  simulator mismatch: valid `Get-Content -Raw greeting.py` was rejected solely because `-Raw` preceded
  the path. After accepting both parameter orders and adding strict tests that reject semantically
  different `.Replace`/`Set-Content` pipelines, the new-directory rerun remained `0/3` (`1, 2, 2`
  invalid calls). The model now reads all three files but produces an unsupported
  `Get-Content | Get-Content -Append | Set-Content` edit; two examples leak the tool-result envelope
  into the proposed file content. Do not add case-specific hints or increase epochs on this evidence.
  Prioritize varied, authorized inspect-to-guarded-edit-to-validate train trajectories with neutral
  target strings, plus separate held-out validation. Keep v2 and both v3 runs immutable. Because the
  suite digest covers cases but not the prompt, track the suite ID and prompt SHA-256 together when
  comparing these runs. The first fixture currently has one extra final newline; use only a pinned,
  revision-verified disposable source copy until that provenance discrepancy is resolved explicitly.
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
SHA-256 verification, one exact `exec_command` argument schema, and one normalized result-envelope
identifier. The contract now passes prompt, append, resume, request-schema, and result-envelope
validation. The schema `0.3` smoke trajectory passes independent validation, and the minimum Stage 1
dataset has completed one checkpoint-producing run plus one digest-pinned evaluation. That adapter
failed both executable cases because its edit syntax did not match Windows PowerShell. Automatic
installed-app discovery, profile migration, multiple shells, and schema negotiation remain deferred
until a new adapter passes the unchanged suite and a genuinely unseen validation fixture.

### Resolved blocking configuration compatibility

- The Codex `0.154.0-alpha.6.2` observable contract is pinned by digest and referenced by the first
  formally published schema `0.3` collection plan. Its publication manifest verifies the exact plan
  bytes before collection. A minimal real invocation measured 9,916 prompt tokens and proved the 8K
  configuration insufficient; the profile preserves this as observed evidence rather than claiming
  access to the hidden Codex system prompt.
- `stage1-codex-001` now has a verified immutable dataset publication containing 8 train and 2
  validation trajectories with zero duplicates. All tool requests and result envelopes pass the
  pinned HarnessProfile after publication reload; all registered fixture snapshots remain unchanged.
- `stage1-codex-powershell-002` has a digest-pinned schema `0.3` collection plan and 8 completed
  train trajectories. Its observable file operations are Windows PowerShell-native, its environment
  recovery is retained as real evidence, and all assignments report complete under the pinned
  HarnessProfile. The combined immutable publication has 16 train, 2 validation, and zero duplicate
  records; the 4K SFT export contains 16,888 tokens with a 2,167-token maximum. Recovery `004`
  produced a hash-verified step-16 adapter and separate-process validation loss `1.022`, but its
  fixed-plus-unseen v2 result is `0/3`, so it remains blocked from promotion.
- `stage1-codex-validation-v2` preserves both fixed v1 cases and adds one revision-pinned fixture that
  is absent from all Teacher and SFT data. Its three-case digest is
  `3e756195c1585c57c4dcc8a3fef40cb2653a67bc57820c6156602f357301b9bb`.
  The recovered checkpoint executed it with `0/3` success; its failure reports are retained as
  evidence for the next data increment.
- `stage1-codex-001-qwen35-2b-v2` is the verified, untruncated Qwen SFT export: 7,537 total tokens and
  6,087 trainable assistant/tool-call tokens at a 4K maximum sequence length. The 2K export clipped
  the 2,167-token recovery trajectory and is retained only for auditability.
- Qwen3.5 tokenizers whose chat template omits `{% generation %}` no longer depend on a native
  assistant mask for SFT export. The minimal fallback derives trainable spans from tokenized message
  prefixes; broader template-family compatibility and performance optimization remain deferred until
  another supported harness or model demonstrates a need.
- `stage1-codex-powershell-contract-004` is the published cmd-only contract increment: a digest-pinned
  schema `0.3` plan plus 8 train trajectories whose every tool request carries only a `cmd` argument
  and whose commands are the accepted Get-Content, guarded `.Replace` + Set-Content, and pytest
  sequence. One trajectory records a genuinely executed exit-126 rejection recovery, and the set
  excludes git apply, Copy-Item, WSL commands, prompt-level edit hints, and every v2 fixture or
  result. Its schema `0.2` Qwen export carries the identical evaluation contract and remains
  untruncated; `stage1-codex-qwen35-2b-004-contract` completed 8 save-first steps, saved verified
  adapter hashes, and passed an independent reload. The unchanged v2 suite then scored it `0/3`.
  Unlike earlier checkpoints, every generated tool request was structurally cmd-only, proving the
  SFT contract fix reached inference; however, generated command policy still used directory probes,
  Bash heredocs, and nested PowerShell invocations instead of the trained guarded edit form. Treat
  this as an optimization-exposure question first, not evidence that the 2B model is incapable.
- `stage1-codex-qwen35-2b-005-contract-32step` resolved that optimization-exposure question: training
  loss dropped to `0.4665`, yet every unchanged-v2 case emitted the exact narration-only first SFT
  turn and stopped without a tool call. The defect is now localized to adjacent assistant-turn
  serialization rather than insufficient steps or cmd-only schema alignment.

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
