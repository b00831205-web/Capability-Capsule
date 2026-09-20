# Capability Capsule MVP Roadmap

## Objective

Prove one minimal, reproducible path from a Teacher model to an offline capsule that lets a user
continue a real project through a selected existing coding harness while offline:

```text
Teacher model
  -> capsule-teacher skill
  -> tools read Student and task configuration
  -> Teacher executes authorized fixture tasks
  -> validated training trajectories
  -> SFT/LoRA training tool
  -> trained adapter/checkpoint
  -> evaluated and recorded capsule model
  -> quantized capsule package and local inference service
  -> CLI injects the capsule model configuration into the selected harness
  -> Codex, Claude Code, DeepSeek, or another supported harness executes tools
  -> usable offline task result
```

The MVP targets one project-scoped task distribution. It does not attempt general equivalence to a
frontier model.

## North-star user experience

The consumer supplies a project goal and an offline deadline, not fixture, trajectory, training, or
harness configuration. For example:

```text
I will be on a plane for the next 12 hours. Prepare a capsule with which I can continue developing
this project offline in my selected coding app.
```

Developer-maintained project recipes and fixture catalogs resolve the authorized train and
validation inputs. During provisioning, the CLI discovers supported installed harnesses and asks the
consumer to choose one when it was not supplied explicitly. The build system creates the collection
plan, harness-aligned trajectories, dataset, adapter, runtime configuration, and offline handoff. The
user continues to interact in the selected coding app; CLI commands are an internal provisioning and
model-injection mechanism, not the user interface.

A capsule is the complete project-scoped execution package, not only model weights:

```text
base model or local model reference
  + adapter or merged/quantized model
  + project knowledge and handoff state
  + prompt and tool policy
  + local inference-service configuration
  + selected-harness launch/injection configuration
  + validators and evaluation evidence
```

## Non-negotiable route guardrails

- Finish the complete loop before adding multi-model routing, automatic hyperparameter search, rich
  interfaces, or broad hardware support.
- Use only authorized `train` and `validation` fixtures for Teacher collection and tuning.
- Never expose the locked `test` split to the Teacher, trainer, model selector, or prompt builder.
- Record only observable Teacher messages, real tool calls, tool results, patches, and validation.
- Every published input and output must have a stable identifier and content digest.
- A failed validator or evaluation remains a failed record; no component may rewrite it as success.
- Reuse the selected existing harness, tool loop, sandbox, approvals, and task UI. Do not build a
  parallel agent harness for the MVP.
- Treat CLI and app-server integration as provisioning and configuration channels. Users interact
  with the capsule through Codex App, not through a training or harness CLI.
- Keep model serving separate from the harness: the local runtime loads the adapter or quantized
  model, while the selected app receives only its versioned harness profile, local provider, and
  model configuration.

## Harness target strategy

The harness is an explicit build target, not an assumption embedded in Teacher prompts or model
weights. The first provisioning command accepts a target such as `codex`, `claude-code`, or
`deepseek`; when omitted, the CLI detects supported installed harnesses and asks the user to choose.
It must not silently select one.

Each supported target has a versioned `HarnessProfile` containing at least:

- harness ID and pinned version;
- provider protocol and model-injection method;
- exact Student-visible tool schemas and allowed-tool policy;
- prompt/template digest and fixed context overhead;
- shell and filesystem semantics;
- tool-result envelope, approval, sandbox, and multi-turn behavior;
- parallel-task capability and harness-specific validators.

Teacher execution tools and Student-visible tools are separate. A Teacher may use its own tools to
prepare and validate fixtures, but a Student trajectory may contain only tools exposed by the chosen
`HarnessProfile`. Core project knowledge may be shared, while harness-alignment datasets and adapters
remain separate for the MVP unless cross-harness compatibility is independently demonstrated.

## Offline-duration workload budget

`--offline-for 12h` is a capacity requirement, not merely a service uptime setting. Before sizing the
knowledge tree, benchmark the exact model, quantization, runtime, harness profile, and deployment
device. Hardware-spec estimates are provisional; a capsule is ready only after measured evidence is
available.

Measure at least cold start, short- and long-context prefill throughput, decode throughput, first-tool
latency, complete edit-and-validate cycle time, retry rate, peak memory, and sustained-performance
degradation. Model one task class as:

```text
cycle_time = input_tokens / prefill_tokens_per_second
           + output_tokens / decode_tokens_per_second
           + tool_time
           + validation_time
           + retry_probability * recovery_time
```

Maintain two deliberately different estimates:

- readiness/build ETA uses a conservative slow sustained rate so the capsule can finish before the
  user's deadline;
- offline coverage uses the fastest measured sustainable task-cycle rate with user delay set to zero,
  producing an upper bound on how much work the model could consume during the offline interval.

Do not assume that a person is always slower than the model for every edit. Setting user delay to zero
already gives the safe maximum-consumption bound for a single foreground interaction loop. Convert
that bound into task-class counts, tool calls, validations, affected project areas, and knowledge
nodes; raw output-token count alone does not define knowledge-tree size. Add an explicit reserve and
report assumptions rather than promising autonomous useful work solely from tokens per second.

## MVP model strategy

1. Use Qwen3.5-0.8B for the first very small pipeline smoke run because it minimizes training and
   conversion cost.
2. Use Qwen3.5-2B as the first primary capability candidate after the smoke loop works.
3. Keep both models in single-model executor mode for MVP. Their initial roles are `executor` and
   `coder`.
4. Run deployment inference under the confirmed WSL hardware profile with at least a 32K context.
   An 8K context was measured as insufficient for the complete Codex tool and safety prompt.
5. Prefer a GPU host for useful LoRA training. A CPU run is acceptable only to prove that the trainer
   starts, consumes the dataset, saves a checkpoint, and exits; it is not a performance experiment.

## Stage 0: Freeze only blocking inputs

Required inputs:

- confirmed hardware profile;
- provisional base-model recommendation;
- selected harness ID and a pinned `HarnessProfile`;
- requested offline duration and build-ready deadline;
- measured or explicitly provisional runtime benchmark;
- derived offline work budget and knowledge-tree coverage boundary;
- exact Teacher model and skill version;
- one fixture family with train and validation tasks;
- explicit allowed tools and validators;
- raw and curated dataset destinations;
- training output directory and random seed.
- pinned selected-app/CLI version and one verified local-provider injection method;
- an offline handoff destination for the project task state.

Exit condition:

- configuration can be represented by existing schemas and no unresolved value prevents one smoke
  trajectory or one trainer invocation.

All non-blocking improvements belong in `PENDING_CONFIGURATION_OPTIMIZATIONS.md`.

## Stage 1: Generate the smoke training set

Minimum scale:

- 8 to 20 `train` trajectories;
- 2 to 5 separate `validation` trajectories;
- at least one real tool-using coding task from start to independent validation.

Execution:

1. Build a `TeacherCollectionPlan` whose assignments reference the accepted `StudentTarget`,
   `HarnessProfile`, and offline work budget.
2. Render each assignment with `render_teacher_prompt`.
3. Invoke `$capsule-teacher` with the immutable assignment JSON.
4. Let the Teacher use real tools only inside the authorized fixture, while recording Student tool
   calls exclusively in the schema and shell semantics exposed by the selected harness.
5. Validate and append each result with `append_teacher_trajectory`.
6. Resume interrupted collection with `pending_teacher_assignments`.

Artifacts:

- collection plan and publication manifest;
- append-only raw Teacher JSONL;
- validator evidence for every successful trajectory.

Exit condition:

- one complete Teacher trajectory passes schema, provenance, tool-policy, fixture, and validator checks.
- its serialized prompt, tool requests, and observable results also pass target-harness replay or an
  equivalent schema-and-envelope validator.

## Stage 2: Curate and export SFT data

Execution:

1. Run `curate_teacher_dataset` to deduplicate and partition trajectories.
2. Run `validate_teacher_dataset` and the existing dataset-integrity checks.
3. Publish immutable train and validation artifacts with `publish_teacher_dataset`.
4. Export each trajectory to the Qwen chat/tool template expected by the trainer.
5. Train only on Assistant outputs and structured tool requests. User messages and observable tool
   results remain model inputs rather than prediction targets.

Artifacts:

- curated dataset publication;
- dataset ID and digest;
- SFT train and validation files;
- discard and duplicate report.

Exit condition:

- one exported example can be tokenized, decoded, and matched back to its source trajectory without
  exposing hidden reasoning or locked-test content.

## Stage 3: Implement the minimal trainer

The first trainer needs only:

- Qwen model and tokenizer loading;
- SFT data loading;
- LoRA attachment;
- Assistant/tool-call loss masking;
- one training loop with validation loss;
- checkpoint and adapter saving;
- deterministic seed;
- JSON/JSONL process logging;
- non-zero exit on invalid data, out-of-memory, or failed checkpoint save.

Do not add hyperparameter search, distributed training, automatic model routing, or production job
orchestration before this trainer completes one smoke run.

Suggested smoke settings:

- model: Qwen3.5-0.8B;
- dataset: the Stage 1 smoke set;
- epochs: 1;
- sequence length: start at 2K or 4K;
- LoRA rank: one fixed conservative value;
- optimization steps: enough to prove loss, backward pass, checkpoint save, and reload;
- output: one adapter checkpoint plus tokenizer and run metadata.

Exit condition:

- a saved adapter can be reloaded with the exact base-model revision and produce a response.

## Stage 4: Record training and evaluation evidence

Recording is part of training, not a later reporting task.

Before training:

1. Call `build_training_dataset_stats` to record trajectory count, serialized bytes, messages, tool
   calls, categories, tokenizer ID, and exact tokens when available.
2. Create and persist a `TrainingRunManifest` containing run ID, capsule ID, dataset ID/digest, base
   model, trainer ID, hardware ID, seed, start time, and hyperparameters.

During training:

- append step number, epoch, training loss, validation loss, learning rate, elapsed time, and checkpoint
  path to an append-only trainer process log;
- preserve the trainer's raw terminal log;
- never overwrite a completed run directory.

At every evaluated checkpoint:

1. Produce `CaseResult` records containing status, score, `duration_ms`, input/output tokens,
   `peak_rss_mb`, tool-call count, and invalid-tool-call count.
2. Call `summarize_training_evaluation` to aggregate success, time-bounded success, score, duration,
   memory, and token metrics.
3. Call `record_training_checkpoint`, or the higher-level `process_training_checkpoint_completion` /
   `process_training_checkpoint_files`, to append the learning-curve point.
4. Preserve knowledge coverage with `write_knowledge_usage_event` and pass its summary into checkpoint
   recording when retrieval is exercised.
5. Record capsule runtime latency and status with `write_run_telemetry` during harness evaluation.
6. Compare measured task-cycle throughput with the offline work budget and update its evidence status
   without rewriting the original estimate.

Required artifacts:

- `training-run.json`;
- append-only trainer process JSONL;
- checkpoint directories;
- evaluation `CaseResult` JSONL;
- append-only learning-curve ledger;
- runtime and knowledge-usage telemetry.

Exit condition:

- the first checkpoint appears in the learning-curve ledger and can be traced back to the exact dataset,
  base model, trainer configuration, hardware profile, and evaluation suite.

## Stage 5: Build the first capsule runtime

Execution:

1. Reload the trained LoRA adapter and verify one held-out validation prompt.
2. Merge or otherwise package the adapter with its pinned base model when required by the selected
   local inference runtime.
3. Convert the deployable result to the selected GGUF format when required.
4. Quantize to the provisional Q4 target.
5. Register the result with one local provider supported by Codex App, initially Ollama or LM Studio.
6. Record hashes for the base model, adapter, merged model, quantized model, tokenizer, prompt template,
   tool schema, runtime revision, and provider configuration.
7. Package these artifacts with the capsule manifest, project knowledge assets, and offline handoff.

Exit condition:

- the local provider loads the capsule offline under WSL and returns a valid response without
  accessing the network.

## Stage 6: Inject the capsule into the selected harness

The selected coding app and its existing harness remain responsible for the agent loop:

```text
start local capsule provider
  -> generate a capsule-specific configuration from its HarnessProfile
  -> invoke the selected CLI/App configuration channel
  -> open or create the offline handoff task in Codex App
  -> user interacts with the capsule through the normal App UI
  -> selected harness validates and executes tool requests
  -> an independent evaluator runs the task validator
  -> evaluation records CaseResult and telemetry
```

The Capsule implementation must not duplicate the selected harness's agent loop, tool dispatcher, sandbox, approval
UI, or task interface. It implements only:

- local model-service lifecycle and health checks;
- harness-specific model/provider configuration generation;
- CLI-driven App launch or configuration injection;
- offline task handoff generation;
- structured event capture for automated checkpoint evaluation;
- independent validator execution and evidence recording.

For a Codex target, use `codex exec --json` only for automated checkpoint evaluation of the same
capsule configuration. Equivalent non-interactive entry points may be used for other harnesses. They
are not the consumer interaction surface; the consumer works in the selected app.

Required integration boundaries:

- workspace-root confinement;
- the selected harness's sandbox and approval policy remain active;
- no network in offline mode;
- observable tool calls and results;
- validator result independent from the model's final claim.
- no persistent mutation of unrelated user Codex configuration when an invocation-scoped or
  capsule-specific profile is sufficient;
- no claim of hot-swapping an already-running App task until that behavior is verified against the
  pinned App/CLI version.

Exit condition:

- the selected app opens an offline handoff task using the local capsule model; one unseen validation task
  causes at least one valid tool call through the existing Codex harness, produces the expected
  project change or answer, passes its independent validator, and writes evaluation evidence.

## Stage 7: Promote from smoke to useful 2B pilot

Only after Stages 1 through 6 pass:

1. Increase the curated dataset to roughly 100 to 300 high-quality trajectories.
2. Train one Qwen3.5-2B LoRA configuration for one epoch.
3. Evaluate unchanged validation cases at baseline and post-training checkpoints.
4. Compare success rate, tool validity, duration, memory, tokens, knowledge coverage, and regression.
5. Add data only while the learning curve is improving; do not assume more trajectories are better.

Exit condition:

- the 2B capsule beats its untrained baseline on the fixed validation suite without violating the WSL
  memory budget or regressing below the agreed safety thresholds.

## Current smoke evidence

- The Teacher-to-SFT-to-LoRA-to-checkpoint-to-reload-to-GGUF path completes for Qwen3.5-2B.
- The pinned Qwen3.5 chat template has no native `{% generation %}` assistant mask. The SFT exporter
  now falls back to message-prefix tokenization; a real harness-aligned sample exported 462 tokens
  with 295 assistant/tool-call tokens selected for training, and tests cover both missing and all-zero
  tokenizer mask responses.
- The 32-step exec-only run reduced held-out validation loss from `1.378` to `0.799`; its adapter
  reload emitted a valid first `exec_command` call.
- The quantized Q4_K_M artifact loads in LM Studio with a 32K context and is callable through
  `codex exec --oss --local-provider lmstudio`.
- The unseen greeting fixture still fails under the complete Codex harness: the 2B model inspects the
  correct file but cannot reliably construct a valid Windows edit command. The fixture remains
  unchanged and the failure is recorded rather than promoted.
- The existing `smoke-pilot-2b-exec-v2` publication is exec-only training data, not a verified Codex
  `HarnessProfile`: it records the `exec_command` name and `cmd` argument but uses simplified result
  text and does not pin the complete Codex prompt, tool schema, or result envelope.
- `HarnessProfileReference` is now implemented end to end: prompt rendering, new trajectory append,
  resumed-collection validation, tool arguments, and normalized tool-result envelopes all verify the
  same digest-pinned profile. The focused suite passes 49 tests without skips.
- The observable Codex `0.154.0-alpha.6.2` contract snapshot records the full MVP `exec_command`
  schema and a conservative measured prompt cost of 9,916 tokens. Two minimal local-provider calls
  were correctly rejected by their 8,192-token contexts, preserving the measured failure evidence
  and confirming the 32K deployment target. The snapshot does not claim to copy Codex's hidden
  system prompt.
- `smoke-pilot-003` is the first formally published and collected schema `0.3` plan. Its manifest
  pins the exact plan bytes and digest, while its one validation assignment resolves the pinned
  Student and HarnessProfile and embeds the verified contract in the Teacher prompt. The observable
  23-message trajectory contains real `exec_command` requests, result envelopes, failure recovery,
  and passing `pytest` plus exact-content evidence; independent reload reports one record and zero
  pending assignments. This is a validation smoke result, not the required Stage 1 dataset.
- `stage1-codex-001` now publishes the complete minimum Stage 1 assignment set: 8 train and 2
  validation trajectories, all pinned to the accepted Student and Codex HarnessProfile. Collection
  completed in disposable workspaces without modifying the registered fixtures. Independent curation
  retained all 10 unique trajectories with zero duplicates, and the immutable publication records
  8 train plus 2 validation artifacts with verified byte counts and SHA-256 digests.
- The canonical Qwen export is `stage1-codex-001-qwen35-2b-v2`, pinned to the accepted Qwen3.5-2B
  tokenizer revision with a 4K sequence budget. It preserves all 10 trajectories without truncation:
  7,537 total tokens, 6,087 trainable assistant/tool-call tokens, and a measured maximum sequence of
  2,167 tokens. The earlier 2K `v1` export is retained only as immutable truncation evidence.
- `stage1-codex-qwen35-2b-001` consumed that canonical export for one epoch and 8 optimization steps
  with seed 42, LoRA rank 8, and batch size 1. The completed recovery attempt recorded training loss
  `1.509` and held-out validation loss `1.087`, saved a 1,674,328-byte safetensors adapter, verified
  its checkpoint hashes, and reloaded it against base revision
  `15852e8c16360a2fea060d615a32b45270f8a8fc` as `PeftModelForCausalLM`.
- The initial attempt validated both examples after every step and caused the CPU-only WSL VM to
  restart after step 3. Its terminal log and explicit failure event are retained. The recovery
  restarted from the pinned base and seed, ran all 8 steps, and evaluated once at the end; this is
  checkpoint-production evidence, not a throughput benchmark or proof of validation-task quality.
- The unchanged `stage1-codex-validation-v1` checkpoint suite now runs through a bounded coding
  evaluator over revision-verified disposable fixture copies. It records real model turns, normalized
  result envelopes, independent validators, tokens, duration, peak RSS, invalid tool calls, and a
  digest-pinned learning-curve point. The step-8 adapter scored `0/2`: after valid reads it repeatedly
  emitted Unix `sed -i` or Bash heredoc edits under the pinned Windows PowerShell harness, reached the
  four-tool-round limit, and left both fixtures failing. Peak RSS was about 4.68 GB; case durations
  were 56.3 and 93.6 seconds. This is a measured harness-alignment failure, so this checkpoint must
  not be promoted or packaged as the MVP capsule.
- `stage1-codex-powershell-002` now contains 8 additional train trajectories collected from
  revision-verified disposable copies. Every trajectory uses guarded Windows PowerShell reads and
  edits, records the real missing-`pytest` failure from the Windows interpreter, recovers through the
  repository's pinned WSL validator, and passes its independent fixture tests. The schema `0.3` plan
  and all tool requests/results validate against the same digest-pinned Codex HarnessProfile; the
  original fixtures and all earlier raw or published datasets remain unchanged.
- The combined immutable `stage1-codex-powershell-002` publication contains 16 train and 2 validation
  trajectories with zero duplicates. Its `stage1-codex-powershell-002-qwen35-2b-v1` export preserves
  all 18 trajectories without truncation: 16,888 total tokens, 13,344 trainable tokens, and a measured
  maximum sequence of 2,167 tokens under the same pinned 4K Qwen tokenizer contract.
- The new training attempts preserve their failures rather than claiming a checkpoint. The first
  attempt still evaluated every step and was stopped after step 2. `recovery-001` completed all 16
  optimization steps with training loss `1.268` but the WSL process terminated during final
  validation before saving an adapter. Two save-first retries then terminated before model loading,
  despite the VM later reporting about 14 GiB available. No new checkpoint exists yet.
- `stage1-codex-validation-v2` preserves the two fixed v1 case models exactly and adds
  `validation-powershell-salutation-unseen-001`, revision
  `bdece654c046d50688c9c3a33eb556c9d728ada54487ba289301e92cb5c0fd0a`, which appears in no Teacher
  publication, SFT export, or training loss. The three-case suite digest is
  `3e756195c1585c57c4dcc8a3fef40cb2653a67bc57820c6156602f357301b9bb`.
- The next blocking step is an explicitly authorized WSL restart, followed by the save-first training
  recovery, separate-process validation-loss measurement, adapter reload, and execution of the v2
  suite. The current two fixed validation trajectories contributed validation loss during training,
  so only the third case can support a genuinely unseen generalization claim.

## MVP completion definition

The MVP is complete only when all of the following are true:

- Teacher collection was driven by `$capsule-teacher` and immutable assignments;
- real tool calls produced a validated train/validation dataset;
- the trainer consumed the published dataset and saved a reloadable LoRA checkpoint;
- training configuration and process metrics were persisted;
- checkpoint evaluation was appended through the existing learning-curve functions;
- a quantized capsule ran offline;
- the explicitly selected app used the injected local capsule through its existing harness on an
  unseen validation task;
- the runtime benchmark and offline work budget show enough measured capacity and knowledge coverage
  for the requested offline interval under their recorded assumptions;
- the task validator passed and telemetry was written;
- every artifact can be traced by IDs, revisions, and digests.

## Codex collaboration rule

- Without explicit user authorization, Codex may edit only files under `tests/`.
- For implementation or script changes without authorization, Codex must output complete code blocks
  and the user applies them.
- Read-only inspection and test execution remain allowed for diagnosis and verification.
- Any write outside `tests/` requires explicit authorization naming the intended scope.
