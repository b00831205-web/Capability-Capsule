# Capability Capsule MVP Roadmap

## Objective

Prove one minimal, reproducible path from a Teacher model to an offline capsule that lets a user
continue a real project through the Codex App while offline:

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
  -> CLI injects the capsule model configuration into Codex App
  -> Codex App uses its existing harness to execute tools
  -> usable offline task result
```

The MVP targets one project-scoped task distribution. It does not attempt general equivalence to a
frontier model.

## North-star user experience

The consumer supplies a project goal and an offline deadline, not fixture, trajectory, training, or
harness configuration. For example:

```text
I will be on a plane for the next 12 hours. Prepare a capsule with which I can continue developing
this project offline in Codex App.
```

Developer-maintained project recipes and fixture catalogs resolve the authorized train and
validation inputs. The build system creates the collection plan, trajectories, dataset, adapter,
runtime configuration, and offline handoff. The user continues to interact in Codex App; CLI
commands are an internal provisioning and model-injection mechanism, not the user interface.

A capsule is the complete project-scoped execution package, not only model weights:

```text
base model or local model reference
  + adapter or merged/quantized model
  + project knowledge and handoff state
  + prompt and tool policy
  + local inference-service configuration
  + Codex App launch/injection configuration
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
- Reuse the Codex App harness, tool loop, sandbox, approvals, and task UI. Do not build a parallel
  agent harness for the MVP.
- Treat CLI and app-server integration as provisioning and configuration channels. Users interact
  with the capsule through Codex App, not through a training or harness CLI.
- Keep model serving separate from the harness: the local runtime loads the adapter or quantized
  model, while Codex App receives only a local provider and model configuration.

## MVP model strategy

1. Use Qwen3.5-0.8B for the first very small pipeline smoke run because it minimizes training and
   conversion cost.
2. Use Qwen3.5-2B as the first primary capability candidate after the smoke loop works.
3. Keep both models in single-model executor mode for MVP. Their initial roles are `executor` and
   `coder`.
4. Run deployment inference under the confirmed WSL hardware profile with an initial 8K context.
5. Prefer a GPU host for useful LoRA training. A CPU run is acceptable only to prove that the trainer
   starts, consumes the dataset, saves a checkpoint, and exits; it is not a performance experiment.

## Stage 0: Freeze only blocking inputs

Required inputs:

- confirmed hardware profile;
- provisional base-model recommendation;
- exact Teacher model and skill version;
- one fixture family with train and validation tasks;
- explicit allowed tools and validators;
- raw and curated dataset destinations;
- training output directory and random seed.
- pinned Codex App/CLI version and one verified local-provider injection method;
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

1. Build a `TeacherCollectionPlan` whose assignments reference the accepted `StudentTarget`.
2. Render each assignment with `render_teacher_prompt`.
3. Invoke `$capsule-teacher` with the immutable assignment JSON.
4. Let the Teacher use real tools only inside the authorized fixture.
5. Validate and append each result with `append_teacher_trajectory`.
6. Resume interrupted collection with `pending_teacher_assignments`.

Artifacts:

- collection plan and publication manifest;
- append-only raw Teacher JSONL;
- validator evidence for every successful trajectory.

Exit condition:

- one complete Teacher trajectory passes schema, provenance, tool-policy, fixture, and validator checks.

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

## Stage 6: Inject the capsule into Codex App

The Codex App and its existing harness remain responsible for the agent loop:

```text
start local capsule provider
  -> generate a capsule-specific Codex configuration
  -> invoke the Codex CLI/App configuration channel
  -> open or create the offline handoff task in Codex App
  -> user interacts with the capsule through the normal App UI
  -> Codex harness validates and executes tool requests
  -> an independent evaluator runs the task validator
  -> evaluation records CaseResult and telemetry
```

The Capsule implementation must not duplicate Codex's agent loop, tool dispatcher, sandbox, approval
UI, or task interface. It implements only:

- local model-service lifecycle and health checks;
- Codex model/provider configuration generation;
- CLI-driven App launch or configuration injection;
- offline task handoff generation;
- structured event capture for automated checkpoint evaluation;
- independent validator execution and evidence recording.

Use `codex exec --json` only for automated checkpoint evaluation of the same capsule configuration.
It is not the consumer interaction surface. The consumer always works in Codex App.

Required integration boundaries:

- workspace-root confinement;
- Codex sandbox and approval policy remain active;
- no network in offline mode;
- observable tool calls and results;
- validator result independent from the model's final claim.
- no persistent mutation of unrelated user Codex configuration when an invocation-scoped or
  capsule-specific profile is sufficient;
- no claim of hot-swapping an already-running App task until that behavior is verified against the
  pinned App/CLI version.

Exit condition:

- Codex App opens an offline handoff task using the local capsule model; one unseen validation task
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

## MVP completion definition

The MVP is complete only when all of the following are true:

- Teacher collection was driven by `$capsule-teacher` and immutable assignments;
- real tool calls produced a validated train/validation dataset;
- the trainer consumed the published dataset and saved a reloadable LoRA checkpoint;
- training configuration and process metrics were persisted;
- checkpoint evaluation was appended through the existing learning-curve functions;
- a quantized capsule ran offline;
- Codex App used the injected local capsule through its existing harness on an unseen validation task;
- the task validator passed and telemetry was written;
- every artifact can be traced by IDs, revisions, and digests.

## Codex collaboration rule

- Without explicit user authorization, Codex may edit only files under `tests/`.
- For implementation or script changes without authorization, Codex must output complete code blocks
  and the user applies them.
- Read-only inspection and test execution remain allowed for diagnosis and verification.
- Any write outside `tests/` requires explicit authorization naming the intended scope.
