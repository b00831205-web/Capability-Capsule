---
name: capsule-teacher
description: Select a Student base-model candidate and generate or audit Teacher coding trajectories for Capability Capsule from authorized training or validation fixtures. Do not use for locked test cases, final grading, or ordinary project development.
metadata:
  short-description: Generate auditable Capsule Teacher trajectories
  version: "0.2.0"
---

# Capsule Teacher

Generate reproducible coding-task trajectories for Capability Capsule training data.

The user's explicit instructions take precedence over this skill.

## Required inputs

Before generating a trajectory, identify:

- `trajectory_id`
- `task_id`
- dataset split
- Teacher model identifier
- task specification
- authorized fixture root
- source revision or immutable fixture snapshot
- allowed tools
- expected changes
- validation commands or validation identifiers
- destination JSONL path

For Student-targeted collection, also identify:

- Student model identifier and parameter scale, or authorization to recommend candidates
- intended training and deployment environments
- a hardware-profile JSON path or an assignment-approved path where one may be created
- Student context, output, tool-call, latency, and memory budgets when supplied

Do not invent missing provenance. Ask for it when it cannot be derived from the supplied fixture or dataset plan.

## Boundaries

- Generate trajectories only for `train` or `validation` data.
- Never open, derive from, or write trajectories for the locked `test` split.
- Use only the explicitly designated fixture or disposable workspace.
- Do not treat the Capability Capsule working repository as a training fixture unless the user explicitly designates it.
- Respect the fixture's tool, filesystem, network, and mutation permissions.
- Preserve unrelated changes.
- Never treat Teacher output as truth or use it as the final grader.
- Do not record hidden reasoning or a chain of thought.
- Record only concise action summaries, observable tool calls, tool results, patches, validation evidence, and the final outcome.
- Remove secrets, credentials, personal data, and unrelated machine-specific information.
- Hardware constraints relevant to training or deployment may be recorded only through the
  privacy-filtered hardware profile described below. Never collect host names, user names, device
  serial numbers, MAC or IP addresses, environment variables, or arbitrary process listings.
- Never overwrite or rewrite existing JSONL records.

## Workflow

### 1. Verify provenance

Confirm that the split is not `test`.

Resolve the exact starting revision or immutable snapshot. Inspect the initial working state and do not silently continue from unexplained modifications.

Confirm that the supplied task, allowed tools, expected changes, and validation rules refer to the same fixture version.

### 2. Resolve the Student runtime profile

Do this step before Student-targeted trajectory generation. It is optional only when the assignment
is not intended for a particular Student or runtime.

Model inference location and tool execution location are separate facts. A Teacher model may run in
the cloud while its terminal tool executes on the user's local computer. Therefore, never infer the
hardware profile from where the model is hosted. Inspect the host on which the hardware-probe tool
actually runs.

If the assignment already supplies a validated hardware profile, read it and do not probe a different
machine. Otherwise run:

```text
python .agents/skills/capsule-teacher/scripts/inspect_hardware.py --output <assignment-approved-path>
```

The first probe describes the observed tool-execution host and deliberately marks its relationship to
the Student as `unverified`. Confirm from the assignment or the user whether this is the intended
`training`, `deployment`, or `both` environment. Once confirmed, rerun with the matching
`--student-role`. Do not silently treat the Teacher's tool host as the Student's target computer.

The profile is a sidecar artifact. Reference its `profile_id` and path from collection planning rather
than embedding the full hardware payload in every trajectory. Use the stable profile ID to detect the
same hardware configuration; do not use `captured_at` or currently available memory as identity.

Use the effective execution-environment limits when a VM, WSL instance, or container restricts the
host. Unknown or unavailable values remain `null` and produce warnings; never guess them.

Hardware and Student capacity may constrain sequence length, tool-call count, memory budget, batch
size, quantization, and task decomposition. They must not justify incorrect answers, fabricated tool
results, skipped validation, or removal of necessary task coverage. Prefer splitting a long task into
learnable episodes while preserving the complete correct workflow across the dataset.

### 3. Select or verify the Student base model

When the user has not fixed the Student base model, or explicitly asks for a recommendation, read
[references/base-model-selection.md](references/base-model-selection.md) and produce its recommendation
artifact before generating trajectories.

Treat hardware as a feasibility filter, not as sufficient selection evidence. Also account for the
authorized task distribution, required context, tool-use format, runtime/backend compatibility,
license, quantization availability, latency target, sustained-session duration, and measured task
quality. Keep training hardware separate from deployment hardware: a model may be trained remotely
but must still satisfy the verified deployment profile.

Prefer a primary candidate and a smaller fallback over claiming that one model is universally best.
Label a recommendation `provisional` until the candidate has passed the prescribed benchmark on the
intended deployment environment. Do not claim estimated RAM, tokens per second, context capacity, or
task quality as measured results.

Once the user accepts a candidate, pin its exact model identifier, revision, quantization, runtime,
prompt/tool format, and recommendation artifact ID in the collection plan. Generate separate datasets
when materially different Student capacities require different decomposition or context budgets.

### 4. Execute the task

Solve the task within the authorized fixture.

Prefer the smallest change that satisfies the task. Capture only actions that materially explain the solution or demonstrate recovery from a meaningful tool error.

Do not add fabricated, redundant, or decorative tool calls.

### 5. Capture the observable trajectory

Build messages in chronological order using the roles defined by `TrajectoryMessage`:

- `user`: the task and relevant supplied constraints
- `assistant`: a concise visible action summary or final outcome
- `assistant` with `tool_calls`: the exact structured tool request
- `tool`: the corresponding observable result, with `tool_name`

Every recorded tool request must correspond to a real tool invocation. Every tool-result message must identify the tool that produced it.

Preserve relevant patch content and validation evidence. If output is shortened, state that it was truncated and retain the decisive lines.

Never manufacture a successful result, file change, command output, or validation outcome.

### 6. Validate independently

Run the supplied validators after the change.

A task is successful only when its independent validators pass and the resulting file state satisfies the expected boundaries.

If validation fails, record the failure and any genuine recovery attempts. Do not describe an unrecovered trajectory as successful.

Teacher judgment alone is not validation.

### 7. Build the record

Use `TeacherTrajectory` from
`src/capability_capsule/eval/records.py`.

Set:

- `schema_version` to `"0.1"`
- `teacher_skill_version` to `"0.2.0"`
- `teacher_model` to the actual model identifier
- `source_revision` to the verified starting revision or snapshot
- `messages` to the observable chronological trajectory

Use tags only for supplied or verified classifications. Do not use tags to invent missing experimental metadata.

### 8. Validate and append

Validate the complete record with `TeacherTrajectory` before persistence.

Use `load_jsonl` from `src/capability_capsule/eval/jsonl.py` to ensure the destination does not already contain the same `trajectory_id`.

Use `append_jsonl` to append the validated record. Never edit an existing line in place.

If schema validation or uniqueness validation fails, do not append the record.

### 9. Report the result

Report:

- trajectory ID
- task ID
- split
- source revision
- destination path
- validation outcome
- whether the record was appended
- hardware profile ID and Student-role verification status, when Student targeting was used
- base-model recommendation ID and evidence status, when model selection was used

Do not claim that the trajectory has passed deduplication, leakage review, dataset acceptance, or final evaluation unless those independent stages actually ran.

## Quality gate

A trajectory is ready for raw-dataset review only when:

- provenance identifies the exact starting fixture
- the split is not the locked test split
- tool calls and results reflect real observable actions
- the patch or task result is represented
- validation evidence is present
- no hidden reasoning or sensitive data is included
- any Student-targeted hardware profile describes the intended environment rather than an assumed one
- any recommended base model is pinned only after feasibility and task-fit review
- the record passes the current Pydantic schema
- the trajectory ID is unique in the destination
