---
name: capsule-teacher
description: Generate and audit Teacher coding trajectories for Capability Capsule from authorized training or validation fixtures. Do not use for locked test cases, final grading, or ordinary project development.
metadata:
  short-description: Generate auditable Capsule Teacher trajectories
  version: "0.1.0"
---

#Capsule Teacher

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
- Never overwrite or rewrite existing JSONL records.

## Workflow

### 1. Verify provenance

Confirm that the split is not `test`.

Resolve the exact starting revision or immutable snapshot. Inspect the initial working state and do not silently continue from unexplained modifications.

Confirm that the supplied task, allowed tools, expected changes, and validation rules refer to the same fixture version.

### 2. Execute the task

Solve the task within the authorized fixture.

Prefer the smallest change that satisfies the task. Capture only actions that materially explain the solution or demonstrate recovery from a meaningful tool error.

Do not add fabricated, redundant, or decorative tool calls.

### 3. Capture the observable trajectory

Build messages in chronological order using the roles defined by `TrajectoryMessage`:

- `user`: the task and relevant supplied constraints
- `assistant`: a concise visible action summary or final outcome
- `assistant` with `tool_calls`: the exact structured tool request
- `tool`: the corresponding observable result, with `tool_name`

Every recorded tool request must correspond to a real tool invocation. Every tool-result message must identify the tool that produced it.

Preserve relevant patch content and validation evidence. If output is shortened, state that it was truncated and retain the decisive lines.

Never manufacture a successful result, file change, command output, or validation outcome.

### 4. Validate independently

Run the supplied validators after the change.

A task is successful only when its independent validators pass and the resulting file state satisfies the expected boundaries.

If validation fails, record the failure and any genuine recovery attempts. Do not describe an unrecovered trajectory as successful.

Teacher judgment alone is not validation.

### 5. Build the record

Use `TeacherTrajectory` from
`src/capability_capsule/eval/records.py`.

Set:

- `schema_version` to `"0.1"`
- `teacher_skill_version` to `"0.1.0"`
- `teacher_model` to the actual model identifier
- `source_revision` to the verified starting revision or snapshot
- `messages` to the observable chronological trajectory

Use tags only for supplied or verified classifications. Do not use tags to invent missing experimental metadata.

### 6. Validate and append

Validate the complete record with `TeacherTrajectory` before persistence.

Use `load_jsonl` from `src/capability_capsule/eval/jsonl.py` to ensure the destination does not already contain the same `trajectory_id`.

Use `append_jsonl` to append the validated record. Never edit an existing line in place.

If schema validation or uniqueness validation fails, do not append the record.

### 7. Report the result

Report:

- trajectory ID
- task ID
- split
- source revision
- destination path
- validation outcome
- whether the record was appended

Do not claim that the trajectory has passed deduplication, leakage review, dataset acceptance, or final evaluation unless those independent stages actually ran.

## Quality gate

A trajectory is ready for raw-dataset review only when:

- provenance identifies the exact starting fixture
- the split is not the locked test split
- tool calls and results reflect real observable actions
- the patch or task result is represented
- validation evidence is present
- no hidden reasoning or sensitive data is included
- the record passes the current Pydantic schema
- the trajectory ID is unique in the destination