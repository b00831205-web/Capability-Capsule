# Capability Capsule Experiment Decision Record

Status: pre-registration draft  
Last updated: 2026-09-10

This file records the decisions made before post-training experiments begin. It is the durable,
version-controlled companion to `offline-capability-decay-experiment-protocol.pdf`. Changes made
after results are observed must be recorded as amendments rather than silently replacing the
original decision.

## Product hypothesis

Capability Capsule will use an Adapter-first architecture. The host keeps a compatible base model,
while each lightweight capsule supplies a task-specific LoRA adapter, model identity, runtime
configuration, tool policy, evaluation evidence, and harness metadata. Switching tasks should mean
switching adapters rather than redistributing a complete model.

The existing retrieval Flight MVP remains a reproducible baseline and optional knowledge layer. It
is not considered evidence that the local model has acquired a task capability through training.

## Initial model comparison

- Compare one approximately 0.8B non-thinking base model with one approximately 1.5B/2B
  non-thinking base model.
- Keep task data, prompts, tool policy, decoding settings, memory budget, and evaluation cases fixed
  wherever the model architecture permits.
- Treat output speed as a constraint and time-bounded task success as the primary product outcome.
- Compare the trained high-precision LoRA reference with its quantized deployment form. The
  acceptable quality margin must be registered before the comparison results are inspected.

Exact model identifiers, quantization formats, LoRA ranks, and numerical acceptance margins remain
open until the local training stack and candidate model licenses are verified.

## Experimental groups

The first post-training study will contain at least these groups:

1. Untrained base-model control for each model size.
2. Retrieval-only Flight MVP baseline where the task permits retrieval.
3. LoRA-trained model evaluated without retrieval.
4. LoRA-trained model with the same optional retrieval layer.
5. Quantized deployment form of each accepted LoRA candidate.

Teacher-model and cloud frontier-model comparisons are reserved for a later study. They must use the
same held-out task definitions and scoring rules, but they do not block initial engineering
selection between the small local candidates.

## Task cycles and knowledge-tree radius

Offline duration is represented by a logical task cycle rather than wall-clock time alone. Longer
cycles widen the expected task distribution (the knowledge-tree radius), which introduces a
trade-off between specialist precision and multi-task coverage under a fixed total-memory budget.

The experiment will therefore evaluate multiple pre-registered cycle lengths and task-mixture
widths. Each model must complete the full logical cycle assigned to its condition. Fast hardware may
replay more logical cycles, but it must not redefine the amount of work in one cycle.

## Repetition and hardware handling

- Use at least 30 independent repetitions per primary condition as an initial floor.
- Determine the final sample count from confidence-interval width or power analysis, not from the
  number 30 alone.
- Record hardware, runtime, model build, quantization, thread count, memory limits, and decoding
  settings for every run.
- Use servers for high-volume statistical replay and the target laptop for final full-cycle
  endurance and deployment validation.
- Report task-normalized results separately by hardware. Do not equate six hours on two different
  devices with the same experimental exposure.

## Primary measurements

- Task success rate and confidence interval.
- Time-bounded task success rate.
- Accuracy or score as a function of logical cycle position.
- Accuracy-decay slope and area under the performance-versus-cycle curve.
- Tasks completed per hour, tokens per second, and end-to-end task latency.
- Peak resident memory, model storage, adapter storage, and total fixed memory consumption.
- Tool-call correctness, invalid action rate, and policy-violation rate.
- Failure categories and recovery rate.

All metrics must be calculated from immutable per-case records. Aggregate tables and charts are
derived outputs and must be reproducible from those records.

## Planned visual outputs

- Accuracy versus logical offline duration, with uncertainty bands.
- Accuracy-decay comparison between model sizes and deployment formats.
- Time-bounded success versus throughput.
- Quality versus total memory footprint Pareto plot.
- Task-type heatmap across knowledge-tree radii.
- Failure-category distribution and tool-policy violation counts.

## Data separation and leakage controls

Teacher-generated training trajectories, validation cases, and final evaluation cases must have
separate identifiers and provenance. Deduplication and contamination checks happen before training.
The final evaluation set remains locked and must not be supplied to the Teacher as training context.

## Immediate implementation order

1. Define versioned schemas for teacher trajectories, experiment manifests, and per-case results.
2. Create the Codex Teacher Skill and its auditable generation workflow.
3. Validate, normalize, deduplicate, and split the first dataset.
4. Freeze a small logical-cycle evaluation set before training.
5. Measure untrained and retrieval-only controls.
6. Run bounded LoRA feasibility experiments in an isolated environment.
7. Validate quantized deployment and generate reproducible statistical charts.
8. Extend the capsule manifest only after the accepted adapter artifacts are known.

