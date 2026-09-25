# Student base-model selection

Use this procedure only when selecting or reviewing the Student base model. It produces a recommendation
for the defined workload and deployment environment, not a general model ranking.

## Inputs

Require or derive the following without inventing missing values:

- verified deployment hardware-profile ID and path
- training environment when it differs from deployment
- workload scope and representative authorized validation tasks
- offline/online constraint and runtime/backend choices
- minimum context and output requirements
- tool-call and structured-output requirements
- latency, memory, concurrency, and sustained-session targets
- candidate model identifiers and immutable revisions
- license and redistribution constraints
- available quantizations and their actual artifact sizes
- existing deployment benchmark results, if any

If model metadata may have changed and network access is authorized, verify it against official model
cards or documentation. Otherwise record the metadata source and mark unverified fields. Never open the
locked test split to choose a model.

## Decision procedure

1. Preserve models explicitly fixed by the user. Review feasibility and report conflicts instead of
   silently replacing the choice.
2. Reject candidates incompatible with the deployment runtime, license, required prompt/tool format,
   minimum context, or safe memory envelope.
3. Estimate memory as model weights plus runtime overhead, KV cache at the target context, and working
   memory. State assumptions and retain safety headroom for the operating system, tools, and long runs.
4. Compare surviving candidates on representative validation tasks. Include task success, structured
   output validity, tool-selection accuracy, peak resident memory, prompt-processing speed, generation
   speed, time to first token, and long-session stability when relevant.
5. Prefer the smallest candidate that meets the acceptance thresholds. A larger model is justified only
   by a measured task-quality benefit that matters to the stated workload.
6. Recommend one primary candidate and, when useful, one smaller fallback. Record why other candidates
   were rejected or deferred.
7. Mark the result `provisional` when it relies on specifications or estimates. Mark it `benchmarked`
   only after the exact model revision, quantization, runtime, and target hardware pass the benchmark.

Do not combine unrelated metrics into an unexplained score. If a weighted score is useful, publish its
formula, units, weights, and raw measurements so the decision can be reproduced.

## Recommendation artifact

Write JSON to the assignment-approved path using this shape:

```json
{
  "schema_version": "0.1",
  "recommendation_id": "base-model-<stable digest>",
  "created_at": "RFC 3339 timestamp",
  "status": "provisional | benchmarked",
  "hardware_profile_id": "hardware-...",
  "workload": {
    "scope": "defined task distribution",
    "offline_required": true,
    "required_context_tokens": null,
    "target_session_hours": null
  },
  "primary": {
    "model_id": "publisher/model",
    "revision": "immutable revision",
    "quantization": "exact format",
    "runtime": "runtime and version",
    "reasons": [],
    "risks": []
  },
  "fallback": null,
  "rejected": [],
  "evidence": {
    "metadata_sources": [],
    "estimated": {},
    "measured": {}
  },
  "required_benchmark": {
    "task_suite": "authorized validation suite identifier",
    "acceptance_thresholds": {},
    "commands": []
  }
}
```

The stable recommendation ID must exclude timestamps and volatile measurements. Do not embed private
machine identifiers or locked-test results. Reference the hardware profile rather than copying it.

## Dataset consequences

After the candidate is accepted:

- pin the exact Student configuration in the collection plan
- set trajectory length and decomposition from demonstrated Student capacity
- retain complete task coverage across episodes
- keep validation tasks separate from training trajectories
- rerun the model benchmark when the model revision, quantization, runtime, prompt format, or verified
  deployment profile changes
