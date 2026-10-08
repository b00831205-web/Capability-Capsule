# Why `stage1-codex-qwen35-2b-003` Scored 0/3 on `stage1-codex-validation-v2`

Analysis date: 2026-09-21. Read-only analysis of committed artifacts plus the run outputs
produced earlier in this session. Scripts and raw output are in `tmp/diag/analyze_failure.py`,
`tmp/diag/analyze_failure3.py`, `tmp/diag/analyze_failure2.out`, and `tmp/diag/analyze3.out`.

## Summary

The round failed for four independent reasons, and none of them is model capacity. Ordered by how
much damage each does:

1. **The trained tool-call protocol does not match the evaluated harness contract.** All 88 training
   tool calls carry `workdir` (plus `shell`/`login`/etc.); the evaluator declares a one-argument
   `cmd` schema with `additionalProperties: false`. The protocol match rate is **zero**.
2. **The SFT export contains no tool schema and no system prompt.** The model was asked to emit calls
   for a tool signature it was never shown, on top of a system prompt it never saw.
3. **A third of the training data teaches edit forms the evaluator rejects.** Only 8 of 24
   trajectories use the accepted `.Replace` + `Set-Content` form; 8 more use `git apply` or
   `Copy-Item`.
4. **Those 8 accepted trajectories are gated on a prompt hint that evaluation removes**, and the hint
   is perfectly confounded with the accepted form — no trajectory shows the form chosen without it.
   That is why the trained behaviour collapses to Unix `sed -i` on a bare "Modify only greeting.py".

## 1. Tool-call protocol mismatch

The export was decoded with the pinned Qwen3.5-2B tokenizer. A training tool call renders as:

```text
<tool_call>
<function=exec_command>
<parameter=cmd>
Get-Content -LiteralPath 'chunker.py' -Raw
</parameter>
<parameter=workdir>
E:/capsule/tmp/stage1-codex-powershell-edit-003/train-chunk-overlap-powershell-edit-004/workspace
</parameter>
<parameter=shell>
powershell
</parameter>
<parameter=login>
False
</parameter>
</function>
</tool_call>
```

Every trajectory in the publication carries extra argument keys. Measured key sets by task group:

```text
task suffix -002 (8 trajectories, oldest increment):
    8x ['cmd', 'max_output_tokens', 'workdir', 'yield_time_ms']
    8x ['cmd', 'justification', 'max_output_tokens', 'prefix_rule', 'sandbox_permissions', 'workdir', 'yield_time_ms']
task suffix -003 (8 trajectories):
   24x ['cmd', 'login', 'max_output_tokens', 'shell', 'workdir', 'yield_time_ms']
    8x ['cmd', 'justification', 'login', 'max_output_tokens', 'prefix_rule', 'sandbox_permissions', 'shell', 'workdir', 'yield_time_ms']
task suffix -004 (8 trajectories, newest increment):
   40x ['cmd', 'login', 'shell', 'workdir']
```

Every one of the 88 training tool calls includes `workdir`. The evaluator declares
`additionalProperties: false` with `required: ["cmd"]`, so **not one training call matches the
evaluated argument schema** — the protocol match rate is literally zero, in all three increments.

The evaluator does the opposite. `evaluate_checkpoint_v2.py` supplies exactly one tool:

```json
{"type": "function", "function": {"name": "exec_command",
 "parameters": {"type": "object", "properties": {"cmd": {"type": "string"}},
 "required": ["cmd"], "additionalProperties": false}}}
```

and `evaluate_coding_case` builds only:

```python
messages = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": case.task.task},
]
```

There is no `workdir` anywhere in the evaluation prompt. During training, the working directory was
carried per call as a parameter. During evaluation the model is expected to infer that the current
directory is the fixture, purely from the system prompt. So the model was trained to resolve file
locations through a channel the evaluation does not provide, and trained to emit arguments the
evaluated schema forbids.

## 2. No tool schema and no system prompt in the export

`encode_sft_trajectory` in `src/capability_capsule/training/sft.py` calls the chat template at
line 279 **without a `tools` argument**:

```python
encoded = tokenizer.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=False,
    return_dict=True,
    return_assistant_tokens_mask=True,
    truncation=True,
    max_length=max_length,
)
```

Decoding the export confirms the consequence:

| Probe | Present in export |
| --- | --- |
| `<|im_start|>system` | **False** (0 occurrences) |
| `"name": "exec_command"` | **False** |
| `<tools>` | **False** |
| `Run one PowerShell command` (tool description) | **False** |
| `Get-Content` | True (inside call arguments only) |
| `.Replace(` | True |
| `Set-Content` | True |

`TRAINING_ENV_BLOCKER.md` §7 noted this as an environment fact under "no tool schema"; it is in fact a
decoded-data defect, not an environment one.

At inference, `TransformersPeftTurnGenerator.generate` **does** pass `tools=list(tools)`, so the
evaluated prompt contains a tool schema block that never appeared in any training example. The model
must therefore generalize from "no declared tools" to "one declared tool with a specific JSON schema"
while also adopting a system role it never saw.

Note also that the training call format is *learnable* and was learned — the parser reconstructed
valid calls, which is why `tool_call_count` is 4 in every case. The problem is not the call syntax.
It is that the accepted **argument set** and the **command vocabulary** do not transfer.

## 3. Two thirds of the data teaches rejected edit forms

The evaluator's `ConstrainedPowerShellExecutor` authorizes exactly three command forms
(`src/capability_capsule/eval/coding_checkpoint.py`):

- `_READ`: `cat`/`type`/`Get-Content` of `greeting.py` or `test_greeting.py`
- `_REPLACE` + `_SET_CONTENT`: a `.Replace(...)` together with `Set-Content ... greeting.py`
- `_PYTEST`: `python -m pytest -q` or `pytest -q`

Anything else returns exit code 126 with `authorized=False`, and `evaluate_coding_case` counts that
as an invalid tool call.

Per-trajectory command chains in the training publication:

```text
train-cache-rate-exec-002                    GITAPPLY->TEST          REJECTED FORM
train-chunk-overlap-exec-002                 COPY->TEST              REJECTED FORM
train-config-toml-exec-002                   COPY->TEST              REJECTED FORM
train-dataset-empty-exec-002                 GITAPPLY->TEST          REJECTED FORM
train-document-suffix-exec-002               GITAPPLY->TEST          REJECTED FORM
train-manifest-build-id-exec-002             GITAPPLY->TEST          REJECTED FORM
train-task-normalization-exec-002            COPY->TEST              REJECTED FORM
train-workspace-path-exec-002                GITAPPLY->TEST          REJECTED FORM
train-*-powershell-003               (8x)    READ->REPLACE->TEST->TEST     accepted
train-*-powershell-edit-004          (8x)    READ->REPLACE->TEST->TEST     accepted
```

Counts: **8 of 24** trajectories contain both `.Replace(` and `Set-Content`; **5** use
`git apply`; **3** use `Copy-Item`. All eight `-exec-002` trajectories — the oldest third of the
cumulative dataset — teach an edit mechanism that is rejected at evaluation time.

So the model's edit distribution during training was roughly one third `.Replace`+`Set-Content`, one
third `git apply`, and one third `Copy-Item`. At evaluation it sampled the wrong thirds: it produced
`sed -i`, which is a *fourth* form present in none of the training data — a generic Unix prior from
the base model rather than a learned behaviour.

This is the direct cause of the observed output. Cases 1 and 2 emitted
`sed -i 's/return f"Hello, {name}"/return f"Welcome, {name}"/' greeting.py`, which match no
authorized pattern, so the executor returned "Command is outside the authorized evaluation subset"
twice per case — exactly the `invalid_tool_call_count: 2` recorded for both.

## 4. The trajectories leak the instruction that made the right form predictable

The eight accepted trajectories do not merely demonstrate the right form; their **user prompts
dictate it**. Measured exactly: those 8 prompts run 189–220 characters and all name the edit form,
while the other 16 run 109–140 characters and name no shell or edit mechanism at all.

```text
Fix chunk_text so adjacent chunks preserve the requested overlap. Modify only chunker.py and
validate the behavior. Use PowerShell Get-Content, a guarded .Replace, Set-Content, and then
validate.
```

The evaluation tasks omit that sentence entirely:

```text
Modify only greeting.py so greeting("Ada") returns "Welcome, Ada" and the supplied test passes.
Do not modify test_greeting.py.
```

Crucially, the hint and the accepted edit form are **perfectly confounded**: the same 8 trajectories
are both the only ones that name PowerShell and the only ones that use `.Replace` + `Set-Content`.
No trajectory shows the accepted form chosen without the hint, so the data gave the model no way to
learn that the form belongs to the task rather than to the prompt. The evaluation removes the
trigger, so the policy falls through to the base model prior (`sed -i`, `cat`).

This is the single most actionable finding, because it is a data-design error rather than a capacity
limit: the demonstration and the trigger must be separated so the form is selected by the *task*,
not by an explicit hint.

## 5. Contributing factors

- **Interaction budget.** `max_tool_rounds=4` and `max_new_tokens=256`. With a wasted read and a
  rejected edit, four rounds leave no room to recover. The unseen case burned all four on
  `cat greeting.py`.
- **No rejection-recovery experience.** Training contains **zero** unauthorized-command results:
  tool results are 68 × `exit_code=0` and 20 × `exit_code=1`, and every `exit_code=1` is the real
  "No module named pytest" environment failure, which the trajectories recover from by switching to
  WSL. The model never learned what to do when a command is refused for being outside an authorized
  subset. At evaluation it received that message twice per fixed case and repeated the same rejected
  command both times — a repetition loop, not a convergent retry.
- **Prose-only assistant turns.** Of 199 assistant turns, **111 (56%)** contain prose and no tool
  call at all ("Inspect the authorized disposable fixture with a PowerShell-native file read.",
  "Run the fixture's independent tests."). These are valid training targets and appear to reinforce
  stalling behaviour. The unseen case's `cat` loop matches this pattern.
- **Capacity and scale.** Low-rank LoRA (r=8, alpha=16, q_proj/v_proj only, 417,792 trainable
  parameters) over 24 trajectories / 24,889 tokens for 24 steps. Validation loss improved
  (1.022344172000885 → 0.9846928119659424), so the adapter did move toward the training data, but a
  third of that data taught a rejected edit form and all of it taught an argument protocol the
  evaluator rejects. More steps on this data would amplify the wrong conditional policy, not fix it.

## What this round does and does not prove

It does **not** prove that Qwen3.5-2B is incapable of the task. The evaluation has never yet been
run against a checkpoint whose training data matched the harness contract:

- the `-002-recovery-004` checkpoint also scored 0/3, and its training data was the older 16
  trajectories with the same protocol mismatch;
- `-003` added 8 protocol-matched trajectories, but they are the only 8 of 24, they are gated on a
  prompt hint, and even they emit the `workdir` argument the evaluated schema forbids.

Until at least one checkpoint is trained on data that matches the evaluated protocol (correct
argument set, tool schema and system prompt present at train time, accepted edit form only, and no
prompt-level hint), a 0/3 result cannot be attributed to model scale.

## Recommended next increment

Each item is a concrete, testable change to the data or the harness contract, not a loosening of the
evaluator.

1. **Align the tool protocol.** Export training examples with the same argument set the evaluator
   uses — a single `cmd` argument — and render a `tools=` block through the chat template in
   `encode_sft_trajectory`. Also render the evaluation `system_prompt` (or an equivalent) into the
   training input so the system role is in distribution. `workdir` must either be removed from
   training calls or added to the evaluated schema; today the two disagree on every call.
2. **Drop or rewrite the 8 `-exec-002` trajectories** so no training example teaches `git apply` or
   `Copy-Item` as the edit mechanism. They are the largest single source of conflicting policy.
3. **Remove the prompt-level hint** from user prompts in new trajectories: state the task the way the
   validation suite states it, and let the assistant choose the PowerShell edit form. Otherwise the
   model keeps learning a cue that evaluation never supplies. This requires a trajectory where the
   accepted form is demonstrated *without* the hint, since none exists today.
4. **Teach rejection recovery explicitly.** Include at least one trajectory where a command is
   refused as outside the authorized subset and the assistant recovers to `.Replace` + `Set-Content`.
   Do not fabricate this: it must be a real executed refusal.
5. **Reconsider the round budget** for the evaluation, or train the model to use fewer rounds, so a
   single rejected attempt is not terminal.

The nearest-term correction requires items 1, 2, and 3 together; item 1 alone would still leave the
model choosing among three edit mechanisms, two of which the evaluator rejects.
