# Training Environment Blocker and `stage1-codex-qwen35-2b-003` Result

Date: 2026-09-21 (UTC)
Authored under the project write boundary: uses only `tests/`, `tmp/`, `runs/`, and `artifacts/`.
`MVP_ROADMAP.md` and `PENDING_CONFIGURATION_OPTIMIZATIONS.md` are **not** written here; their proposed
edits are supplied as text in the final report for the user to apply.

## 1. State found at session start

`runs/training/stage1-codex-qwen35-2b-003` was the most recent training attempt and contained:

- `training-run.json` pinning dataset `stage1-codex-powershell-edit-003`, dataset digest
  `b3adec98d16a581a849aa26ac98ec9d26dd1dd8c5745f89d8a3ffab776854dbe`, SFT export
  `stage1-codex-powershell-edit-003-qwen35-2b-v1`, 24 train / 2 validation trajectories, base
  revision `15852e8c16360a2fea060d615a32b45270f8a8fc`, seed 42, `max_steps` 24, LoRA rank 8, alpha
  16, dropout 0.05, targets `q_proj`/`v_proj`, sequence length 4096;
- `trainer-process.jsonl` with a single `started` event at `2026-09-21T08:56:21.194183Z`;
- no `metric`, `checkpoint_saved`, `completed`, or `failed` event, no `adapter-final/`, no
  `checkpoint.json`.

So `-003` had produced no steps and no checkpoint. The last checkpoint-producing run remained
`stage1-codex-qwen35-2b-002-recovery-004` (step 16, training loss 1.2680502831935883, validation loss
1.022344172000885), already scored `0/3`.

## 2. Root cause of the `-003` death: memory exhaustion, not a peft import fault

Two earlier hypotheses in this investigation were wrong and are corrected here.

**Wrong hypothesis A — "the WSL VM crashes when importing peft".** The `import peft` failures were a
side effect: each apparent import crash happened on a VM that had *already* been restarted after an
OOM kill, and the post-restart shell was cold and slow. Importing the full stack is in fact
survivable, as proven below.

**Wrong hypothesis B — "the peft import is the hard blocker".** The real blocker is peak memory.

Decisive evidence, from `dmesg` on the boot that contained the crashed `-003` attempt:

```text
python invoked oom-killer: gfp_mask=0x140cca(GFP_HIGHUSER_MOVABLE|__GFP_COMP), order=0, oom_score_adj=0
oom-kill:constraint=CONSTRAINT_NONE,...,global_oom,task_memcg=/init.scope,task=python,pid=14078,uid=1000
Out of memory: Killed process 14078 (python) total-vm:29359992kB, anon-rss:14282608kB, ...
oom_reaper: reaped process 14078 (python)
systemd-journald[53]: Received SIGTERM from PID 1 (systemd-shutdow)
```

The trainer was OOM-killed with **peak anonymous RSS 14.28 GB against a 15,384 MB VM**, and that kill
initiated the WSL restart. Because the kernel killed the process outright, no Python exception was
raised and no `failed` event could be written — exactly the observed silent death.

Facts that ruled out alternatives:

| Observation | Conclusion |
| --- | --- |
| 14,553 MB available immediately before one crash | not gradual memory pressure from idle services |
| `peft` 0.21.0 has no `*.so` and an intact `RECORD` | not a corrupt install |
| `torch` alone: OK (479 MB RSS); `transformers` alone: OK (66 MB); `accelerate` alone: OK | no single broken dependency |
| `torch.cuda.is_available()` returns `False`; `torch.distributed` imports fine | not CUDA/NCCL initialization |
| `E:\` read throughput 140 MB/s, load average ~1.7, 13.5 GB available | not disk or CPU starvation |

The pre-OOM `-003` attempt had in fact reached the training loop and completed step 1 (loss 1.501)
before the step-2 peak killed it, which is consistent with peak memory being the trigger rather than
any import.

## 3. Reproduction and fix

The trainer's own stack was verified to import and run on CPU once a device-probe guard is installed.
`peft`'s import chain calls vendor accelerator probes through `accelerate.utils`
(`is_xpu_available`, `is_npu_available`, `is_hpu_available`, and five more). On this AMD host those
probes fault at the WSL driver boundary. The working driver applies this guard before importing
`transformers` or `peft`:

```python
import accelerate.utils as acc_utils
import accelerate.utils.imports as acc_imports

for name in (
    "is_xpu_available", "is_npu_available", "is_mps_available", "is_hpu_available",
    "is_mlu_available", "is_sdaa_available", "is_musa_available", "is_torch_xla_available",
):
    if hasattr(acc_imports, name):
        setattr(acc_imports, name, lambda *a, **k: False)
    if hasattr(acc_utils, name):
        setattr(acc_utils, name, lambda *a, **k: False)
```

With the guard in place, the model loads, LoRA attaches (417,792 trainable parameters), and a
forward/backward pass runs.

Memory, measured across attempts:

| Configuration | Peak RSS | Outcome |
| --- | --- | --- |
| float32 weights, gradient checkpointing off | 14,282 MB | OOM-killed at step 1 |
| float32 weights, gradient checkpointing on | 13,415 MB | fell to 117 MB free; stopped deliberately before OOM |
| bf16 weights, gradient checkpointing on | 13,168 MB | **24/24 steps completed** |

The host reports `torch.backends.cpu.get_cpu_capability() == "AVX512"` with oneDNN available, and
bf16 matmul and linear kernels work. bf16 halves the 7.5 GB float32 parameter footprint.

A separate, self-inflicted problem wasted several runs and is recorded so it is not repeated: piping
a `wsl.exe` call into `Select-Object -First N` closes the pipe early and kills the WSL process,
sometimes before its command had run at all.

## 4. Final result: `stage1-codex-qwen35-2b-003`

Run directory: `runs/training/stage1-codex-qwen35-2b-003`.

- dataset `stage1-codex-powershell-edit-003`, 24 train / 2 validation trajectories, 311 messages,
  88 tool calls, 24,889 train tokens (all 24 consumed, none truncated);
- base `Qwen/Qwen3.5-2B` at revision `15852e8c16360a2fea060d615a32b45270f8a8fc`, seed 42;
- device CPU, dtype bf16, gradient checkpointing enabled, `eval_strategy="no"`;
- **24 of 24 optimization steps**, one epoch, **training loss 1.0836982702215512**, wall clock
  509.8 s (about 20 s per step), **peak RSS 13,168.6 MB**;
- adapter saved to `adapter-final/` with `adapter_config_sha256`
  `604cd23943abad5ee3c58a175969f0f4e013c8b9695debbd56594de4f13e1723` and `adapter_model_sha256`
  `6341f1a18e78700a5a5907ea820adbadfb0c518d3b961eec26e9e0f9c26b8045`;
- `trainer-process.jsonl` records `started`, `checkpoint_saved` (step 24), and `completed`.

Separate-process validation (`validate_saved_adapter.py`) reloaded the adapter through
`reload_lora_adapter` with hash verification and reported:

- `model_class`: `PeftModelForCausalLM`;
- `validation_loss`: **0.9846928119659424** on the two held-out validation examples;
- per-example loss: 0.6326336860656738 and 1.336751937866211.

Comparison with the previous checkpoint on identical inputs:

| Run | Train steps | Train loss | Validation loss | v2 suite |
| --- | --- | --- | --- | --- |
| `stage1-codex-qwen35-2b-002-recovery-004` | 16 | 1.2680502831935883 | 1.022344172000885 | 0/3 |
| `stage1-codex-qwen35-2b-003` | 24 | 1.0836982702215512 | 0.9846928119659424 | 0/3 |

## 5. Evaluation on the unchanged v2 suite

Suite `stage1-codex-validation-v2`, capability `codex-windows-powershell-single-file-change`,
evaluation directory `runs/evaluation/stage1-codex-qwen35-2b-003-v2`.

The suite digest was recomputed during recording and equals the previously documented
`3e756195c1585c57c4dcc8a3fef40cb2653a67bc57820c6156602f357301b9bb`, so the three cases are
demonstrably unmodified and the result is comparable with earlier checkpoints.

Result: **0/3**. All three cases ended `ToolRoundLimitExceeded` at the four-tool-round limit.

| Case | Input tokens | Output tokens | Tool calls | Invalid tool calls | Peak RSS | Duration |
| --- | --- | --- | --- | --- | --- | --- |
| `smoke-validation-greeting-change-codex-001` | 2773 | 195 | 4 | 2 | 4627.7 MB | 48.6 s |
| `validation-greeting-exec-exec-002` | 2741 | 193 | 4 | 2 | 4647.1 MB | 44.9 s |
| `validation-powershell-salutation-unseen-001` | 2770 | 145 | 4 | 0 | 4667.0 MB | 41.0 s |

What the adapter actually emitted matters more than the score:

- The genuinely unseen case emitted `cat greeting.py` **four times** and never edited the file. Zero
  invalid tool calls, because `cat` is a valid PowerShell alias, but it is a read loop with no
  progress and no edit attempt.
- The two fixed cases emitted `cat greeting.py` / `cat test_greeting.py` and then Unix
  `sed -i 's/.../.../' greeting.py`, which the Windows PowerShell harness rejects — two invalid tool
  calls each.

So the new increment did not change the failure mode in the direction the roadmap requires. The
adapter got closer to attempting an edit, but it still reaches for Unix `sed -i`, and on the unseen
case it does not attempt an edit at all. The data increment taught the transitions in its own
trajectories without transferring the PowerShell edit form to these cases.

Recording: `runs/evaluation/stage1-codex-qwen35-2b-003-v2/learning-curve.jsonl` now contains the
points `stage1-codex-qwen35-2b-003-step-24` with `cumulative_trajectory_count` 24,
`cumulative_token_count` 24889, and `success_rate` 0.0, plus `checkpoint-record.json`.

## 6. Defect found in the recording path

`process_training_checkpoint_files` (in `src/capability_capsule/eval/training_completion.py`) cannot
record any coding evaluation suite. It calls `inspect_evaluation_suite`, which calls
`load_cases` from `eval/dataset.py`; that loader is a `TypeAdapter(tuple[RetrievalCase, ...])` for
*retrieval* cases. A coding suite is a single JSON object
(`schema_version`, `capability_id`, `evaluation_suite_id`, `evaluation_split`, `system_prompt`,
`cases`), so validation fails:

```text
pydantic_core._pydantic_core.ValidationError: 1 validation error for
tuple[function-after[validate_expected_node(), RetrievalCase], ...]
  Input should be a valid array [type=tuple_type, input_value={'schema_version': '0.1',...
```

The three cases themselves ran and wrote `case-results.jsonl` before this failure, so only the ledger
step was lost. The `-003` result was recorded by
`tmp/stage1-codex-qwen35-2b-003/record_evaluation_v2.py`, which reproduces `inspect_evaluation_suite`'s
canonicalization exactly and asserts the recomputed digest before writing. This defect is
pre-existing and also affects the earlier `-002-recovery-004` recording path; it needs a coding-aware
suite inspection function, and it is a source change outside `tests/`, so it is reported rather than
applied.

## 7. Environment facts

- Ubuntu WSL2, `/etc/wsl.conf` sets `systemd=true`; Windows `~/.wslconfig` sets
  `[wsl2] networkingMode=mirrored` and no memory cap, so the VM inherits 15,384 MB of 16 GB.
- Python 3.13.13, torch 2.14.0+cu130 (GPU build retained and unused by MVP CPU training),
  transformers 5.17.0, peft 0.21.0, accelerate 1.15.0.
- The `.venv` is a Linux environment and is **not** runnable from Windows.
- `causal_conv1d` and `flash-linear-attention` are absent, so Qwen3.5's linear-attention kernels fall
  back to reference PyTorch implementations. This is correct but slower.
- Peak RSS at 13.2 GB leaves only about 2.2 GB of headroom on this VM. The adapter proved it fits,
  but it is close enough that the training configuration is fragile.

## 8. Preserved failure evidence

- `runs/training/stage1-codex-qwen35-2b-003-aborted-20260921` — the original `started`-only attempt.
- `runs/training/stage1-codex-qwen35-2b-003-oom-step1-20260921` — the float32 attempt that completed
  step 1 (loss 1.501) and was OOM-killed at step 2.
- `tmp/diag/` — the probe scripts and logs, including `env_check.sh`, `probe_guarded.sh`,
  `smoke_cpu_train.sh`, `oom_check.sh`, `memwatch.sh`, and their `.log`/`.out` results.

## 9. Next blocking step

The roadmap's requirement is unchanged and now measured: teach short, observable
inspection→guarded-PowerShell-edit→validation transitions that transfer to an unseen single-file
change. The `-003` increment did not achieve that, because the adapter still emits Unix `sed -i` and
on the unseen case never leaves the read loop. A further data increment should target exactly that
gap, and the evaluator must not be loosened to accept Unix commands merely to raise the score.
