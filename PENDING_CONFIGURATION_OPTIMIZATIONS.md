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
- The authorized `stage1-codex-powershell-edit-005` train increment now has 12 unique, revision-pinned
  assignments and 12 appended raw trajectories. Each contains an observed `Get-Content` read, a
  guarded `.Replace` plus `Set-Content` edit, and a passing `pytest -q` result; independent pytest
  also passed for every disposable workspace. The existing train source hash stayed unchanged, and
  the new plan normalizes Student artifact references to portable relative paths while preserving
  their pinned hashes and the existing HarnessProfile. Before publication, perform an independent
  duplicate/leakage and fixture-provenance review; do not mix this raw increment into the existing
  published dataset or train another checkpoint until that gate is explicitly authorized and passed.
- `stage1-codex-powershell-edit-005` is now independently published with 12 train, 0 validation, and
  0 duplicate records. Automated review found no exact fingerprint overlap with earlier published
  train records and no fixed-validation task, fixture-revision, or split-group overlap. Publication
  integrity verification passed, and the train artifact retains the reviewed raw SHA-256
  `86e537f9ec6d1f0459c98f6efa82f92f0a7ea43fc4e13423d702e6afb2abcda2`. Preserve this version and raw;
  define the next SFT training composition explicitly before exporting to a fresh schema `0.3`
  directory. These checks do not measure semantic generalization or checkpoint task success.
- The composition is now fixed as edit-005-only in schema `0.3` export
  `stage1-codex-powershell-edit-005-qwen35-2b-v1`: 12 train / 0 validation examples, 8,169 total tokens,
  2,496 assistant-label tokens, and a 691-token longest sequence. Decoded audit verified all 36
  trainable tool calls and excluded result envelopes; persisted samples and artifact hashes match
  the audit. Use this immutable export for the next fresh-base LoRA experiment. Declare whether
  the control holds optimizer steps or epochs fixed, since changing from 8 to 12 trajectories makes
  equal epochs unequal update counts. Record that choice before training and assess executable
  edits and target text with independently scoped validation.
- Run `stage1-codex-qwen35-2b-007-edit005-32step` now fixes optimizer updates at 32 against run 006
  while consuming the 12-trajectory edit-005-only export from a fresh pinned base. Actual exposure
  is 2.6667 epochs; per-step logs and Trainer state are retained so the configured 4-epoch value
  cannot be mistaken for measured exposure. Training loss was `0.5857911370694637`, adapter hashes
  verified, and independent-process reload passed. The dataset differs from run 006, so their
  training losses are not a task-quality comparison. Next evaluate the new adapter independently
  with fixed task identities and validators, record the exact executor and prompt conditions, and
  include fresh held-out validation before changing training exposure or claiming improvement.
- The authorized run-007 evaluation now freezes fixed-v3 and a separate two-case heldout-v1 suite
  before inference. Hold the simulator, generic command-policy prompt, deterministic generation,
  256-token limit and four-call budget fixed; record prompt/tool/executor hashes alongside case
  digests because the existing suite digest covers cases only. Keep heldout scores and ledgers
  separate, and describe their same-domain limitation. Run 007 starts from a fresh base on edit-005
  only: ledger counts are 12 trajectories / 8,169 tokens, not accumulated run-006 plus run-007 data.
  Do not infer a causal data-volume learning rate from this dataset-replacement comparison.
- Run 007 completed fixed-v3 `0/3` and fresh heldout-v1 `0/2` with no protected-input changes
  (97 files checked). Every read was authorized; no generated edit executed. All five cases
  repeated unsupported `Get-Content -Append` pipelines and hit the fixed tool budget. Invalid
  calls were `[3, 2, 3]` on fixed cases and `[3, 3]` on heldout cases. The keyed-template and
  tuple-join preflight edits passed, so the new fixtures are solvable by the unchanged simulator.
  Prioritize a read-only comparison of full decoded SFT contexts versus inference contexts,
  including the tool-result envelope, assistant-turn boundary and guarded-edit label exposure.
  If needed, separately authorize a training-task diagnostic replay to distinguish failure to
  reproduce a learned edit from heldout transfer failure. Neither this result nor train loss
  identifies insufficient volume, exposure, adapter capacity or model incapability as the cause.
- Read-only post-007 audits reproduced all 12 saved SFT encodings, matched all 48 generation
  prefixes against the actual input_ids, verified all 12 read outputs against their pinned sources,
  and confirmed the guarded edits are labeled. Two teacher-forced training edit contexts yielded
  v2: 1/2 supported edits and 0/2 exact targets; v3: 0/2 supported edits and 0/2 exact targets.
  These memory-only probes localize incomplete seen-context reproduction as well as prompt
  sensitivity, not just heldout transfer. They do not establish a closed-loop training-task score.
  The authorized fresh-base run 008 doubles max_steps to 64 without adding data or changing the
  initial learning rate, seed, model or LoRA settings. Its default linear scheduler gets a longer
  horizon; record this limitation rather than calling it a pure continuation. Evaluate identical
  v3 cases and repeated heldout regression fixtures before deciding whether exposure helped.
- Run 008 completed 64 updates / 5.3333 actual epochs with mean train loss
  `0.27337406313745305`, saved a reloadable adapter and preserved all protected inputs. Its
  unchanged-v3 fixed and heldout-regression scores remain `0/3` and `0/2`. Do not select another
  exposure increase based on loss alone. New generations put `.Replace` inside unsupported
  selection/object pipelines; one rejection led to parameter repetition and a truncated malformed
  tool call. The training-context v2 probes now yield 1/2 exact supported edits (versus 007's
  memory-only 0/2); v3 remains 0/2. In the v2 format case the model emits
  `'return "Hello, {}"'.format(name)` as an argument instead of keeping `.format(name)` inside
  the quoted Python-source literal. Diagnose command argument serialization and prompt sensitivity
  separately. Next use an independently recorded v2 closed-loop diagnostic with unchanged cases,
  executor and budgets; it must not replace the failed v3 acceptance evidence. One seed and two
  teacher-forced contexts do not establish broad capability, prompt-only causality or overfitting.
- Run-008 v2 closed-loop diagnosis is now separately authorized and pinned to the persisted
  training contract, unchanged checkpoint and the same five v3 cases. Freeze the source snapshots,
  validators, executor, tools and budgets; only the system prompt changes. Record distinct suite
  IDs, prompt/suite hashes and isolated fixed/heldout ledgers. Reused heldout fixtures are regression
  cases, not a new unseen set. Preserve all failed v3 evidence and avoid retraining or collecting
  more data before assessing this prompt-only contrast. The completed result is fixed `3/3`
  versus v3 `0/3`, with zero invalid calls and all three within budget; repeated heldout remains
  `0/2`. Mapping-template edits still put `.format(...)` outside a quoted source literal and
  target the return instead of the preserved template. Tuple output behavior passes pytest,
  but the edit bypasses the join mechanism and fails the frozen exact-content structure gate.
  Prompt alignment is therefore materially relevant, not a complete fix. Keep behavior-pass
  evidence distinct from full task success. Prioritize minimal constant/template replacement
  and implementation-preservation training on independent train fixtures and target wording,
  with no per-validation hints. Any future move from exact-content to a semantic/AST structure
  validator needs an independently reviewed, versioned suite; do not loosen the existing gate
  retroactively. Preserve old failed v3 evidence and do not yet promote the checkpoint as ready
  for unseen tasks. This diagnostic preserved 207 files and passed 68 related pipeline tests.
- The targeted `stage1-codex-powershell-literal-edit-006` data increment has now been collected
  separately published and exported to SFT; a fresh training run has now completed:
  eight independent train-only fixtures cover literal substitutions in
  format strings, dictionaries, tuples and branches without replacing their implementation mechanisms.
  All 24 recorded constrained-executor calls and eight independent validators passed; baseline target
  tests failed on all eight initial snapshots. Exact local-patch checks and unchanged validator bytes
  supplement behavior checks, while AST-normalized tests verify that nonliteral structure survives.
  No known exact train/validation overlap or duplicate was found; semantic deduplication remains deferred.
  All 581 protected existing files remain byte-identical; 128 related regression tests passed.
  The plan folder retains its strict two-file
  layout; source/workspace copies and review evidence are separate new artifacts. Keep the v2 contract,
  tools and executor unchanged for any later SFT export. The immutable publication contains eight train,
  zero validation and zero duplicate records; train bytes exactly match the reviewed raw. Publication
  repeated all eight independent validators with cache/bytecode writes disabled and preserved 654
  existing input files, including collection artifacts. Its audit is a separate test-sidecar, never
  an extra file inside the strict four-file published version; 152 related regression tests passed.
  The separately authorized schema-0.3 Qwen SFT v1 now contains eight train examples / zero validation,
  5,506 total tokens and 1,588 trainable tokens (maximum 705, no clipping at the unchanged 4K cap).
  All 24 tool requests and 32 inference-generation prefixes passed decoded audit; nonassistant tokens
  remain masked and narration/tool requests share one assistant turn. The v2 contract, model revision,
  executor and source bytes are unchanged. SFT export preserved 658 files and did not mix older data.
  All 155 related regression tests passed.
  The authorized `009-literal006-32step` training uses a conservative first-pass 32-step budget
  (four epochs on eight examples), with unchanged seed 42, LR 2e-4, rank eight and v2 SFT contract.
  Start from the base model, not run 008's adapter; retain separate logs, checkpoint hashes and an
  independent reload check. The run completed all 32 steps / four measured epochs with mean train
  loss `0.5821066312491894`, preserving 663 existing inputs. Independent-process adapter reload
  passed model revision and adapter hash checks. No training validation or checkpoint task evaluation is included.
  All 175 related tests passed with one nonfatal CPU-only DataLoader `pin_memory` warning.
  Do not treat lower train loss as generalized capability or call this a one-variable control:
  training composition and exposure per example differ from edit005. The separately authorized
  checkpoint-009 evaluation is complete: identical fixed-v2 cases plus first-use trimmed-dictionary
  and uppercase-tuple validation snapshots, frozen with behavior/structure/test-preservation validators
  before inference. Keep v2 prompt/tools, executor and 256-token/four-round budgets fixed and retain
  independent fixed/unseen ledgers. New exact fixtures remain within the greeting domain, not a new
  repository or broad independence claim. Do not reuse training targets, change old suites, relax
  validators after observing outputs, or merge fresh and repeated-case success rates. This data review is not evidence
  that the Student learned the edits or generalized. Do not increase prompt hints, widen the executor,
  use validation targets for training, or retroactively loosen the failed heldout structure gate.
  Results are fixed `0/3` and new unseen `0/2`, both with zero behavior-validator passes. Every read
  succeeded, but every first edit copied the tool-result JSON envelope into an unsupported whole-file
  write; recovery used repeated writes, pipelines or nested PowerShell. All five cases had three
  rejected calls and exhausted the tool budget without editing. All 681 protected files and the four
  new source/test snapshots remain unchanged. All 180 related regression tests passed with one
  nonfatal CPU-only DataLoader warning; pipeline tests do not convert model failures into success.
  Keep this failed checkpoint unpromoted. Before changing
  serialization, adding prompts or changing data again, prioritize a separately authorized 64-step
  fresh-base optimization-exposure control on identical literal006 SFT and hyperparameters. A linear
  scheduler horizon change is a recorded limitation, not continuation of the old run. Do not claim
  that training duration alone is the proven cause; run 008 differed in data as well as update budget.
  Future use of these exact unseen cases is regression; new unseen evidence needs independently
  authored, separately frozen cases, never training on their targets.
  The user has now authorized `010-literal006-64step`, using the same immutable SFT, pinned base,
  seed, learning rate and LoRA configuration as 009. Training completed with max_steps 64
  (eight effective epochs; configured epochs four is overridden). Keep the existing linear scheduler
  and record its changed horizon; do not describe this as resuming 009 or identical LR prefixes.
  Save to a new run, verify independent adapter reload and preservation of existing artifacts.
  Task evaluation remains a separate authorization; loss reduction alone cannot establish capability.
  The first attempt was interrupted after two updates by a WSL environment restart (no checkpoint,
  no Python traceback; restart cause unconfirmed). Preserve its evidence unchanged and use a new
  `010-literal006-64step-retry1` directory for an identical fresh-base retry. Do not change training
  configuration or claim a model/convergence failure from this interruption.
  The retry completed 64 steps / eight measured epochs, mean train loss `0.26792120598838665`
  and last loss `0.010339532978832722`. Independent-process reload verified pinned model revision
  and adapter hashes/default activation; all 722 protected files stayed byte-identical. Persist
  separate training logs/checkpoint evidence. No task evaluation or promotion is authorized here.
  Next seek independent unchanged-v2 checkpoint evaluation, separating repeated fixed cases from
  genuinely first-use frozen validation cases. Compare task behavior and rejection/recovery traces,
  not training loss alone; do not infer duration is the proven sole cause or claim overfitting is ruled out.
  All 184 related regression tests passed with one nonfatal CPU-only DataLoader warning. Keep
  pipeline correctness separate from the not-yet-measured checkpoint-010 task-success rate.
  Checkpoint-010 task evaluation is now separately authorized: exact fixed-v2 plus first-use
  casefold-dictionary and trim-title-tuple cases frozen before inference, with unchanged contract,
  executor and budgets. Preflight references separately; retain behavior/structure/test-preservation
  gates and record fixed versus first-use scores in separate new ledgers. Preserve all prior
  experiments and never train on validation targets. Same-domain exact novelty is not proof of
  cross-project generalization. Results: strict fixed `2/3`, behavior `3/3`; first-use unseen strict
  and behavior `2/2`, with zero rejected calls in all five cases. The remaining fixed case hardcodes
  Ada, passing its Ada-only pytest but failing the unchanged parameter-preserving exact-content gate.
  Do not loosen it retrospectively. All 740 protected files and four new snapshots stayed unchanged.
  The paired fixed comparison improves from 009 `0/3` to `2/3`, with the earlier envelope-copy
  failure absent. It supports optimization exposure as a factor, not a sole proven step-count cause:
  schedule horizon changes too, only one seed was tested, and unseen fixture definitions differ.
  Hold promotion. Prioritize parameter-preserving literal edits and multi-input behavioral checks
  in separately authorized new experiments, not blind extra steps, prompt hints or executor changes.
  Same-domain two-case success does not rule out overfitting or establish cross-project readiness.
  All 189 related regression tests passed (one nonfatal CPU-only DataLoader warning); do not
  conflate this pipeline result with checkpoint task success or promotion readiness.
  Before adding parameter-preservation training, a separately authorized paired diagnostic will
  reuse the original salutation source and Ada-only visible test with checkpoint 010. Compare
  original task wording versus explicit arbitrary-name semantics, holding the executor/contract,
  checkpoint/budget and frozen independent multi-name/exact-content/test-byte gates identical.
  Reference and hardcoded preflights must prove those gates distinguish dynamic behavior.
  Save separate new diagnostic evidence; do not modify fixed v2 or old results, claim fresh unseen
  novelty, or append promotion learning-curve points. Results: both task wordings fail strict and
  multi-name gates. Original wording hardcodes Ada while passing the visible test. Explicit wording
  hardcodes Ada with the wrong Hello prefix, repeats failing pytest instead of repair, and exhausts
  tool rounds. Clarification did not rescue this pair; ambiguous wording alone is not a sufficient
  demonstrated explanation or fix. One pair does not rule out wording sensitivity or prove a sole cause.
  All 779 protected old files and two snapshot files stay unchanged. Existing fixed `2/3` and unseen
  `2/2` remain historical scores; no promotion ledger point is added. Keep this checkpoint unpromoted.
  Literal006 includes one f-string via an external prefix constant, but no inline f-string literal
  edit preserving its placeholder. Prioritize a separately authorized small independent train-only
  increment for that pattern with multiple/empty-name behavior; record recovery only if real.
  Do not reuse validation sources/targets, increase prompt hints, loosen gates, widen the executor,
  or treat additional steps as the automatically appropriate remedy. Coverage is a hypothesis.
  Related pipeline verification passed: 194 distinct tests / 205 executions (eleven repeated
  integrity checks), with one nonfatal CPU-only DataLoader warning; diagnostic scores stay failed.
  The user requested six train-only inline f-string coverage trajectories under a new
  `stage1-codex-powershell-fstring-edit-007` plan/raw version. Preserve all old inputs, use the
  pinned Student/harness and capsule-teacher 0.2.0, and check multiple names plus empty behavior.
  Freeze independent source/target examples, require local literal-only changes and independent
  validation, audit exact historical/validation overlap and save actual tool-result evidence.
  Do not fabricate recovery traces, reuse validation answers, publish, export or train in this phase.
  Collection completed with six train, zero validation, 21 actual recorded tools and six independent
  final validator passes. Multiple/empty-name tests and f-string expression/structure/test-byte
  preservation passed. Known exact duplicates/validation overlap are zero; 803 old files unchanged.
  Save collector incidents separately: literal-suffix AST normalization was fixed and harmless
  colon/slash text must be edited in smaller spans under the unchanged constrained executor.
  Three examples use two edits; all fit the four-tool budget. Raw contains successful observed
  traces, not reconstructed measurements or fabricated failure recovery. This coverage increment
  is not proof the Student learned parameter preservation. Next independently authorize immutable
  publication; SFT composition (increment only versus replay mix), training and evaluation remain
  separate decisions, never automatic continuations.
  All 200 related regression tests passed with one nonfatal CPU-only DataLoader warning. Keep
  raw-review readiness separate from unmeasured Student task success after any future training.
  Independent publication of the six fstring-edit-007 records is authorized: re-audit traces,
  dynamic expressions, multi-input validators, plan/Student/harness digests and known exact overlap.
  Publish only this increment under a fresh canonical four-file version; preserve raw/fixtures and
  old evidence. No automatic replay mix, SFT export, training, evaluation or remote push follows.
  Publication completed independently: six train, zero validation, zero duplicate records in the
  four canonical files. Train is byte-identical to raw (16,429 bytes); all six independent multi-input
  tests and digest/structure/envelope/known exact overlap reviews passed. All 846 protected files
  stayed unchanged. Separate publication evidence is under tests; historical collection flags stay
  untouched. Dataset reload verifies artifact digests; Student capability remains unmeasured here.
  Before an SFT export choose its composition explicitly; retain immutable source manifests and do
  not modify old data, treat publication as training evidence, or silently continue to training.
  All 129 related publication/collection/data/provenance/diagnostic-preservation tests passed;
  the Student's readiness remains unchanged by dataset publication alone.
- The authorized literal006/fstring007 mix contains the original eight plus six train trajectories
  exactly once, not oversampled recovery data. Create only a new canonical publication and schema
  0.3 Qwen SFT, pin source manifests/model/v2 contract, independently validate all fourteen solved
  copies without writing to fixtures, audit all assistant spans and three/four-call boundaries,
  and hash-protect old experimental artifacts. Preserve same-ID origin references in derived
  versions while rejecting changed-content ID collisions and distinct-ID duplicate collections.
  This is input readiness only; training and Student evaluation are separately authorized.
  Overfitting remains unresolved: all fourteen examples are still greeting edits, old-data inclusion
  supports coverage retention rather than guaranteeing generalization, and low training loss is
  not a promotion gate. Do not default to 64 steps or maximize repetitions. Predeclare any new
  training budget and retain the unchanged fixed suite as development regression evidence only.
  Future independent validation must exercise different business scenarios and code structures,
  multiple/empty inputs and unchanged unrelated behavior, frozen before model inference and never
  used in training. Do not repeatedly select checkpoints on a supposedly fresh holdout; report
  selection use explicitly. Same-domain success alone cannot establish cross-project readiness.
  Mix and export completed: 14 train / zero validation / zero duplicates, exact ordered source-record
  bytes, fourteen independent multi-input validator passes. The unchanged schema 0.3/4096-token
  contract yields 9,979 total / 3,093 trainable tokens, max 801, no truncation, 45 audited tool calls
  and 59 matched generation prefixes. All 850 protected prior files stayed byte-identical.
  The original eight exported examples remain unchanged; the new six include every actual edit.
  Source and SFT manifest digests plus decoded-label evidence are persisted in the separate tests
  audit. Dataset/SFT readiness does not update Student scores or establish absence of overfitting;
  no training or model evaluation has started, and promotion remains blocked.
  All 99 related mix/SFT/source/data/provenance regression tests passed without skips. Keep this
  pipeline result separate from model quality and any future heldout-selection evidence.
- Freeze the separately authorized six-case `stage1-codex-mix011-validation-v1` fixture suite and
  original/reference preflights without model inference. Groups: two new f-string structures, two
  branch/composition greeting structures, two non-greeting two-parameter status/event functions.
  Retain executor filenames and all prompts/tools/gates; snapshot sources and strict multi-input
  expectations before the proposed training run. Source and test byte gates retain unrelated code.
  The preregistered pilot is fresh-base mixed-SFT 28 updates / two epochs, seed42, batch/accumulation1,
  LR2e-4 linear decay/no warmup, unchanged LoRA r8/alpha16/dropout0.05 q_proj/v_proj and 4096 length.
  No training-time validation or heldout-based checkpoint selection; final checkpoint only and no
  automatic training extension. Training itself is not authorized by fixture preparation.
  Later separately authorize fixed-v2 new-checkpoint regression and a same-six-case new/old-010
  comparison using separate untouched copies, greedy 256 tokens and four tools. Report subgroup,
  multi-input/structure/test preservation, invalid-call and time-bounded metrics; target fixed3/3,
  new6/6. All six passing still does not rule out overfitting or establish cross-project readiness.
  Exposure and data both differ, so this is not a causal single-variable comparison. Never train on
  validation sources/tasks/references/preflights; acknowledge that repeated selection consumes
  validation independence. Keep all old files and historical scores unchanged.
  Freeze completed: six original failures and six reference passes through 18 actual authorized
  tool calls, with all 858 protected prior files unchanged. Exact isolation review covers 67 train
  task IDs / 51 revisions, 11 previous validation IDs / 9 revisions. Suite and case digests, original
  source hashes, copied-workspace results and the 28-step proposal are persisted in the new fixture
  directory. Pure function gates test multiple/empty inputs independently of strict source equality.
  Compatibility note: preserve the initial preparation failure and support the 88 historical
  plaintext result occurrences alongside JSON envelopes (including raw/publication mirrors; read
  occurrences, not new executions); no old record or prepared snapshot is rewritten.
  No Student inference, training, model-evaluation run or ledger update occurred. Future authorizations
  must reference this frozen suite rather than authoring validation after inspecting model outputs.
  All 120 related regressions passed without skips (91 suite/data/provenance checks plus 29 fixture/
  executor unit tests). Every case rejects synthetic first-input constant output through pure-function
  gates alone. Validator sensitivity is not Student success or proof of no overfitting.
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
