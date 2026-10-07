"""Paired task-clarity diagnostic; never a fresh-unseen or fixed-v2 replacement."""

import json
import sys
from datetime import UTC, datetime
from hashlib import sha256

import train_stage1_literal006_64step as training
from run_stage1_007_evaluation import ProgressGenerator, TOOLS, EXECUTOR, digest
from capability_capsule.eval.coding_checkpoint import (
    CodingEvaluationCase, CodingEvaluationSuite, ConstrainedPowerShellExecutor, TransformersPeftTurnGenerator,
    copy_verified_evaluation_fixture, evaluate_coding_case, load_coding_evaluation_suite, run_coding_validators,
)
from capability_capsule.eval.fixture_provenance import inspect_fixture
from capability_capsule.eval.jsonl import append_jsonl

ROOT = training.ROOT
TRAINING = training.RUN
RUN_ID = training.RUN_ID + "-parameter-diagnostic-v1"
RUN = ROOT / "runs/evaluation" / RUN_ID
SUITE = ROOT / "plans/evaluation/stage1-codex-validation-v2/evaluation-suite.json"
SOURCE_CASE_ID = "validation-powershell-salutation-unseen-001"
GROUPS = ("original-task", "explicit-dynamic-task")
CLARIFICATION = " The same prefix and suffix must apply to every input name; preserve dynamic use of the name argument rather than hardcoding Ada."
CHECKS = (("Ada", "PowerShell ready, Ada!"), ("Mira", "PowerShell ready, Mira!"),
          ("Owen", "PowerShell ready, Owen!"), ("", "PowerShell ready, !"))
save = training.save


def read(path):
    return json.loads(path.read_text("utf-8"))


def protected_hashes():
    hashes = training.protected_hashes()
    for directory in (TRAINING, ROOT / "plans/evaluation"):
        for path in directory.rglob("*"):
            if path.is_file() and not any(part in {"__pycache__", ".pytest_cache"} for part in path.parts):
                hashes[path.relative_to(ROOT).as_posix()] = digest(path)
    prefix = RUN.relative_to(ROOT).as_posix() + "/"
    return {path: value for path, value in hashes.items() if not path.startswith(prefix)}


def source_case():
    fixed = load_coding_evaluation_suite(SUITE)
    return fixed, next(case for case in fixed.cases if case.case_id == SOURCE_CASE_ID)


def prepare():
    if RUN.exists():
        raise FileExistsError("Refusing to overwrite a diagnostic")
    training.verify_preservation()
    reload = read(TRAINING / "adapter-reload-validation.json")
    if not all(reload[key] for key in ("independent_process", "adapter_hashes_verified", "base_model_revision_verified")):
        raise ValueError("Independent reload is required")
    fixed, original = source_case()
    contract = training.baseline.source.contract()
    if fixed.system_prompt != contract.system_prompt or tuple(contract.tools) != TOOLS:
        raise ValueError("Persisted v2 contract mismatch")
    before = protected_hashes()
    RUN.mkdir(parents=True)
    save(RUN / "protected-input-hashes.json", before)
    snapshot = RUN / "fixture"
    copy_verified_evaluation_fixture(original, artifact_root=ROOT, destination=snapshot)
    source = (snapshot / "greeting.py").read_text("utf-8")
    test = (snapshot / "test_greeting.py").read_text("utf-8")
    if source.count("Hello, {name}") != 1:
        raise ValueError("Unexpected source")
    validators = [v.model_dump(mode="json") for v in original.validators]
    validators += [{"validator_id": f"python-assert:dynamic-{index}", "kind": "python_call", "target": "greeting.py",
        "function": "greeting", "arguments": [name], "expected": expected} for index, (name, expected) in enumerate(CHECKS)]
    validators += [{"validator_id": "unchanged:test_greeting.py", "kind": "exact_text", "target": "test_greeting.py", "expected": test}]
    cases = {}
    for group in GROUPS:
        case_id = f"diagnostic-literal010-parameter-{group}-001"
        task = {**original.task.model_dump(mode="json"), "task_id": case_id,
            "task": original.task.task + (CLARIFICATION if group == "explicit-dynamic-task" else ""),
            "validation_ids": [v["validator_id"] for v in validators]}
        case = CodingEvaluationCase.model_validate({"case_id": case_id, "task": task,
            "fixture_root": snapshot.relative_to(ROOT).as_posix(), "validators": validators})
        suite = CodingEvaluationSuite.model_validate({**fixed.model_dump(mode="json"),
            "evaluation_suite_id": f"stage1-codex-parameter-diagnostic-v1-{group}", "cases": [case]})
        directory = RUN / group
        directory.mkdir()
        save(directory / "evaluation-suite.json", suite.model_dump(mode="json"))
        copy_verified_evaluation_fixture(case, artifact_root=ROOT, destination=directory / "workspace")
        cases[group] = case
    if cases[GROUPS[0]].validators != cases[GROUPS[1]].validators:
        raise ValueError("Paired validator mismatch")
    preflights = {}
    for name, replacement in (("original", None), ("reference", "PowerShell ready, {name}!"),
                              ("hardcoded", "PowerShell ready, Ada!")):
        copy = RUN / "preflight" / name
        copy_verified_evaluation_fixture(cases[GROUPS[0]], artifact_root=ROOT, destination=copy)
        if replacement is not None:
            command = f"$text = Get-Content -LiteralPath greeting.py -Raw; $updated = $text.Replace('Hello, {{name}}', '{replacement}'); Set-Content -LiteralPath greeting.py -Value $updated"
            result = ConstrainedPowerShellExecutor().execute(command, copy)
            if not result.authorized or result.exit_code:
                raise ValueError("Preflight edit rejected")
        outcomes = run_coding_validators(cases[GROUPS[0]], copy)
        preflights[name] = [v.model_dump(mode="json") for v in outcomes]
    if preflights["original"][0]["passed"] or not all(v["passed"] for v in preflights["reference"]):
        raise ValueError("Original/reference preflight failed")
    hardcoded = {v["validator_id"]: v["passed"] for v in preflights["hardcoded"]}
    if not hardcoded["pytest:test_greeting.py"] or hardcoded["exact-content:greeting.py"] or hardcoded["python-assert:dynamic-1"]:
        raise ValueError("Hardcoding gate is ineffective")
    save(RUN / "preflight.json", preflights)
    save(RUN / "prepared.json", {"prepared_at": datetime.now(UTC).isoformat(),
        "checkpoint_sha256": digest(TRAINING / "checkpoint.json"), "executor_sha256": digest(EXECUTOR),
        "fixed_suite_sha256": digest(SUITE), "system_prompt_sha256": sha256(fixed.system_prompt.encode()).hexdigest(),
        "tools_sha256": sha256(json.dumps(TOOLS, sort_keys=True).encode()).hexdigest(),
        "chat_contract_sha256": contract.sha256(), "sft_manifest_sha256": digest(training.SFT / "manifest.json"),
        "source_case_id": original.case_id, "source_revision": original.task.fixture_revision,
        "snapshot_hashes": {p.relative_to(ROOT).as_posix(): digest(p) for p in snapshot.iterdir()},
        "groups": {g: {"suite_sha256": digest(RUN / g / "evaluation-suite.json"),
            "case_id": cases[g].case_id} for g in GROUPS},
        "comparison_policy": "Same checkpoint/source/visible tests/validators/contract/budgets; only task clarification changes, apart from bookkeeping IDs.",
        "scope": "Historical validation diagnostic, not fresh unseen, fixed-v2 replacement or promotion evidence",
        "order": list(GROUPS), "replications_per_group": 1, "learning_curve_ledger_written": False})
    print("Frozen paired task descriptions and identical multi-name gates; original/reference/hardcoded preflights passed.", flush=True)


def audit_prepared():
    prepared = read(RUN / "prepared.json")
    before = read(RUN / "protected-input-hashes.json")
    current = protected_hashes()
    if any(current.get(path) != pinned for path, pinned in before.items()):
        raise ValueError("Protected inputs changed")
    if digest(TRAINING / "checkpoint.json") != prepared["checkpoint_sha256"] or digest(EXECUTOR) != prepared["executor_sha256"]:
        raise ValueError("Checkpoint/executor changed")
    for relative, pinned in prepared["snapshot_hashes"].items():
        if digest(ROOT / relative) != pinned:
            raise ValueError("Frozen source changed")
    for group, pins in prepared["groups"].items():
        directory = RUN / group
        if digest(directory / "evaluation-suite.json") != pins["suite_sha256"]:
            raise ValueError("Diagnostic definition changed")
    return prepared, before


def evaluate():
    prepared, before = audit_prepared()
    for group in GROUPS:
        directory = RUN / group
        if (directory / "case-report.json").exists() or (directory / "case-results.jsonl").exists():
            raise FileExistsError("Refusing repeated diagnostic inference")
        case = load_coding_evaluation_suite(directory / "evaluation-suite.json").cases[0]
        if inspect_fixture(case.task.fixture_id, directory / "workspace").revision != case.task.fixture_revision:
            raise ValueError("Workspace is not the frozen original")
    generator = ProgressGenerator(TransformersPeftTurnGenerator(TRAINING / "checkpoint.json", max_new_tokens=256))
    checkpoint, manifest = read(TRAINING / "checkpoint.json"), read(TRAINING / "training-run.json")
    summaries = {}
    for group in GROUPS:
        directory = RUN / group
        suite = load_coding_evaluation_suite(directory / "evaluation-suite.json")
        case = suite.cases[0]
        print(f"Starting diagnostic: {group}", flush=True)
        outcome = evaluate_coding_case(case, workspace=directory / "workspace", generator=generator,
            executor=ConstrainedPowerShellExecutor(), system_prompt=suite.system_prompt, tools=TOOLS,
            experiment_id=RUN_ID, run_id=f"{RUN_ID}:{group}", model_id=checkpoint["output_capsule_id"],
            hardware_id=manifest["hardware_id"], cycle_position=1, max_tool_rounds=4)
        append_jsonl(directory / "case-results.jsonl", outcome.result)
        save(directory / "case-report.json", {"result": outcome.result.model_dump(mode="json"),
            "completions": list(outcome.completions), "tool_results": [v.model_dump(mode="json") for v in outcome.tool_results],
            "validators": [v.model_dump(mode="json") for v in outcome.validators]})
        passes = {v.validator_id: v.passed for v in outcome.validators}
        summary = {"strict_success": outcome.result.success,
            "visible_ada_test_passed": passes["pytest:test_greeting.py"],
            "multi_name_behavior_passed": all(passes[f"python-assert:dynamic-{i}"] for i in range(len(CHECKS))),
            "parameter_preserving_exact_content_passed": passes["exact-content:greeting.py"],
            "test_file_unchanged": passes["unchanged:test_greeting.py"],
            "invalid_tool_call_count": outcome.result.invalid_tool_call_count, "duration_ms": outcome.result.duration_ms,
            "completed_at": datetime.now(UTC).isoformat(), "prepared_at": prepared["prepared_at"],
            "suite_sha256": prepared["groups"][group]["suite_sha256"], "checkpoint_sha256": prepared["checkpoint_sha256"],
            "max_new_tokens": 256, "max_tool_rounds": 4, "do_sample": False, "enable_thinking": False,
            "scope": prepared["scope"]}
        save(directory / "run-manifest.json", summary)
        summaries[group] = summary
        print(json.dumps(summary), flush=True)
    audit_prepared()
    save(RUN / "input-preservation.json", {"unchanged": True, "protected_file_count": len(before), "changed_paths": []})
    save(RUN / "diagnostic-summary.json", {"groups": summaries, "comparison_policy": prepared["comparison_policy"],
        "scope": prepared["scope"], "replications_per_group": 1, "learning_curve_ledger_written": False})


if __name__ == "__main__":
    if sys.argv[1:] == ["--prepare-only"]:
        prepare()
    elif sys.argv[1:] == ["--evaluate-prepared"]:
        evaluate()
    elif not sys.argv[1:]:
        prepare()
        evaluate()
    else:
        raise ValueError("Use --prepare-only or --evaluate-prepared")
