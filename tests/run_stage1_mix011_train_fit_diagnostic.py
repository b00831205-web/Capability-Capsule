"""One-shot free-running diagnostic on train cases, never validation relabeling."""

import json
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from time import monotonic

import prepare_stage1_mix011_train_fit_diagnostic as preparation
from capability_capsule.eval.coding_checkpoint import (
    ConstrainedPowerShellExecutor, TransformersPeftTurnGenerator, parse_qwen_tool_completion,
)
from capability_capsule.eval.fixture_provenance import inspect_fixture
from run_stage1_007_evaluation import ProgressGenerator

ROOT, RUN = preparation.ROOT, preparation.PROPOSED_OUTPUT
read, digest = preparation.read, preparation.digest
DOCS = ("MVP_ROADMAP.md", "PENDING_CONFIGURATION_OPTIMIZATIONS.md")


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def append(path, value):
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, ensure_ascii=False) + "\n")


def protected_hashes():
    prefix = RUN.relative_to(ROOT).as_posix() + "/"
    hashes = {}
    # Enumerate explicitly: older collection/evaluation helpers exclude their
    # then-new outputs, but every one of those outputs is immutable in this run.
    for relative in ("src", "plans/teacher", "plans/student", "plans/harness", "plans/evaluation",
                     "datasets/teacher", "artifacts/sft", "artifacts/teacher-fixtures",
                     "artifacts/evaluation-fixtures", "runs/training", "runs/evaluation",
                     "fixtures/smoke-pilot/validation-greeting-change"):
        for path in (ROOT / relative).rglob("*"):
            name = path.relative_to(ROOT).as_posix()
            if (path.is_file() and not name.startswith(prefix)
                    and not any(p in {"__pycache__", ".pytest_cache"} for p in path.parts)):
                hashes[name] = digest(path)
    return hashes


def verify_preservation():
    before = read(RUN / "protected-input-hashes.json")
    supplemental = RUN / "additional-protected-evaluation-hashes.json"
    if supplemental.exists():
        before.update(read(supplemental))
    after = protected_hashes()
    # Preserve explicitly pinned audit files from older inventories too.
    for path in before.keys() - after.keys():
        source = ROOT / path
        if source.is_file():
            after[path] = digest(source)
    changed = [p for p, pinned in before.items() if after.get(p) != pinned]
    if changed:
        raise ValueError(f"Protected input changed: {changed}")
    return len(before)


def pytest_result(workspace):
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pytest", "-q", "test_greeting.py", "-p", "no:cacheprovider"],
            cwd=workspace, capture_output=True, text=True, timeout=60,
        )
        return {"exit_code": result.returncode, "output": (result.stdout + result.stderr)[-8000:]}
    except subprocess.TimeoutExpired:
        return {"exit_code": 124, "output": "Independent pytest timed out after 60 seconds"}


def copy_source(case, destination):
    source = ROOT / case["fixture_root"]
    if inspect_fixture(case["fixture_id"], source).revision != case["fixture_revision"]:
        raise ValueError("Source fixture changed")
    shutil.copytree(source, destination, ignore=shutil.ignore_patterns("__pycache__", ".pytest_cache", "*.pyc"))
    if inspect_fixture(case["fixture_id"], destination).revision != case["fixture_revision"]:
        raise ValueError("Fixture copy differs")


def reference_source(case):
    # Existing train reference only; never supplied to the Student generator.
    import collect_stage1_literal_edit006 as literal
    import collect_stage1_fstring_edit007 as fstring
    module = literal if case["task_id"].startswith("train-literal-edit-006-") else fstring
    slug = (ROOT / case["fixture_root"]).name
    variant = next(v for v in module.VARIANTS if v.slug == slug)
    source = (ROOT / case["fixture_root"] / "greeting.py").read_text("utf-8")
    if source != variant.source:
        raise ValueError("Original reference definition mismatch")
    expected = source.replace(variant.old, variant.new)
    previous = module.WORKSPACES / slug / "greeting.py"
    if previous.read_text("utf-8") != expected:
        raise ValueError("Existing collected reference differs from intended edit")
    return expected


def prepare():
    if RUN.exists():
        raise FileExistsError("Refusing to overwrite diagnostic")
    plan = preparation.build_plan()
    before = protected_hashes()
    RUN.mkdir(parents=True)
    save(RUN / "protected-input-hashes.json", before)
    save(RUN / "planning-before.json", {p: digest(ROOT / p) for p in DOCS})
    save(RUN / "execution-authorization.json", {
        "model_diagnostic_authorized": True, "planning_updates_authorized": True,
        "training_authorized": False, "old_artifact_edits_authorized": False,
        "scope": "011, fourteen train cases once each, new disposable copies and diagnostic reports",
    })
    save(RUN / "diagnostic-plan.json", plan)
    preflights = []
    for case in plan["cases"]:
        task_id = case["task_id"]
        workspace = RUN / "workspaces" / task_id
        original = RUN / "preflight/original" / task_id
        reference = RUN / "preflight/reference" / task_id
        for target in (workspace, original, reference):
            copy_source(case, target)
        expected = reference_source(case)
        (reference / "greeting.py").write_text(expected, encoding="utf-8", newline="")
        baseline, correct = pytest_result(original), pytest_result(reference)
        report = {"task_id": task_id, "split": "train", "original": baseline, "reference": correct,
                  "reference_source_sha256": digest(reference / "greeting.py")}
        save(RUN / "preflight/reports" / (task_id + ".json"), report)
        if baseline["exit_code"] != 1 or correct["exit_code"] != 0:
            raise ValueError(f"Preflight failed: {task_id}")
        preflights.append(report)
        print(f"Preflight {task_id}: original fails, reference passes", flush=True)
    count = verify_preservation()
    save(RUN / "prepared.json", {"prepared_at": datetime.now(UTC).isoformat(),
        "case_count": len(preflights), "diagnostic_plan_sha256": digest(RUN / "diagnostic-plan.json"),
        "protected_file_count": count, "originals_failed": 14, "references_passed": 14,
        "reference_hashes": {r["task_id"]: r["reference_source_sha256"] for r in preflights}})


def run_case(case, contract, workspace, generator, executor, expected_source, on_turn=lambda row: None):
    if case["split"] != "train":
        raise ValueError("Train-only diagnostic")
    started = monotonic()
    initial_source = digest(workspace / "greeting.py")
    initial_test = digest(workspace / "test_greeting.py")
    messages = preparation.model_messages(contract, case)
    turns, executions = [], []
    reached_final, error = False, None
    try:
        for index in range(5):
            generated = generator.generate(messages, contract["tools"])
            turn = {"turn": index + 1, **generated.model_dump(mode="json")}
            turns.append(turn)
            parsed = parse_qwen_tool_completion(generated.completion)
            if not parsed.tool_calls:
                reached_final = True
                on_turn(turn)
                break
            if len(executions) >= 4:
                error = "ToolRoundLimitExceeded"
                on_turn(turn)
                break
            call = parsed.tool_calls[0]
            if call.name != "exec_command" or not isinstance(call.arguments.get("cmd"), str):
                raise ValueError("Invalid exec_command request")
            command = call.arguments["cmd"]
            execution = executor.execute(command, workspace)
            entry = {"command": command, **execution.model_dump(mode="json")}
            executions.append(entry)
            turn["execution"] = entry
            on_turn(turn)
            messages.append({"role": "assistant", "content": parsed.content, "tool_calls": [
                {"type": "function", "function": {"name": call.name, "arguments": dict(call.arguments)}}]})
            messages.append({"role": "tool", "name": call.name, "content": execution.envelope()})
    except Exception as exc:
        error = type(exc).__name__ + ": " + str(exc)
        on_turn({"error": error})
    tests_unchanged = digest(workspace / "test_greeting.py") == initial_test
    behavior = pytest_result(workspace) if tests_unchanged else {"exit_code": 125, "output": "Test bytes changed"}
    exact = (workspace / "greeting.py").read_text("utf-8") == expected_source
    denied = sum(not row["authorized"] for row in executions)
    strict = reached_final and error is None and tests_unchanged and exact and behavior["exit_code"] == 0 and denied == 0
    duration = (monotonic() - started) * 1000
    return {"task_id": case["task_id"], "split": "train", "seen_during_training": True,
        "strict_success": strict, "behavior_pass": behavior["exit_code"] == 0,
        "exact_source_pass": exact, "tests_unchanged": tests_unchanged,
        "source_changed": digest(workspace / "greeting.py") != initial_source,
        "model_invoked_pytest": any(row["authorized"] and row["command"].strip().lower() in {
            "pytest -q", "pytest -q test_greeting.py", "python -m pytest -q", "python -m pytest -q test_greeting.py"
        } for row in executions),
        "denied_command_count": denied, "tool_call_count": len(executions),
        "generation_turn_count": len(turns), "reached_final": reached_final,
        "error": error, "duration_ms": duration, "time_bounded_success": strict and duration <= 180000,
        "turns": turns, "tool_results": executions, "independent_pytest": behavior}


def summarize(reports):
    return {"case_count": len(reports), **{key: sum(r[key] for r in reports) for key in (
        "strict_success", "behavior_pass", "exact_source_pass", "tests_unchanged", "source_changed",
        "model_invoked_pytest", "denied_command_count", "generation_turn_count", "time_bounded_success")}}


def record_interruption():
    evidence = read(RUN / "model-load-interruption.json")
    if evidence["diagnostic_process_found_after_crash"] or (RUN / "case-results.jsonl").exists() or (RUN / "generation-log.jsonl").exists():
        raise ValueError("Interruption record requires no recorded model output and an exited process")
    for case in read(RUN / "diagnostic-plan.json")["cases"]:
        workspace = RUN / "workspaces" / case["task_id"]
        if inspect_fixture(case["fixture_id"], workspace).revision != case["fixture_revision"]:
            raise ValueError("An interrupted workspace changed; manual review required")
    count = verify_preservation()
    save(RUN / "input-preservation.json", {"unchanged": True, "protected_file_count": count,
        "changed_paths": [], "planning_files_excluded_by_current_authorization": list(DOCS),
        "model_diagnostic_completed": False, "all_14_model_workspaces_unmodified": True})
    save(RUN / "planning-update.json", {"explicitly_authorized": True,
        "before": read(RUN / "planning-before.json"), "after": {p: digest(ROOT / p) for p in DOCS},
        "reason": "Record actual training/evaluation history and the authorized diagnostic's host-resource interruption"})
    print(f"Recorded interruption, no model scores; {count} protected inputs and 14 model copies unchanged", flush=True)


def evaluate():
    if (RUN / "inference-attempt.json").exists():
        raise FileExistsError("Refusing repeated inference")
    prepared = read(RUN / "prepared.json")
    if digest(RUN / "diagnostic-plan.json") != prepared["diagnostic_plan_sha256"]:
        raise ValueError("Prepared diagnostic plan changed")
    preparation.build_plan()
    # Seal any previous-run files absent from the initial preparation inventory
    # before the first model load. This is additive; original hashes stay intact.
    additional = {p: d for p, d in protected_hashes().items()
                  if p not in read(RUN / "protected-input-hashes.json")}
    if additional:
        supplemental = RUN / "additional-protected-evaluation-hashes.json"
        if supplemental.exists():
            if read(supplemental) != additional:
                raise ValueError("Additional protected input inventory changed")
        else:
            save(supplemental, additional)
    verify_preservation()
    plan = read(RUN / "diagnostic-plan.json")
    for case in plan["cases"]:
        workspace = RUN / "workspaces" / case["task_id"]
        if inspect_fixture(case["fixture_id"], workspace).revision != case["fixture_revision"]:
            raise ValueError("Diagnostic workspace already changed")
        if digest(RUN / "preflight/reference" / case["task_id"] / "greeting.py") != prepared["reference_hashes"][case["task_id"]]:
            raise ValueError("Prepared reference changed")
    save(RUN / "inference-attempt.json", {"started_at": datetime.now(UTC).isoformat(), "one_shot": True,
         "checkpoint_sha256": preparation.PINS[preparation.CHECKPOINT]})
    print(f"Loading {preparation.RUN_ID} for fourteen train-only cases", flush=True)
    generator = ProgressGenerator(TransformersPeftTurnGenerator(preparation.CHECKPOINT, max_new_tokens=256))
    reports = []
    for case in plan["cases"]:
        print(f"Starting {len(reports) + 1}/14: {case['task_id']}", flush=True)
        reference = (RUN / "preflight/reference" / case["task_id"] / "greeting.py").read_text("utf-8")
        report = run_case(case, plan["contract"], RUN / "workspaces" / case["task_id"], generator,
            ConstrainedPowerShellExecutor(), reference,
            on_turn=lambda turn: append(RUN / "generation-log.jsonl", {"task_id": case["task_id"], **turn}))
        save(RUN / "case-reports" / (case["task_id"] + ".json"), report)
        append(RUN / "case-results.jsonl", {k: v for k, v in report.items() if k not in {"turns", "tool_results", "independent_pytest"}})
        reports.append(report)
        print(f"strict={report['strict_success']}, behavior={report['behavior_pass']}, denied={report['denied_command_count']}", flush=True)
    count = verify_preservation()
    summary = {**summarize(reports), "split": "train", "seen_during_training": True,
        "groups": {"literal006": summarize(reports[:8]), "fstring007": summarize(reports[8:])},
        "checkpoint_id": preparation.RUN_ID, "checkpoint_promoted": False, "reruns": 0,
        "completed_at": datetime.now(UTC).isoformat(),
        "interpretation": "Seen-task fit diagnostic under unchanged executor; cannot measure unseen generalization or isolate exposure causality"}
    save(RUN / "summary.json", summary)
    save(RUN / "input-preservation.json", {"unchanged": True, "protected_file_count": count,
        "changed_paths": [], "planning_files_excluded_by_current_authorization": list(DOCS)})
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    options = {"--prepare-only": prepare, "--evaluate": evaluate, "--record-interruption": record_interruption}
    if len(sys.argv) != 2 or sys.argv[1] not in options:
        raise ValueError("Use --prepare-only, --evaluate or --record-interruption")
    options[sys.argv[1]]()
