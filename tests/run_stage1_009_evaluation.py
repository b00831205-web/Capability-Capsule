"""Freeze fixed v2 plus fresh validation cases before evaluating checkpoint 009."""

import json
import shutil
import sys
from datetime import UTC, datetime
from hashlib import sha256

import train_stage1_literal006_32step as training
from run_stage1_007_evaluation import ProgressGenerator, TOOLS, EXECUTOR, digest, published_training_provenance
from capability_capsule.eval.coding_checkpoint import (
    CodingEvaluationCase, CodingEvaluationSuite, ConstrainedPowerShellExecutor, TransformersPeftTurnGenerator,
    copy_verified_evaluation_fixture, evaluate_coding_case, load_coding_evaluation_suite, run_coding_validators,
)
from capability_capsule.eval.fixture_provenance import inspect_fixture
from capability_capsule.eval.jsonl import append_jsonl
from capability_capsule.eval.learning_curve_ledger import append_learning_curve_point
from capability_capsule.eval.learning_rate import LearningCurvePoint

ROOT = training.ROOT
RUN_ID = training.RUN_ID
TRAINING = training.RUN
RUN = ROOT / "runs/evaluation" / RUN_ID
SUITE = ROOT / "plans/evaluation/stage1-codex-validation-v2/evaluation-suite.json"
GROUPS = ("fixed-v2", "unseen-v1")

# Authored after checkpoint 009 completed, before any checkpoint inference on these cases.
UNSEEN = (
    ("trimmed-dictionary",
     'ANNOUNCEMENTS = {"public": "Salute, {visitor}!", "private": "Private channel"}\n\n'
     'def greeting(name: str) -> str:\n    template = ANNOUNCEMENTS["public"]\n    return template.format(visitor=name.strip())\n',
     "Salute, {visitor}!", "Clear horizon, {visitor}.",
     (("  Lark  ", "Clear horizon, Lark."), ("Evan", "Clear horizon, Evan.")),
     'Change greeting.py only: greeting("  Lark  ") must return "Clear horizon, Lark." and greeting("Evan") '
     'must return "Clear horizon, Evan." Preserve name trimming, the dictionary entries and template formatting. '
     'Run the supplied tests without editing test_greeting.py.'),
    ("upper-tuple",
     '# Preserve uppercase rendering and the unnamed-visitor branch.\n'
     'def greeting(name: str) -> str:\n    if not name:\n        return "Guest not specified"\n'
     '    display_name = name.upper()\n    chunks = ("Hail", ", ", display_name, "!")\n    return "".join(chunks)\n',
     "Hail", "Forward together",
     (("ivy", "Forward together, IVY!"), ("Teo", "Forward together, TEO!"), ("", "Guest not specified")),
     'Change greeting.py only so ivy and Teo receive "Forward together, IVY!" and "Forward together, TEO!". '
     'Keep uppercase conversion, tuple assembly and joining, the comment, and the empty-name result unchanged. '
     'Run the supplied tests without changing test_greeting.py.'),
)


def save(path, value):
    training.save(path, value)


def protected_hashes():
    hashes = training.protected_hashes()
    for directory in (TRAINING, ROOT / "plans/evaluation"):
        for path in directory.rglob("*"):
            if path.is_file() and not any(part in {"__pycache__", ".pytest_cache"} for part in path.parts):
                hashes[path.relative_to(ROOT).as_posix()] = digest(path)
    prefix = RUN.relative_to(ROOT).as_posix() + "/"
    return {path: value for path, value in hashes.items() if not path.startswith(prefix)}


def audit_unseen(suite):
    revisions, task_ids, sources = published_training_provenance()
    known_validation_sources, known_validation_ids = set(), set()
    for path in (ROOT / "runs/evaluation").rglob("evaluation-suite.json"):
        if RUN in path.parents:
            continue
        for case in load_coding_evaluation_suite(path).cases:
            known_validation_ids.add(case.case_id)
            source = ROOT / case.fixture_root / "greeting.py"
            if source.exists():
                known_validation_sources.add(source.read_text("utf-8"))
    training_text = "\n".join(path.read_text("utf-8") for path in (ROOT / "datasets/teacher/published").glob("*/train.jsonl"))
    cases = []
    for case, variant in zip(suite.cases, UNSEEN, strict=True):
        source = (ROOT / case.fixture_root / "greeting.py").read_text("utf-8")
        if (case.task.fixture_revision in revisions or case.task.task_id in task_ids or source in sources
                or case.case_id in known_validation_ids or source in known_validation_sources):
            raise ValueError("Fresh case has known exact training/previous-validation overlap")
        if variant[3] in training_text or any(expected in training_text for _, expected in variant[4] if expected != "Guest not specified"):
            raise ValueError("Fresh target text already occurs in published training data")
        cases.append({"case_id": case.case_id, "source_revision": case.task.fixture_revision,
                      "task_id": case.task.task_id, "source_sha256": digest(ROOT / case.fixture_root / "greeting.py")})
    return {"published_training_revision_count": len(revisions), "published_training_task_count": len(task_ids),
        "previous_validation_case_count": len(known_validation_ids), "exact_overlap": False, "cases": cases,
        "scope": "New exact sources/tasks/targets, first inference after checkpoint freeze; same greeting domain, not semantic independence or cross-project generalization."}


def prepare():
    if RUN.exists():
        raise FileExistsError("Refusing to overwrite an evaluation")
    training.verify_preservation()
    reload = json.loads((TRAINING / "adapter-reload-validation.json").read_text("utf-8"))
    if not reload["independent_process"] or not reload["adapter_hashes_verified"]:
        raise ValueError("Independent checkpoint reload is required")
    fixed = load_coding_evaluation_suite(SUITE)
    contract = training.source.contract()
    if fixed.system_prompt != contract.system_prompt or tuple(contract.tools) != TOOLS:
        raise ValueError("Fixed v2 must match persisted training contract")
    before = protected_hashes()
    RUN.mkdir(parents=True)
    save(RUN / "protected-input-hashes.json", before)
    fixed_dir = RUN / "fixed-v2"
    fixed_dir.mkdir()
    shutil.copyfile(SUITE, fixed_dir / "evaluation-suite.json")
    sources = {}
    for case in fixed.cases:
        source = ROOT / case.fixture_root
        if inspect_fixture(case.task.fixture_id, source).revision != case.task.fixture_revision:
            source = ROOT / "runs/evaluation/stage1-codex-qwen35-2b-006-turns-v2-v2/workspaces" / case.case_id
        relative = source.relative_to(ROOT).as_posix()
        copy_case = CodingEvaluationCase.model_validate({**case.model_dump(mode="json"), "fixture_root": relative})
        copy_verified_evaluation_fixture(copy_case, artifact_root=ROOT, destination=fixed_dir / "workspaces" / case.case_id)
        sources[case.case_id] = relative
    unseen_dir = RUN / "unseen-v1"
    cases, snapshot_hashes = [], {}
    for slug, source, old, new, checks, task_text in UNSEEN:
        case_id = f"validation-literal009-{slug}-001"
        snapshot = unseen_dir / "fixtures" / slug
        snapshot.mkdir(parents=True)
        (snapshot / "greeting.py").write_bytes(source.encode())
        test = "from greeting import greeting\n\ndef test_greeting():\n" + "".join(
            f"    assert greeting({name!r}) == {expected!r}\n" for name, expected in checks)
        if slug == "trimmed-dictionary":
            test += '\n    from greeting import ANNOUNCEMENTS\n    assert ANNOUNCEMENTS["private"] == "Private channel"\n'
        (snapshot / "test_greeting.py").write_bytes(test.encode())
        if source.count(old) != 1:
            raise ValueError("Ambiguous reference edit")
        revision = inspect_fixture(case_id, snapshot).revision
        validators = [
            {"validator_id": "pytest:test_greeting.py", "kind": "pytest", "target": "test_greeting.py"},
            {"validator_id": "exact-content:greeting.py", "kind": "exact_text", "target": "greeting.py", "expected": source.replace(old, new)},
            {"validator_id": "unchanged:test_greeting.py", "kind": "exact_text", "target": "test_greeting.py", "expected": test},
        ]
        task = {**fixed.cases[0].task.model_dump(mode="json"), "task_id": case_id, "fixture_id": case_id,
            "fixture_family_id": f"unseen-literal009-{slug}", "fixture_revision": revision,
            "defect_family": f"unseen-{slug}", "task": task_text, "time_limit": "PT3M",
            "knowledge_distance": 0.4, "validation_ids": [v["validator_id"] for v in validators]}
        case = CodingEvaluationCase.model_validate({"case_id": case_id,
            "fixture_root": snapshot.relative_to(ROOT).as_posix(), "task": task, "validators": validators})
        cases.append(case)
        for path in snapshot.iterdir():
            snapshot_hashes[path.relative_to(ROOT).as_posix()] = digest(path)
        copy_verified_evaluation_fixture(case, artifact_root=ROOT, destination=unseen_dir / "workspaces" / case_id)
        preflight = unseen_dir / "preflight" / case_id
        copy_verified_evaluation_fixture(case, artifact_root=ROOT, destination=preflight)
        original = run_coding_validators(case, preflight)
        command = ("$text = Get-Content -LiteralPath greeting.py -Raw; $updated = $text.Replace('"
            + old + "', '" + new + "'); Set-Content -LiteralPath greeting.py -Value $updated")
        executed = ConstrainedPowerShellExecutor().execute(command, preflight)
        corrected = run_coding_validators(case, preflight)
        if original[0].passed or not executed.authorized or executed.exit_code != 0 or not all(v.passed for v in corrected):
            raise ValueError("Frozen unseen validators failed preflight")
        save(unseen_dir / f"preflight-{slug}.json", {"original": [v.model_dump(mode="json") for v in original],
            "corrected": [v.model_dump(mode="json") for v in corrected], "execution": executed.model_dump(mode="json")})
    unseen = CodingEvaluationSuite.model_validate({**fixed.model_dump(mode="json"),
        "evaluation_suite_id": "stage1-codex-validation-literal009-unseen-v1", "cases": cases})
    save(unseen_dir / "evaluation-suite.json", unseen.model_dump(mode="json"))
    save(RUN / "unseen-overlap-audit.json", audit_unseen(unseen))
    groups = {"fixed-v2": fixed, "unseen-v1": unseen}
    save(RUN / "prepared.json", {"prepared_at": datetime.now(UTC).isoformat(),
        "checkpoint_sha256": digest(TRAINING / "checkpoint.json"), "executor_sha256": digest(EXECUTOR),
        "system_prompt_sha256": sha256(contract.system_prompt.encode()).hexdigest(),
        "tools_sha256": sha256(json.dumps(TOOLS, sort_keys=True).encode()).hexdigest(),
        "chat_contract_sha256": contract.sha256(), "sft_manifest_sha256": digest(training.SFT / "manifest.json"),
        "fixed_fixture_sources": sources, "unseen_snapshot_hashes": snapshot_hashes,
        "groups": {name: {"suite_sha256": digest(RUN / name / "evaluation-suite.json"),
            "case_digest": suite.identity().evaluation_suite_digest} for name, suite in groups.items()}})
    print("Frozen identical 3 fixed-v2 and 2 first-use unseen validation cases; preflights passed.", flush=True)


def evaluate():
    prepared = json.loads((RUN / "prepared.json").read_text("utf-8"))
    before = json.loads((RUN / "protected-input-hashes.json").read_text("utf-8"))
    if before != protected_hashes():
        raise ValueError("Protected inputs changed after preparation")
    if digest(TRAINING / "checkpoint.json") != prepared["checkpoint_sha256"] or digest(EXECUTOR) != prepared["executor_sha256"]:
        raise ValueError("Checkpoint or executor changed")
    for relative, pinned in prepared["unseen_snapshot_hashes"].items():
        if digest(ROOT / relative) != pinned:
            raise ValueError("Unseen source snapshot changed")
    for group, pins in prepared["groups"].items():
        directory = RUN / group
        if (directory / "case-results.jsonl").exists() or (directory / "case-reports").exists():
            raise FileExistsError("Refusing to repeat or overwrite checkpoint inference")
        if digest(directory / "evaluation-suite.json") != pins["suite_sha256"]:
            raise ValueError("Frozen suite changed")
        suite = load_coding_evaluation_suite(directory / "evaluation-suite.json")
        for case in suite.cases:
            if inspect_fixture(case.task.fixture_id, directory / "workspaces" / case.case_id).revision != case.task.fixture_revision:
                raise ValueError("Inference workspace is not the original frozen source")
    generator = ProgressGenerator(TransformersPeftTurnGenerator(TRAINING / "checkpoint.json", max_new_tokens=256))
    train_manifest = json.loads((TRAINING / "training-run.json").read_text("utf-8"))
    checkpoint = json.loads((TRAINING / "checkpoint.json").read_text("utf-8"))
    summaries = {}
    for group, pins in prepared["groups"].items():
        directory = RUN / group
        suite = load_coding_evaluation_suite(directory / "evaluation-suite.json")
        (directory / "case-reports").mkdir()
        results, behavior_passes = [], 0
        for position, case in enumerate(suite.cases, 1):
            print(f"Starting {group}: {case.case_id}", flush=True)
            outcome = evaluate_coding_case(case, workspace=directory / "workspaces" / case.case_id,
                generator=generator, executor=ConstrainedPowerShellExecutor(), system_prompt=suite.system_prompt,
                tools=TOOLS, experiment_id=f"{RUN_ID}:{group}", run_id=f"{RUN_ID}:{group}:{case.case_id}",
                model_id=checkpoint["output_capsule_id"], hardware_id=train_manifest["hardware_id"],
                cycle_position=position, max_tool_rounds=4)
            append_jsonl(directory / "case-results.jsonl", outcome.result)
            save(directory / "case-reports" / f"{case.case_id}.json", {"result": outcome.result.model_dump(mode="json"),
                "completions": list(outcome.completions), "tool_results": [r.model_dump(mode="json") for r in outcome.tool_results],
                "validators": [v.model_dump(mode="json") for v in outcome.validators]})
            results.append(outcome.result)
            behavior_passes += all(v.passed for v in outcome.validators if v.validator_id.startswith(("pytest:", "python-assert:")))
            print(f"success={outcome.result.success}, invalid={outcome.result.invalid_tool_call_count}, error={outcome.result.error_type}", flush=True)
        count = sum(r.success for r in results)
        stats = train_manifest["dataset_stats"]["train"]
        append_learning_curve_point(directory / "learning-curve.jsonl", LearningCurvePoint(
            checkpoint_id=f"{RUN_ID}-step-{checkpoint['step']}", training_run_id=RUN_ID,
            base_model_id=checkpoint["base_model_id"], capability_id=suite.capability_id,
            evaluation_suite_id=suite.evaluation_suite_id, evaluation_suite_digest=suite.identity().evaluation_suite_digest,
            task_family_id="greeting-fixed-v2" if group == "fixed-v2" else "greeting-unseen-trimmed-dictionary-upper-tuple",
            evaluation_split="validation", cumulative_trajectory_count=stats["trajectory_count"],
            cumulative_token_count=stats["exact_token_count"], tokenizer_id=stats["tokenizer_id"], success_rate=count / len(results)))
        manifest = {**pins, "checkpoint_sha256": prepared["checkpoint_sha256"], "executor_sha256": prepared["executor_sha256"],
            "system_prompt_sha256": prepared["system_prompt_sha256"], "tools_sha256": prepared["tools_sha256"],
            "max_new_tokens": 256, "max_tool_rounds": 4, "do_sample": False, "enable_thinking": False,
            "case_count": len(results), "success_count": count, "behavior_pass_count": behavior_passes,
            "time_bounded_success_count": sum(r.time_bounded_success for r in results),
            "prepared_at": prepared["prepared_at"], "completed_at": datetime.now(UTC).isoformat(),
            "validation_scope": "Frozen repeated fixed suite" if group == "fixed-v2" else "First-use exact fixtures in the same greeting domain, not broad semantic independence",
            "ledger_scope": "009 fresh-base literal006-only run: 8 examples / 5506 tokens, not accumulated across prior runs"}
        save(directory / "run-manifest.json", manifest)
        summaries[group] = manifest
    after = protected_hashes()
    changed = sorted(path for path in set(before) | set(after) if before.get(path) != after.get(path))
    for relative, pinned in prepared["unseen_snapshot_hashes"].items():
        if digest(ROOT / relative) != pinned:
            changed.append(relative)
    save(RUN / "input-preservation.json", {"unchanged": not changed, "protected_file_count": len(before), "changed_paths": changed})
    if changed:
        raise ValueError("Protected inputs or unseen snapshots changed")
    save(RUN / "evaluation-summary.json", summaries)


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
