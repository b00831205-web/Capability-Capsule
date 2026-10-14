"""Frozen fixed-v2 and first-use validation for the literal006 64-step control."""

import json
import shutil
import sys
from datetime import UTC, datetime
from hashlib import sha256

import train_stage1_literal006_64step as training
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
UNSEEN = (
    ("casefold-dictionary",
     '# Keep the case-folded name and the unused template.\n'
     'PATTERNS = {"main": "Good day, {name}!", "spare": "Reserved wording"}\n\n'
     'def greeting(name: str) -> str:\n    normalized = name.casefold()\n'
     '    return PATTERNS["main"].format(name=normalized)\n',
     "Good day, {name}!", "Open skies, {name}!",
     (("UMA", "Open skies, uma!"), ("NiLs", "Open skies, nils!")),
     'Change greeting.py only so greeting("UMA") returns "Open skies, uma!" and '
     'greeting("NiLs") returns "Open skies, nils!". Preserve case folding, both dictionary '
     'entries, template formatting and the comment. Run the supplied tests without changing test_greeting.py.'),
    ("trim-title-tuple",
     '# Preserve whitespace handling, title casing and the unnamed branch.\n'
     'def greeting(name: str) -> str:\n    cleaned = name.strip()\n'
     '    if not cleaned:\n        return "No name supplied"\n'
     '    display = cleaned.title()\n    parts = ("Well met", ", ", display, ".")\n'
     '    return "".join(parts)\n',
     "Well met", "Fresh momentum",
     (("  juno  ", "Fresh momentum, Juno."), ("KAI", "Fresh momentum, Kai."), ("   ", "No name supplied")),
     'Edit greeting.py only so greeting("  juno  ") returns "Fresh momentum, Juno." and '
     'greeting("KAI") returns "Fresh momentum, Kai.". Keep trimming, title casing, tuple assembly '
     'and joining, the comment and the whitespace-only result "No name supplied" unchanged. '
     'Run the supplied tests without editing test_greeting.py.'),
)
save = training.save


def protected_hashes():
    hashes = training.protected_hashes()
    for directory in (TRAINING, ROOT / "plans/evaluation"):
        for path in directory.rglob("*"):
            if path.is_file() and not any(p in {"__pycache__", ".pytest_cache"} for p in path.parts):
                hashes[path.relative_to(ROOT).as_posix()] = digest(path)
    prefix = RUN.relative_to(ROOT).as_posix() + "/"
    return {path: value for path, value in hashes.items() if not path.startswith(prefix)}


def audit_unseen(suite):
    revisions, task_ids, sources = published_training_provenance()
    prior_ids, prior_sources = set(), set()
    for path in (ROOT / "runs/evaluation").rglob("evaluation-suite.json"):
        if RUN in path.parents:
            continue
        for case in load_coding_evaluation_suite(path).cases:
            prior_ids.add(case.case_id)
            source = ROOT / case.fixture_root / "greeting.py"
            if source.exists():
                prior_sources.add(source.read_text("utf-8"))
    train_text = "\n".join(path.read_text("utf-8") for path in (ROOT / "datasets/teacher/published").glob("*/train.jsonl"))
    cases = []
    for case, variant in zip(suite.cases, UNSEEN, strict=True):
        source = (ROOT / case.fixture_root / "greeting.py").read_text("utf-8")
        if (case.task.fixture_revision in revisions or case.task.task_id in task_ids or source in sources
                or case.case_id in prior_ids or source in prior_sources):
            raise ValueError("Known exact training/previous-validation overlap")
        targets = [variant[3], *(expected for _, expected in variant[4] if expected != "No name supplied")]
        if any(target in train_text for target in targets):
            raise ValueError("Fresh target already occurs in published train data")
        cases.append({"case_id": case.case_id, "source_revision": case.task.fixture_revision,
            "task_id": case.task.task_id, "source_sha256": digest(ROOT / case.fixture_root / "greeting.py")})
    return {"published_training_revision_count": len(revisions), "published_training_task_count": len(task_ids),
        "previous_validation_case_count": len(prior_ids), "exact_overlap": False, "cases": cases,
        "scope": "First-use exact sources/tasks/targets in the same greeting domain; not semantic independence or cross-project generalization."}


def prepare():
    if RUN.exists():
        raise FileExistsError("Refusing to overwrite an evaluation")
    training.verify_preservation()
    reload = json.loads((TRAINING / "adapter-reload-validation.json").read_text("utf-8"))
    if not all(reload[key] for key in ("independent_process", "adapter_hashes_verified", "base_model_revision_verified")):
        raise ValueError("Independent reload is required")
    fixed = load_coding_evaluation_suite(SUITE)
    contract = training.baseline.source.contract()
    if fixed.system_prompt != contract.system_prompt or tuple(contract.tools) != TOOLS:
        raise ValueError("Fixed suite differs from persisted SFT contract")
    RUN.mkdir(parents=True)
    save(RUN / "protected-input-hashes.json", protected_hashes())
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
    cases, snapshots = [], {}
    for slug, source, old, new, checks, task_text in UNSEEN:
        case_id = f"validation-literal010-{slug}-001"
        snapshot = unseen_dir / "fixtures" / slug
        snapshot.mkdir(parents=True)
        (snapshot / "greeting.py").write_bytes(source.encode())
        test = "from greeting import greeting\n\ndef test_greeting():\n" + "".join(
            f"    assert greeting({name!r}) == {expected!r}\n" for name, expected in checks)
        if slug == "casefold-dictionary":
            test += '\n    from greeting import PATTERNS\n    assert PATTERNS["spare"] == "Reserved wording"\n'
        (snapshot / "test_greeting.py").write_bytes(test.encode())
        if source.count(old) != 1:
            raise ValueError("Ambiguous reference edit")
        validators = [
            {"validator_id": "pytest:test_greeting.py", "kind": "pytest", "target": "test_greeting.py"},
            {"validator_id": "exact-content:greeting.py", "kind": "exact_text", "target": "greeting.py", "expected": source.replace(old, new)},
            {"validator_id": "unchanged:test_greeting.py", "kind": "exact_text", "target": "test_greeting.py", "expected": test},
        ]
        task = {**fixed.cases[0].task.model_dump(mode="json"), "task_id": case_id, "fixture_id": case_id,
            "fixture_family_id": f"unseen-literal010-{slug}", "fixture_revision": inspect_fixture(case_id, snapshot).revision,
            "defect_family": f"unseen-{slug}", "task": task_text, "time_limit": "PT3M", "knowledge_distance": 0.4,
            "validation_ids": [v["validator_id"] for v in validators]}
        case = CodingEvaluationCase.model_validate({"case_id": case_id,
            "fixture_root": snapshot.relative_to(ROOT).as_posix(), "task": task, "validators": validators})
        cases.append(case)
        for path in snapshot.iterdir():
            snapshots[path.relative_to(ROOT).as_posix()] = digest(path)
        copy_verified_evaluation_fixture(case, artifact_root=ROOT, destination=unseen_dir / "workspaces" / case_id)
        preflight = unseen_dir / "preflight" / case_id
        copy_verified_evaluation_fixture(case, artifact_root=ROOT, destination=preflight)
        original = run_coding_validators(case, preflight)
        command = f"$text = Get-Content -LiteralPath greeting.py -Raw; $updated = $text.Replace('{old}', '{new}'); Set-Content -LiteralPath greeting.py -Value $updated"
        executed = ConstrainedPowerShellExecutor().execute(command, preflight)
        corrected = run_coding_validators(case, preflight)
        if original[0].passed or not executed.authorized or executed.exit_code != 0 or not all(v.passed for v in corrected):
            raise ValueError("Unseen preflight failed")
        save(unseen_dir / f"preflight-{slug}.json", {"original": [v.model_dump(mode="json") for v in original],
            "corrected": [v.model_dump(mode="json") for v in corrected], "execution": executed.model_dump(mode="json")})
    unseen = CodingEvaluationSuite.model_validate({**fixed.model_dump(mode="json"),
        "evaluation_suite_id": "stage1-codex-validation-literal010-unseen-v1", "cases": cases})
    save(unseen_dir / "evaluation-suite.json", unseen.model_dump(mode="json"))
    save(RUN / "unseen-overlap-audit.json", audit_unseen(unseen))
    groups = {"fixed-v2": fixed, "unseen-v1": unseen}
    save(RUN / "prepared.json", {"prepared_at": datetime.now(UTC).isoformat(),
        "checkpoint_sha256": digest(TRAINING / "checkpoint.json"), "executor_sha256": digest(EXECUTOR),
        "system_prompt_sha256": sha256(contract.system_prompt.encode()).hexdigest(),
        "tools_sha256": sha256(json.dumps(TOOLS, sort_keys=True).encode()).hexdigest(),
        "chat_contract_sha256": contract.sha256(), "sft_manifest_sha256": digest(training.SFT / "manifest.json"),
        "fixed_fixture_sources": sources, "unseen_snapshot_hashes": snapshots,
        "groups": {name: {"suite_sha256": digest(RUN / name / "evaluation-suite.json"),
            "case_digest": suite.identity().evaluation_suite_digest} for name, suite in groups.items()}})
    print("Frozen unchanged 3 fixed-v2 + 2 first-use unseen cases; reference preflights passed.", flush=True)


def evaluate():
    prepared = json.loads((RUN / "prepared.json").read_text("utf-8"))
    before = json.loads((RUN / "protected-input-hashes.json").read_text("utf-8"))
    if before != protected_hashes():
        raise ValueError("Protected inputs changed after freeze")
    if digest(TRAINING / "checkpoint.json") != prepared["checkpoint_sha256"] or digest(EXECUTOR) != prepared["executor_sha256"]:
        raise ValueError("Checkpoint/executor changed")
    for relative, pinned in prepared["unseen_snapshot_hashes"].items():
        if digest(ROOT / relative) != pinned:
            raise ValueError("Unseen snapshot changed")
    for group, pins in prepared["groups"].items():
        directory = RUN / group
        if (directory / "case-results.jsonl").exists() or (directory / "case-reports").exists():
            raise FileExistsError("Refusing to overwrite or repeat inference")
        if digest(directory / "evaluation-suite.json") != pins["suite_sha256"]:
            raise ValueError("Frozen suite changed")
        for case in load_coding_evaluation_suite(directory / "evaluation-suite.json").cases:
            if inspect_fixture(case.task.fixture_id, directory / "workspaces" / case.case_id).revision != case.task.fixture_revision:
                raise ValueError("Inference copy is not the frozen original")
    generator = ProgressGenerator(TransformersPeftTurnGenerator(TRAINING / "checkpoint.json", max_new_tokens=256))
    manifest = json.loads((TRAINING / "training-run.json").read_text("utf-8"))
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
                model_id=checkpoint["output_capsule_id"], hardware_id=manifest["hardware_id"],
                cycle_position=position, max_tool_rounds=4)
            append_jsonl(directory / "case-results.jsonl", outcome.result)
            save(directory / "case-reports" / f"{case.case_id}.json", {"result": outcome.result.model_dump(mode="json"),
                "completions": list(outcome.completions), "tool_results": [r.model_dump(mode="json") for r in outcome.tool_results],
                "validators": [v.model_dump(mode="json") for v in outcome.validators]})
            results.append(outcome.result)
            behavior_passes += all(v.passed for v in outcome.validators if v.validator_id.startswith(("pytest:", "python-assert:")))
            print(f"success={outcome.result.success}, invalid={outcome.result.invalid_tool_call_count}, error={outcome.result.error_type}", flush=True)
        count = sum(r.success for r in results)
        stats = manifest["dataset_stats"]["train"]
        append_learning_curve_point(directory / "learning-curve.jsonl", LearningCurvePoint(
            checkpoint_id=f"{RUN_ID}-step-{checkpoint['step']}", training_run_id=RUN_ID,
            base_model_id=checkpoint["base_model_id"], capability_id=suite.capability_id,
            evaluation_suite_id=suite.evaluation_suite_id, evaluation_suite_digest=suite.identity().evaluation_suite_digest,
            task_family_id="greeting-fixed-v2" if group == "fixed-v2" else "greeting-unseen-casefold-dictionary-trim-title-tuple",
            evaluation_split="validation", cumulative_trajectory_count=stats["trajectory_count"],
            cumulative_token_count=stats["exact_token_count"], tokenizer_id=stats["tokenizer_id"], success_rate=count / len(results)))
        report = {**pins, "checkpoint_sha256": prepared["checkpoint_sha256"], "executor_sha256": prepared["executor_sha256"],
            "system_prompt_sha256": prepared["system_prompt_sha256"], "tools_sha256": prepared["tools_sha256"],
            "max_new_tokens": 256, "max_tool_rounds": 4, "do_sample": False, "enable_thinking": False,
            "case_count": len(results), "success_count": count, "behavior_pass_count": behavior_passes,
            "time_bounded_success_count": sum(r.time_bounded_success for r in results),
            "prepared_at": prepared["prepared_at"], "completed_at": datetime.now(UTC).isoformat(),
            "validation_scope": "Repeated frozen fixed-v2" if group == "fixed-v2" else "First-use exact fixtures in same greeting domain, not broad semantic independence",
            "ledger_scope": "010 fresh-base literal006-only 64-step run: 8 examples / 5506 tokens, not accumulated across runs"}
        save(directory / "run-manifest.json", report)
        summaries[group] = report
    after = protected_hashes()
    changed = sorted(path for path in set(before) | set(after) if before.get(path) != after.get(path))
    changed.extend(relative for relative, pinned in prepared["unseen_snapshot_hashes"].items() if digest(ROOT / relative) != pinned)
    save(RUN / "input-preservation.json", {"unchanged": not changed, "protected_file_count": len(before), "changed_paths": changed})
    if changed:
        raise ValueError("Protected inputs or snapshots changed")
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
