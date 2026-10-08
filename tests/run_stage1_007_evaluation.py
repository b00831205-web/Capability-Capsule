"""Pin and independently evaluate run 007 without changing existing artifacts."""

import json
import shutil
import sys
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path

from capability_capsule.eval.coding_checkpoint import (
    CodingEvaluationCase, CodingEvaluationSuite, ConstrainedPowerShellExecutor,
    TransformersPeftTurnGenerator, copy_verified_evaluation_fixture,
    evaluate_coding_case, load_coding_evaluation_suite, run_coding_validators,
)
from capability_capsule.eval.fixture_provenance import inspect_fixture
from capability_capsule.eval.jsonl import append_jsonl
from capability_capsule.eval.learning_curve_ledger import append_learning_curve_point
from capability_capsule.eval.learning_rate import LearningCurvePoint
from run_stage1_006_v3_command_policy import TOOLS

ROOT = Path(__file__).resolve().parents[1]
RUN_ID = "stage1-codex-qwen35-2b-007-edit005-32step"
RUN = ROOT / "runs/evaluation" / RUN_ID
TRAINING = ROOT / "runs/training" / RUN_ID
BASELINE = ROOT / "runs/evaluation/stage1-codex-qwen35-2b-006-turns-v2-v3-command-policy-raw-order"
SUITE = ROOT / "plans/evaluation/stage1-codex-validation-v3-command-policy/evaluation-suite.json"
EXECUTOR = ROOT / "src/capability_capsule/eval/coding_checkpoint.py"

# Validation-only definitions, fixed before checkpoint inference. They do not
# produce Teacher trajectories or enter a published training dataset.
HELDOUT = (
    (
        "mapping-template",
        '"""Preserve the keyed template and its formatting mechanism."""\n\n'
        'TEMPLATES = {"default": "Hello, {person}"}\n\n'
        'def greeting(name: str) -> str:\n'
        '    return TEMPLATES["default"].format(person=name)\n',
        'Hello, {person}', 'Safe travels, {person}!',
        (("Zora", "Safe travels, Zora!"), ("Basil", "Safe travels, Basil!")),
        'Change only greeting.py so greeting("Zora") returns "Safe travels, Zora!" '
        'and greeting("Basil") returns "Safe travels, Basil!". Preserve the module '
        'description and keyed-template structure. Validate without changing test_greeting.py.',
    ),
    (
        "tuple-join",
        '# Keep the tuple assembly and empty-name behavior unchanged.\n'
        'def greeting(name: str) -> str:\n'
        '    if not name:\n'
        '        return "No guest"\n'
        '    pieces = ("Hello", ", ", name, ".")\n'
        '    return "".join(pieces)\n',
        '("Hello", ", ", name, ".")', '("Welcome aboard", ", ", name, ".")',
        (("Kei", "Welcome aboard, Kei."), ("Oren", "Welcome aboard, Oren."), ("", "No guest")),
        'Edit greeting.py only: greeting("Kei") must return "Welcome aboard, Kei." '
        'and greeting("Oren") must return "Welcome aboard, Oren." Keep the tuple assembly, '
        'comment and empty-name result "No guest" unchanged. Run the supplied tests.',
    ),
)


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def protected_hashes():
    paths = set()
    for directory in (ROOT / "datasets/teacher", ROOT / "artifacts/sft", TRAINING):
        paths.update(p for p in directory.rglob("*") if p.is_file())
    for case in load_coding_evaluation_suite(SUITE).cases:
        paths.update(p for p in (ROOT / case.fixture_root).rglob("*")
                     if p.is_file() and "__pycache__" not in p.parts and ".pytest_cache" not in p.parts)
    paths.add(SUITE)
    paths.add(EXECUTOR)
    return {p.relative_to(ROOT).as_posix(): digest(p) for p in sorted(paths)}


def published_training_provenance():
    revisions, task_ids, sources = set(), set(), set()
    for path in (ROOT / "datasets/teacher/published").glob("*/train.jsonl"):
        for line in path.read_text("utf-8").splitlines():
            record = json.loads(line)
            if record.get("source_revision"):
                revisions.add(record["source_revision"])
            task_ids.add(record["task_id"])
            for message in record["messages"]:
                if message["role"] == "tool":
                    try:
                        sources.add(json.loads(message.get("content", "")).get("output", ""))
                    except (ValueError, TypeError):
                        pass
    return revisions, task_ids, sources


def prepare():
    before = protected_hashes()
    if RUN.exists():
        if (RUN / "prepared.json").exists() or (RUN / "heldout-v1").exists():
            raise FileExistsError(RUN)
        if json.loads((RUN / "protected-input-hashes.json").read_text("utf-8")) != before:
            raise ValueError("Incomplete preparation has changed inputs")
    else:
        RUN.mkdir(parents=True)
        save(RUN / "protected-input-hashes.json", before)
    fixed = load_coding_evaluation_suite(SUITE)
    if fixed != load_coding_evaluation_suite(BASELINE / "evaluation-suite.json"):
        raise ValueError("Fixed suite differs from the matched baseline")
    groups = {"fixed-v3": fixed}
    fixed_dir = RUN / "fixed-v3"
    fixed_dir.mkdir(exist_ok=True)
    if (fixed_dir / "evaluation-suite.json").exists():
        if (fixed_dir / "evaluation-suite.json").read_bytes() != SUITE.read_bytes():
            raise ValueError("Incomplete fixed suite differs")
    else:
        shutil.copyfile(SUITE, fixed_dir / "evaluation-suite.json")
    sources = {}
    for case in fixed.cases:
        source = ROOT / case.fixture_root
        if inspect_fixture(case.task.fixture_id, source).revision != case.task.fixture_revision:
            source = ROOT / "runs/evaluation/stage1-codex-qwen35-2b-006-turns-v2-v2/workspaces" / case.case_id
        relative = source.relative_to(ROOT).as_posix()
        copy_case = CodingEvaluationCase.model_validate({**case.model_dump(mode="json"), "fixture_root": relative})
        destination = fixed_dir / "workspaces" / case.case_id
        if destination.exists():
            if inspect_fixture(case.task.fixture_id, destination).revision != case.task.fixture_revision:
                raise ValueError("Incomplete fixed workspace differs")
        else:
            copy_verified_evaluation_fixture(copy_case, artifact_root=ROOT, destination=destination)
        sources[case.case_id] = relative

    cases = []
    training_revisions, training_task_ids, training_sources = published_training_provenance()
    held_dir = RUN / "heldout-v1"
    held_dir.mkdir()
    for slug, source, old, new, checks, task_text in HELDOUT:
        case_id = f"validation-edit007-{slug}-001"
        snapshot = held_dir / "fixtures" / slug
        snapshot.mkdir(parents=True)
        (snapshot / "greeting.py").write_bytes(source.encode("utf-8"))
        test = "from greeting import greeting\n\ndef test_greeting():\n" + "".join(
            f"    assert greeting({name!r}) == {expected!r}\n" for name, expected in checks)
        (snapshot / "test_greeting.py").write_bytes(test.encode("utf-8"))
        revision = inspect_fixture(case_id, snapshot).revision
        if revision in training_revisions or case_id in training_task_ids or source in training_sources:
            raise ValueError("Heldout fixture overlaps published training")
        validators = [
            {"validator_id": "pytest:test_greeting.py", "kind": "pytest", "target": "test_greeting.py"},
            {"validator_id": "exact-content:greeting.py", "kind": "exact_text", "target": "greeting.py",
             "expected": source.replace(old, new)},
            {"validator_id": "unchanged:test_greeting.py", "kind": "exact_text", "target": "test_greeting.py", "expected": test},
        ]
        task = {**fixed.cases[0].task.model_dump(mode="json"), "task_id": case_id, "fixture_id": case_id,
                "fixture_family_id": f"heldout-edit007-{slug}", "fixture_revision": revision,
                "defect_family": f"heldout-{slug}", "task": task_text, "time_limit": "PT3M",
                "knowledge_distance": 0.4, "validation_ids": [v["validator_id"] for v in validators]}
        case = CodingEvaluationCase.model_validate({"case_id": case_id,
                "fixture_root": snapshot.relative_to(ROOT).as_posix(), "task": task, "validators": validators})
        cases.append(case)
        copy_verified_evaluation_fixture(case, artifact_root=ROOT, destination=held_dir / "workspaces" / case_id)
        # Prove the frozen validator rejects the original and accepts the intended
        # constrained edit in a separate preflight copy, never the model workspace.
        preflight = held_dir / "preflight" / case_id
        copy_verified_evaluation_fixture(case, artifact_root=ROOT, destination=preflight)
        original = run_coding_validators(case, preflight)
        command = ("$text = Get-Content -LiteralPath greeting.py -Raw; $updated = $text.Replace('"
                   + old + "', '" + new + "'); Set-Content -LiteralPath greeting.py -Value $updated")
        execution = ConstrainedPowerShellExecutor().execute(command, preflight)
        corrected = run_coding_validators(case, preflight)
        if all(v.passed for v in original) or not execution.authorized or execution.exit_code or not all(v.passed for v in corrected):
            raise ValueError("Heldout preflight failed")
        save(held_dir / f"preflight-{slug}.json", {"original": [v.model_dump(mode="json") for v in original],
             "corrected": [v.model_dump(mode="json") for v in corrected], "execution": execution.model_dump(mode="json")})
    heldout = CodingEvaluationSuite.model_validate({**fixed.model_dump(mode="json"),
        "evaluation_suite_id": "stage1-codex-validation-edit007-heldout-v1", "cases": cases})
    groups["heldout-v1"] = heldout
    save(held_dir / "evaluation-suite.json", heldout.model_dump(mode="json"))
    save(RUN / "prepared.json", {"prepared_at": datetime.now(UTC).isoformat(),
         "checkpoint_sha256": digest(TRAINING / "checkpoint.json"), "fixture_sources": sources,
         "executor_sha256": digest(EXECUTOR), "tools_sha256": sha256(json.dumps(TOOLS, sort_keys=True).encode()).hexdigest(),
         "groups": {name: {"suite_sha256": digest(RUN / name / "evaluation-suite.json"),
                    "case_digest": suite.identity().evaluation_suite_digest} for name, suite in groups.items()},
         "heldout_overlap_audit": {"published_training_revision_count": len(training_revisions),
              "exact_revision_task_source_overlap": False,
              "limitation": "New exact fixtures and outputs in the same greeting domain, not a new repository or semantic independence proof."}})
    print("Pinned 3 fixed cases and 2 heldout cases; both heldout preflights passed.", flush=True)


def audit_prepared():
    revisions, task_ids, sources = published_training_provenance()
    suite = load_coding_evaluation_suite(RUN / "heldout-v1/evaluation-suite.json")
    for case in suite.cases:
        source = (ROOT / case.fixture_root / "greeting.py").read_text("utf-8")
        if case.task.fixture_revision in revisions or case.task.task_id in task_ids or source in sources:
            raise ValueError("Heldout fixture overlaps published training")
    audit = {"published_training_revision_count": len(revisions),
             "published_training_task_count": len(task_ids), "exact_revision_task_source_overlap": False,
             "limitation": "Same greeting domain; no semantic independence or new-repository claim."}
    save(RUN / "heldout-overlap-audit.json", audit)
    # Correct preparation metadata before any checkpoint inference, without
    # changing a pinned fixture, task, validator, suite, or preparation timestamp.
    prepared = json.loads((RUN / "prepared.json").read_text("utf-8"))
    prepared["heldout_overlap_audit"] = audit
    save(RUN / "prepared.json", prepared)


class ProgressGenerator:
    def __init__(self, generator):
        self.generator = generator
        self.turn = 0

    def generate(self, messages, tools):
        self.turn += 1
        print(f"Generating turn {self.turn}...", flush=True)
        result = self.generator.generate(messages, tools)
        print(f"Generated {result.output_tokens} tokens.", flush=True)
        return result


def evaluate():
    audit_prepared()
    prepared = json.loads((RUN / "prepared.json").read_text("utf-8"))
    if prepared["checkpoint_sha256"] != digest(TRAINING / "checkpoint.json") or prepared["executor_sha256"] != digest(EXECUTOR):
        raise ValueError("Prepared checkpoint or executor changed")
    if protected_hashes() != json.loads((RUN / "protected-input-hashes.json").read_text("utf-8")):
        raise ValueError("Protected inputs changed after preparation")
    for name, pins in prepared["groups"].items():
        directory = RUN / name
        if (directory / "case-results.jsonl").exists():
            raise FileExistsError("Refusing to overwrite or repeat an existing evaluation")
        if digest(directory / "evaluation-suite.json") != pins["suite_sha256"]:
            raise ValueError("Pinned suite changed")
        suite = load_coding_evaluation_suite(directory / "evaluation-suite.json")
        for case in suite.cases:
            if inspect_fixture(case.task.fixture_id, directory / "workspaces" / case.case_id).revision != case.task.fixture_revision:
                raise ValueError("Prepared workspace changed")
    generator = ProgressGenerator(TransformersPeftTurnGenerator(TRAINING / "checkpoint.json", max_new_tokens=256))
    training = json.loads((TRAINING / "training-run.json").read_text("utf-8"))
    checkpoint = json.loads((TRAINING / "checkpoint.json").read_text("utf-8"))
    summaries = {}
    for name in prepared["groups"]:
        directory = RUN / name
        suite = load_coding_evaluation_suite(directory / "evaluation-suite.json")
        reports_dir = directory / "case-reports"
        reports_dir.mkdir()
        results = []
        for position, case in enumerate(suite.cases, start=1):
            print(f"Starting {name}: {case.case_id}", flush=True)
            outcome = evaluate_coding_case(case, workspace=directory / "workspaces" / case.case_id,
                generator=generator, executor=ConstrainedPowerShellExecutor(), system_prompt=suite.system_prompt,
                tools=TOOLS, experiment_id=f"{RUN_ID}:{name}", run_id=f"{RUN_ID}:{name}:{case.case_id}",
                model_id=checkpoint["output_capsule_id"], hardware_id=training["hardware_id"],
                cycle_position=position, max_tool_rounds=4)
            append_jsonl(directory / "case-results.jsonl", outcome.result)
            save(reports_dir / f"{case.case_id}.json", {"result": outcome.result.model_dump(mode="json"),
                 "completions": list(outcome.completions), "tool_results": [r.model_dump(mode="json") for r in outcome.tool_results],
                 "validators": [v.model_dump(mode="json") for v in outcome.validators]})
            results.append(outcome.result)
            print(f"{case.case_id}: success={outcome.result.success}, invalid={outcome.result.invalid_tool_call_count}", flush=True)
        count = sum(r.success for r in results)
        point = LearningCurvePoint(checkpoint_id=f"{RUN_ID}-step-{checkpoint['step']}", training_run_id=RUN_ID,
            base_model_id=checkpoint["base_model_id"], capability_id=suite.capability_id,
            evaluation_suite_id=suite.evaluation_suite_id, evaluation_suite_digest=suite.identity().evaluation_suite_digest,
            task_family_id=("greeting-single-file-change-plus-unseen-powershell-salutation" if name == "fixed-v3"
                            else "greeting-heldout-mapping-and-tuple"), evaluation_split="validation",
            cumulative_trajectory_count=training["dataset_stats"]["train"]["trajectory_count"],
            cumulative_token_count=training["dataset_stats"]["train"]["exact_token_count"],
            tokenizer_id=training["dataset_stats"]["train"]["tokenizer_id"], success_rate=count / len(results))
        append_learning_curve_point(directory / "learning-curve.jsonl", point)
        manifest = {**prepared["groups"][name], "checkpoint": (TRAINING / "checkpoint.json").relative_to(ROOT).as_posix(),
            "checkpoint_sha256": prepared["checkpoint_sha256"], "executor_sha256": prepared["executor_sha256"],
            "tools_sha256": prepared["tools_sha256"], "system_prompt_sha256": sha256(suite.system_prompt.encode()).hexdigest(),
            "baseline_run": BASELINE.relative_to(ROOT).as_posix() if name == "fixed-v3" else None,
            "max_new_tokens": 256, "max_tool_rounds": 4, "do_sample": False,
            "case_count": len(results), "success_count": count,
            "ledger_scope": "Fresh-base edit005-only run; counts are this run's data, not accumulated across runs 006/007.",
            "prepared_at": prepared["prepared_at"], "completed_at": datetime.now(UTC).isoformat()}
        save(directory / "run-manifest.json", manifest)
        summaries[name] = manifest
    after = protected_hashes()
    before = json.loads((RUN / "protected-input-hashes.json").read_text("utf-8"))
    save(RUN / "input-preservation.json", {"unchanged": before == after, "protected_file_count": len(before),
         "changed_paths": sorted(p for p in set(before) | set(after) if before.get(p) != after.get(p))})
    if before != after:
        raise ValueError("Protected input bytes changed")
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
