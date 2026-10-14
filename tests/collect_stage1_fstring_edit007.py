"""Six independently authored train-only inline f-string literal edits."""

import ast
import json
import shutil
import subprocess
import sys
from datetime import timedelta
from hashlib import sha256
from pathlib import Path

import collect_stage1_edit005 as shared
from capability_capsule.eval.coding_checkpoint import load_coding_evaluation_suite
from capability_capsule.eval.curation import trajectory_fingerprint
from capability_capsule.eval.dataset_validation import validate_teacher_dataset
from capability_capsule.eval.fixture_provenance import inspect_fixture
from capability_capsule.eval.harness_profile import verify_harness_profile
from capability_capsule.eval.jsonl import load_jsonl
from capability_capsule.eval.records import DatasetSplit, TeacherTrajectory
from capability_capsule.eval.student_target import verify_student_target
from capability_capsule.eval.teacher_collection import append_teacher_trajectory
from capability_capsule.eval.records import TrajectoryMessage
from capability_capsule.eval.tasks import ChangeOperation, ExpectedChange, TaskCategory, TaskDifficulty, TaskSpec
from capability_capsule.eval.teacher_collection_plan import build_teacher_collection_plan
from capability_capsule.eval.teacher_collection_plan_publication import PublishedTeacherCollectionPlan, load_teacher_collection_plan

ROOT = shared.ROOT
PLAN_ID = "stage1-codex-powershell-fstring-edit-007"
PLAN_DIR = ROOT / "plans/teacher" / PLAN_ID
ARTIFACTS = ROOT / "artifacts/teacher-fixtures" / PLAN_ID
SOURCES = ARTIFACTS / "sources"
WORKSPACES = ARTIFACTS / "workspaces"
EVIDENCE = ARTIFACTS / "review"
RAW_RELATIVE = Path("datasets/teacher") / PLAN_ID / "raw.jsonl"
RAW = ROOT / RAW_RELATIVE
Variant = shared.Variant
VARIANTS = (
    Variant("inline-docstring", '"""Keep the public greeting function."""\n\ndef greeting(name: str) -> str:\n    return f"Hello there, {name}."\n',
        "Hello there, {name}.", "Keep building, {name}!", "Wren", "Keep building, Wren!",
        'Change greeting.py only so callers receive "Keep building, <name>!", including Wren. Preserve the function and module description; validate without editing the tests.'),
    Variant("single-quoted", "def greeting(name: str) -> str:\n    return f'Hey visitor: {name}'\n",
        "Hey visitor: {name}", "Build boldly: {name}.", "Bram", "Build boldly: Bram.",
        'Make greeting.py return "Build boldly: <name>." for any caller, such as Bram. Keep the existing string construction and validate the change.'),
    Variant("local-message", 'def greeting(name: str) -> str:\n    message = f"Hello partner / {name}"\n    return message\n',
        "Hello partner / {name}", "Stay inventive / {name}!", "Alba", "Stay inventive / Alba!",
        'Update the message in greeting.py to say "Stay inventive / <name>!" for callers such as Alba. Preserve the intermediate variable and return statement. Run the tests.'),
    Variant("upper-expression", 'def greeting(name: str) -> str:\n    return f"Good morning :: {name.upper()}"\n',
        "Good morning :: {name.upper()}", "Make progress :: {name.upper()}.", "sia", "Make progress :: SIA.",
        'In greeting.py, return "Make progress :: <uppercase name>." while keeping uppercase conversion. Check multiple callers without altering test_greeting.py.'),
    Variant("trim-expression", 'def greeting(name: str) -> str:\n    return f"Hi colleague - {name.strip()}"\n',
        "Hi colleague - {name.strip()}", "Create steadily - {name.strip()}!", " Luca ", "Create steadily - Luca!",
        'Return "Create steadily - <trimmed name>!" from greeting.py. Keep whitespace stripping and validate the existing behavior for different inputs.'),
    Variant("empty-branch", '# Keep the attendee fallback.\ndef greeting(name: str) -> str:\n    if not name:\n        return "No attendee"\n    return f"Hello contributor, {name}"\n',
        "Hello contributor, {name}", "Next idea, {name}.", "Dion", "Next idea, Dion.",
        'Named callers should receive "Next idea, <name>." from greeting.py. Preserve the empty-name response and comment, and run the supplied tests.', "No attendee"),
)
CHECKS = {
    "inline-docstring": (("Wren", "Keep building, Wren!"), ("Rhea", "Keep building, Rhea!"), ("", "Keep building, !")),
    "single-quoted": (("Bram", "Build boldly: Bram."), ("Hugo", "Build boldly: Hugo."), ("", "Build boldly: .")),
    "local-message": (("Alba", "Stay inventive / Alba!"), ("Omar", "Stay inventive / Omar!"), ("", "Stay inventive / !")),
    "upper-expression": (("sia", "Make progress :: SIA."), ("Tobin", "Make progress :: TOBIN."), ("", "Make progress :: .")),
    "trim-expression": ((" Luca ", "Create steadily - Luca!"), ("  Elsa  ", "Create steadily - Elsa!"), ("", "Create steadily - !"), ("   ", "Create steadily - !")),
    "empty-branch": (("Dion", "Next idea, Dion."), ("Petra", "Next idea, Petra."), ("", "No attendee")),
}
EDIT_PARTS = {
    "single-quoted": (("Hey visitor", "Build boldly"), ("{name}", "{name}.")),
    "local-message": (("Hello partner", "Stay inventive"), ("{name}", "{name}!")),
    "upper-expression": (("Good morning", "Make progress"), ("{name.upper()}", "{name.upper()}.")),
}


def save(path, value):
    shared.write_once(path, (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode())


def protected_hashes():
    hashes = {}
    for relative in ("src", "plans/teacher", "plans/student", "plans/harness", "plans/evaluation",
                     "datasets/teacher", "artifacts/sft", "artifacts/teacher-fixtures",
                     "artifacts/evaluation-fixtures", "runs/training", "runs/evaluation",
                     "fixtures/smoke-pilot/validation-greeting-change"):
        for path in (ROOT / relative).rglob("*"):
            if (not path.is_file() or PLAN_DIR in path.parents or ARTIFACTS in path.parents or RAW.parent in path.parents
                    or any(p in {"__pycache__", ".pytest_cache"} for p in path.parts)):
                continue
            hashes[path.relative_to(ROOT).as_posix()] = sha256(path.read_bytes()).hexdigest()
    return hashes


def normalized_tree(code):
    tree = ast.parse(code)
    for node in ast.walk(tree):
        if isinstance(node, ast.JoinedStr):
            # Prefix/suffix text may introduce another literal segment; placeholders cannot change.
            node.values = [value for value in node.values
                           if not (isinstance(value, ast.Constant) and isinstance(value.value, str))]
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            node.value = "<string>"
    return ast.dump(tree)


def source_for(variant):
    source = SOURCES / variant.slug
    # Author new fixtures; do not copy any validation source or locked-test fixture.
    source.mkdir(parents=True, exist_ok=True)
    shared.write_once(source / "greeting.py", variant.source.encode())
    tests = "from greeting import greeting\n\ndef test_multiple_names_and_empty_input():\n" + "".join(
        f"    assert greeting({name!r}) == {expected!r}\n" for name, expected in CHECKS[variant.slug])
    shared.write_once(source / "test_greeting.py", tests.encode())
    if variant.source.count(variant.old) != 1 or "return" in variant.old + variant.new:
        raise ValueError("Must replace exactly one inline string literal")
    corrected = variant.source.replace(variant.old, variant.new)
    if normalized_tree(variant.source) != normalized_tree(corrected):
        raise ValueError("Python structure changed")
    if not any(isinstance(node, ast.JoinedStr) for node in ast.walk(ast.parse(variant.source))):
        raise ValueError("Inline f-string required")
    evidence = EVIDENCE / "baseline-checks" / f"{variant.slug}.json"
    if evidence.exists():
        recorded = json.loads(evidence.read_text("utf-8"))
        if recorded["exit_code"] != 1 or "failed" not in recorded["output"]:
            raise ValueError("Invalid recorded baseline")
        return source
    baseline = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider"],
        cwd=source, capture_output=True, text=True, timeout=60)
    if baseline.returncode != 1 or "failed" not in baseline.stdout:
        raise ValueError("Original must fail target behavior")
    save(evidence, {"exit_code": baseline.returncode,
        "output": baseline.stdout + baseline.stderr})
    return source


def audit(plan):
    raw = load_jsonl(RAW, TeacherTrajectory)
    harness = verify_harness_profile(plan.harness_profile, artifact_root=ROOT)
    validate_teacher_dataset(raw, tasks=[a.task for a in plan.assignments], harness_profile=harness)
    previous = []
    for path in (ROOT / "datasets/teacher").glob("*/raw.jsonl"):
        if path != RAW:
            previous.extend(load_jsonl(path, TeacherTrajectory))
    for path in (ROOT / "datasets/teacher/published").glob("*/train.jsonl"):
        if path.parent.name != PLAN_ID:
            previous.extend(load_jsonl(path, TeacherTrajectory))
    from teacher_replay_history import distinct_history
    previous = distinct_history(raw, previous)
    if len(raw) != len(VARIANTS) or len({trajectory_fingerprint(t) for t in raw}) != len(VARIANTS):
        raise ValueError("Count or internal duplicate mismatch")
    if {trajectory_fingerprint(t) for t in raw} & {trajectory_fingerprint(t) for t in previous}:
        raise ValueError("Historical exact duplicate")
    if {t.task_id for t in raw} & {t.task_id for t in previous} or {t.source_revision for t in raw} & {t.source_revision for t in previous}:
        raise ValueError("Historical task/revision reuse")
    validation_revisions, validation_texts, validation_sources, targets = set(), set(), set(), set()
    suites = set((ROOT / "plans/evaluation").rglob("evaluation-suite.json")) | set((ROOT / "runs/evaluation").rglob("evaluation-suite.json"))
    for path in suites:
        for case in load_coding_evaluation_suite(path).cases:
            if case.task.split is not DatasetSplit.VALIDATION:
                raise ValueError("Unexpected non-validation suite")
            validation_revisions.add(case.task.fixture_revision)
            validation_texts.add(case.task.task)
            initial = ROOT / case.fixture_root / "greeting.py"
            if initial.exists():
                validation_sources.add(initial.read_text("utf-8"))
            for validator in case.validators:
                if validator.kind == "python_call" and isinstance(validator.expected, str):
                    targets.add(validator.expected)
                elif validator.kind == "exact_text" and validator.target == "greeting.py":
                    for node in ast.walk(ast.parse(validator.expected)):
                        if (isinstance(node, ast.Constant) and isinstance(node.value, str)
                                and sum(character.isalpha() for character in node.value) >= 5):
                            targets.add(node.value)
    if {t.source_revision for t in raw} & validation_revisions or {t.task for t in raw} & validation_texts:
        raise ValueError("Validation revision/task overlap")
    initial_sources = {v.source for v in VARIANTS}
    if initial_sources & validation_sources:
        raise ValueError("Validation source overlap")
    texts = [v.new for v in VARIANTS] + [expected for checks in CHECKS.values() for _, expected in checks]
    if any(target in text for target in targets for text in texts):
        raise ValueError("Known validation target reuse")
    total_calls = 0
    for assignment, trajectory, variant in zip(plan.assignments, raw, VARIANTS, strict=True):
        source = ROOT / assignment.authorized_fixture_root
        if inspect_fixture(assignment.task.fixture_id, source).revision != trajectory.source_revision:
            raise ValueError("Pinned snapshot changed")
        workspace = WORKSPACES / variant.slug
        if (workspace / "greeting.py").read_text("utf-8") != variant.source.replace(variant.old, variant.new):
            raise ValueError("Unexpected edit")
        if (source / "test_greeting.py").read_bytes() != (workspace / "test_greeting.py").read_bytes():
            raise ValueError("Validator changed")
        total_calls += sum(len(message.tool_calls) for message in trajectory.messages)
    return {"train_count": len(raw), "validation_count": 0, "observed_tool_calls": total_calls,
        "independent_validators_passed": len(raw), "baseline_target_tests_failed": len(raw),
        "known_exact_duplicates": 0, "validation_exact_overlap": False,
        "validation_revision_count": len(validation_revisions), "raw_sha256": sha256(RAW.read_bytes()).hexdigest(),
        "source_revisions": {t.task_id: t.source_revision for t in raw},
        "coverage": "Six inline f-string literal edits; multiple names and empty input, preserving placeholders/expressions/branches",
        "execution_backend": "ConstrainedPowerShellExecutor: real file reads/replacements/pytest, not full native PowerShell",
        "teacher_model": "gpt-5", "teacher_skill_version": "0.2.0",
        "student_model": plan.student_target.recommendation.model_id,
        "hardware_profile_id": plan.student_target.hardware.profile_id,
        "hardware_verification_status": plan.student_target.hardware.verification_status,
        "hardware_purposes": list(plan.student_target.hardware.purposes),
        "recommendation_id": plan.student_target.recommendation.recommendation_id,
        "recommendation_evidence_status": plan.student_target.recommendation.evidence_status,
        "leakage_scope": "Known exact historical/validation overlap checked; same greeting domain, not semantic independence",
        "recovery_traces": "None unless genuinely observed; original failing tests are preflight evidence, not simulated recovery",
        "published": False, "sft_exported": False, "training_started": False}


def collect(assignment, variant):
    from capability_capsule.eval.coding_checkpoint import ConstrainedPowerShellExecutor
    source = ROOT / assignment.authorized_fixture_root
    if inspect_fixture(assignment.task.fixture_id, source).revision != assignment.task.fixture_revision:
        raise ValueError("Pinned snapshot changed")
    workspace = WORKSPACES / variant.slug
    if workspace.exists():
        if inspect_fixture(assignment.task.fixture_id, workspace).revision != assignment.task.fixture_revision:
            raise ValueError("Refusing unexplained workspace changes")
    else:
        # Teacher assignments are train; use a verified plain copy instead of validation-only cases.
        shutil.copytree(source, workspace)
        if inspect_fixture(assignment.task.fixture_id, workspace).revision != assignment.task.fixture_revision:
            raise ValueError("Workspace copy mismatch")
    executor = ConstrainedPowerShellExecutor()
    messages = [TrajectoryMessage(role="user", content=assignment.task.task),
                TrajectoryMessage(role="assistant", content="Inspect greeting.py before editing.")]
    read_command = "Get-Content -LiteralPath greeting.py -Raw"
    result = executor.execute(read_command, workspace)
    shared.record_call(messages, read_command, result)
    if result.exit_code or result.output != variant.source:
        raise ValueError("Read did not match snapshot")
    steps = EDIT_PARTS.get(variant.slug, ((variant.old, variant.new),))
    for old, new in steps:
        messages.append(TrajectoryMessage(role="assistant", content="Update the greeting text while preserving the dynamic name expression."))
        command = f"$text = Get-Content -LiteralPath greeting.py -Raw; $updated = $text.Replace('{old}', '{new}'); Set-Content -LiteralPath greeting.py -Value $updated"
        result = executor.execute(command, workspace)
        shared.record_call(messages, command, result)
        if not result.authorized or result.exit_code:
            raise ValueError(f"Observed edit failed: {result.output}")
    if (workspace / "greeting.py").read_text("utf-8") != variant.source.replace(variant.old, variant.new):
        raise ValueError("Local edits did not produce intended source")
    if (workspace / "test_greeting.py").read_bytes() != (source / "test_greeting.py").read_bytes():
        raise ValueError("Test file changed")
    messages.append(TrajectoryMessage(role="assistant", content="Run the supplied tests for multiple names and empty input."))
    result = executor.execute("pytest -q", workspace)
    shared.record_call(messages, "pytest -q", result)
    if result.exit_code or "passed" not in result.output:
        raise ValueError("Observed pytest failed")
    independent = subprocess.run([sys.executable, "-m", "pytest", "-q", "test_greeting.py", "-p", "no:cacheprovider"],
        cwd=workspace, capture_output=True, text=True, timeout=60)
    if independent.returncode:
        raise ValueError("Independent pytest failed")
    save(EVIDENCE / "independent-checks" / f"{variant.slug}.json", {"exit_code": independent.returncode,
        "output": independent.stdout + independent.stderr, "test_sha256": sha256((source / "test_greeting.py").read_bytes()).hexdigest()})
    messages.append(TrajectoryMessage(role="assistant", content="Edited only greeting.py; all supplied input cases passed."))
    trajectory = TeacherTrajectory(trajectory_id=assignment.trajectory_id, task_id=assignment.task.task_id,
        split=DatasetSplit.TRAIN, teacher_model=assignment.teacher_model, teacher_skill_version=assignment.teacher_skill_version,
        task=assignment.task.task, messages=tuple(messages), source_revision=assignment.task.fixture_revision,
        tags=("cmd-only-contract", "guarded-replace", "inline-fstring-literal", "dynamic-name-preserved"))
    append_teacher_trajectory(assignment, trajectory, artifact_root=ROOT)
    print(f"{variant.slug}: appended; {len(steps) + 2} observed calls; independent pytest passed", flush=True)


def main():
    if sys.argv[1:] == ["--audit-only"]:
        plan = load_teacher_collection_plan(PLAN_DIR).plan
        before = json.loads((EVIDENCE / "protected-input-hashes.json").read_text("utf-8"))
        report = audit(plan)
        current = protected_hashes()
        if any(current.get(path) != pinned for path, pinned in before.items()):
            raise ValueError("Protected artifacts changed")
        report["protected_file_count"] = len(before)
        save(EVIDENCE / "input-preservation.json", {"unchanged": True, "file_count": len(before)})
        save(EVIDENCE / "collection-audit.json", report)
        print(json.dumps(report, indent=2), flush=True)
        return
    if sys.argv[1:] == ["--resume-collection"]:
        plan = load_teacher_collection_plan(PLAN_DIR).plan
        before = json.loads((EVIDENCE / "protected-input-hashes.json").read_text("utf-8"))
        current = protected_hashes()
        if any(current.get(path) != pinned for path, pinned in before.items()):
            raise ValueError("Protected inputs changed")
        existing = {trajectory.trajectory_id for trajectory in load_jsonl(RAW, TeacherTrajectory)}
        for assignment, variant in zip(plan.assignments, VARIANTS, strict=True):
            if assignment.trajectory_id not in existing:
                collect(assignment, variant)
        report = audit(plan)
        current = protected_hashes()
        if any(current.get(path) != pinned for path, pinned in before.items()):
            raise ValueError("Protected inputs changed")
        report["protected_file_count"] = len(before)
        save(EVIDENCE / "input-preservation.json", {"unchanged": True, "file_count": len(before)})
        save(EVIDENCE / "collection-audit.json", report)
        print(json.dumps(report, indent=2), flush=True)
        return
    resume = sys.argv[1:] == ["--resume-preparation"]
    if sys.argv[1:] not in ([], ["--resume-preparation"]):
        raise ValueError("Use --audit-only or --resume-preparation")
    if PLAN_DIR.exists() or RAW.parent.exists() or (ARTIFACTS.exists() and not resume):
        raise FileExistsError("Refusing to overwrite or resume collection artifacts")
    old = load_teacher_collection_plan(ROOT / "plans/teacher/stage1-codex-powershell-literal-edit-006").plan
    verify_student_target(old.student_target, artifact_root=ROOT)
    verify_harness_profile(old.harness_profile, artifact_root=ROOT)
    before = protected_hashes()
    if resume:
        if before != json.loads((EVIDENCE / "protected-input-hashes.json").read_text("utf-8")):
            raise ValueError("Existing inputs changed during preparation")
    else:
        save(EVIDENCE / "protected-input-hashes.json", before)
    tasks, roots, ids = [], {}, {}
    for index, variant in enumerate(VARIANTS, 1):
        source = source_for(variant)
        fixture_id = f"train-fstring-edit-007-{variant.slug}-v1"
        task_id = f"train-fstring-edit-007-{index:02d}-{variant.slug}"
        tasks.append(TaskSpec(task_id=task_id, fixture_id=fixture_id, fixture_family_id="train-fstring-edit-007",
            fixture_revision=inspect_fixture(fixture_id, source).revision, defect_family=variant.slug,
            split=DatasetSplit.TRAIN, category=TaskCategory.SINGLE_FILE_CHANGE, difficulty=TaskDifficulty.EASY,
            task=variant.task, knowledge_distance=0.2, logical_arrival=timedelta(0), time_limit=timedelta(minutes=3),
            allowed_tools=("exec_command",), expected_changes=(ExpectedChange(path="greeting.py", operation=ChangeOperation.MODIFY),),
            validation_ids=("pytest:test_greeting.py", "exact-inline-fstring-edit", "unchanged-test-file")))
        roots[fixture_id] = source.relative_to(ROOT).as_posix()
        ids[task_id] = f"train-fstring-edit-007-{index:02d}-trajectory-001"
    plan = build_teacher_collection_plan(plan_id=PLAN_ID, tasks=tasks, teacher_model="gpt-5",
        teacher_skill_version="0.2.0", fixture_roots=roots, destination_jsonl=RAW_RELATIVE, trajectory_ids=ids,
        student_target=old.student_target, harness_profile=old.harness_profile)
    data = (plan.model_dump_json(indent=2) + "\n").encode()
    shared.write_once(PLAN_DIR / "collection-plan.json", data)
    manifest = PublishedTeacherCollectionPlan(schema_version=plan.schema_version, plan_id=PLAN_ID,
        byte_count=len(data), sha256=sha256(data).hexdigest(), plan=plan)
    shared.write_once(PLAN_DIR / "manifest.json", (manifest.model_dump_json(indent=2) + "\n").encode())
    load_teacher_collection_plan(PLAN_DIR)
    WORKSPACES.mkdir(parents=True, exist_ok=True)
    for assignment, variant in zip(plan.assignments, VARIANTS, strict=True):
        collect(assignment, variant)
    report = audit(plan)
    if before != protected_hashes():
        raise ValueError("Existing protected artifacts changed")
    report["protected_file_count"] = len(before)
    save(EVIDENCE / "input-preservation.json", {"unchanged": True, "file_count": len(before)})
    save(EVIDENCE / "collection-audit.json", report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
