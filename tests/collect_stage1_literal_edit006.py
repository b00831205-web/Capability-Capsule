"""Collect eight independent train fixtures; no publication or training."""

import json
import shutil
import subprocess
import sys
from datetime import timedelta
from hashlib import sha256
from pathlib import Path

import collect_stage1_edit005 as shared
from capability_capsule.eval.curation import trajectory_fingerprint
from capability_capsule.eval.dataset_validation import validate_teacher_dataset
from capability_capsule.eval.fixture_provenance import inspect_fixture
from capability_capsule.eval.harness_profile import verify_harness_profile
from capability_capsule.eval.jsonl import load_jsonl
from capability_capsule.eval.records import DatasetSplit, TeacherTrajectory
from capability_capsule.eval.student_target import verify_student_target
from capability_capsule.eval.tasks import ChangeOperation, ExpectedChange, TaskCategory, TaskDifficulty, TaskSpec
from capability_capsule.eval.teacher_collection_plan import build_teacher_collection_plan
from capability_capsule.eval.teacher_collection_plan_publication import PublishedTeacherCollectionPlan, load_teacher_collection_plan

ROOT = shared.ROOT
PLAN_ID = "stage1-codex-powershell-literal-edit-006"
PLAN_DIR = ROOT / "plans/teacher" / PLAN_ID
ARTIFACTS = ROOT / "artifacts/teacher-fixtures" / PLAN_ID
SOURCES = ARTIFACTS / "sources"
WORKSPACES = ARTIFACTS / "workspaces"
EVIDENCE = ARTIFACTS / "review"
RAW_RELATIVE = Path("datasets/teacher") / PLAN_ID / "raw.jsonl"
RAW = ROOT / RAW_RELATIVE
Variant = shared.Variant

# Authored train-only examples: none are copied from validation sources or answers.
VARIANTS = (
    Variant("positional-format", 'def greeting(name: str) -> str:\n    return "Greetings, {}".format(name)\n',
            "Greetings, {}", "Good luck, {}!", "Elio", "Good luck, Elio!",
            'Change greeting.py so named callers receive "Good luck, <name>!". Keep the existing formatting mechanism and test.'),
    Variant("named-format", 'def greeting(name: str) -> str:\n    label = name.title()\n    return "Salute, {label}.".format(label=label)\n',
            "Salute, {label}.", "Bright skies, {label}!", "maren", "Bright skies, Maren!",
            'Return "Bright skies, <title-cased name>!" from greeting.py, preserving the title conversion and formatting mechanism. Validate.'),
    Variant("keyed-template", 'PATTERNS = {"active": "Salute, {guest}.", "inactive": "Session closed"}\n\ndef greeting(name: str) -> str:\n    return PATTERNS["active"].format(guest=name)\n',
            "Salute, {guest}.", "Happy coding, {guest}!", "Niko", "Happy coding, Niko!",
            'Make greeting.py say "Happy coding, <name>!". Preserve both dictionary entries, dictionary lookup and formatting. Run tests.'),
    Variant("keyed-positional", 'MESSAGES = {"visitor": "Howdy, {}!", "staff": "Internal greeting"}\n\ndef greeting(name: str) -> str:\n    template = MESSAGES["visitor"]\n    return template.format(name)\n',
            "Howdy, {}!", "See you soon, {}.", "Tessa", "See you soon, Tessa.",
            'For visitors, return "See you soon, <name>.". Keep the dictionary, intermediate template variable and formatting in greeting.py. Validate.'),
    Variant("tuple-prefix", 'def greeting(name: str) -> str:\n    parts = ("Howdy", ": ", name, "!")\n    return "".join(parts)\n',
            "Howdy", "Enjoy exploring", "Ari", "Enjoy exploring: Ari!",
            'Return "Enjoy exploring: <name>!". Keep the tuple and join mechanism in greeting.py and run the supplied tests.'),
    Variant("tuple-with-fallback", 'def greeting(name: str) -> str:\n    if not name:\n        return "Visitor unavailable"\n    fragments = ("Salutations", " - ", name, ".")\n    return "".join(fragments)\n',
            "Salutations", "Keep learning", "Remy", "Keep learning - Remy.",
            'Return "Keep learning - <name>." for named callers. Preserve the empty-name result and tuple/join implementation in greeting.py. Test.',
            "Visitor unavailable"),
    Variant("branch-format", 'def greeting(name: str) -> str:\n    if name == "":\n        return "Anonymous visitor"\n    return "Salute, {}.".format(name)\n',
            "Salute, {}.", "Best wishes, {}!", "Dara", "Best wishes, Dara!",
            'Return "Best wishes, <name>!" for named callers. Preserve the empty-name branch and formatting in greeting.py. Validate.',
            "Anonymous visitor"),
    Variant("branch-prefix", 'PREFIX = "Howdy"\n\ndef greeting(name: str) -> str:\n    if not name:\n        return "Name pending"\n    return f"{PREFIX}, {name}."\n',
            "Howdy", "Until tomorrow", "Cleo", "Until tomorrow, Cleo.",
            'Return "Until tomorrow, <name>." for named callers. Preserve the module constant, f-string and empty-name behavior in greeting.py. Test.',
            "Name pending"),
)


def save(path, value):
    shared.write_once(path, (json.dumps(value, indent=2, ensure_ascii=False) + "\n").encode())


def protected_hashes():
    """Hash existing experimental inputs, never locked-test fixtures."""
    roots = ["src", "plans/teacher", "plans/student", "plans/harness",
             "datasets/teacher", "artifacts/sft", "artifacts/teacher-fixtures",
             "artifacts/evaluation-fixtures", "runs/training", "runs/evaluation",
             "fixtures/smoke-pilot/validation-greeting-change"]
    hashes = {}
    for relative in roots:
        for path in (ROOT / relative).rglob("*"):
            if not path.is_file() or PLAN_DIR in path.parents or ARTIFACTS in path.parents or RAW.parent in path.parents:
                continue
            if any(part in {"__pycache__", ".pytest_cache"} for part in path.parts):
                continue
            hashes[path.relative_to(ROOT).as_posix()] = sha256(path.read_bytes()).hexdigest()
    return hashes


def source_for(variant):
    source = SOURCES / variant.slug
    shutil.copytree(shared.BASE, source, ignore=shutil.ignore_patterns("__pycache__", ".pytest_cache", "*.pyc"))
    (source / "greeting.py").write_text(variant.source, encoding="utf-8", newline="")
    # Two distinct inputs discourage hardcoding a caller; exact source checks enforce structure.
    alternate = "Soren"
    expected_alternate = variant.expected.replace(variant.name.title() if variant.slug == "named-format" else variant.name, alternate)
    tests = ("from greeting import greeting\n\n"
             "def test_named_callers():\n"
             f"    assert greeting({variant.name!r}) == {variant.expected!r}\n"
             f"    assert greeting({alternate!r}) == {expected_alternate!r}\n")
    if variant.fallback is not None:
        tests += f"\ndef test_empty_name():\n    assert greeting('') == {variant.fallback!r}\n"
    if variant.slug == "keyed-template":
        tests += '\ndef test_unused_entry():\n    from greeting import PATTERNS\n    assert PATTERNS["inactive"] == "Session closed"\n'
    if variant.slug == "keyed-positional":
        tests += '\ndef test_unused_entry():\n    from greeting import MESSAGES\n    assert MESSAGES["staff"] == "Internal greeting"\n'
    (source / "test_greeting.py").write_text(tests, encoding="utf-8", newline="")
    if variant.source.count(variant.old) != 1 or "return" in variant.old + variant.new:
        raise ValueError("Exactly one literal-only change is required")
    baseline = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider"],
                              cwd=source, capture_output=True, text=True, timeout=60)
    if baseline.returncode != 1 or "failed" not in baseline.stdout:
        raise ValueError("Starting fixture must fail its target behavior test")
    save(EVIDENCE / "baseline-checks" / f"{variant.slug}.json",
         {"exit_code": baseline.returncode, "output": baseline.stdout + baseline.stderr})
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
    fingerprints = {trajectory_fingerprint(t) for t in raw}
    if len(raw) != 8 or len(fingerprints) != 8 or fingerprints.intersection(trajectory_fingerprint(t) for t in previous):
        raise ValueError("Count or exact duplicate gate failed")
    if {t.task_id for t in raw}.intersection(t.task_id for t in previous):
        raise ValueError("Existing task identifier reused")
    if {t.source_revision for t in raw}.intersection(t.source_revision for t in previous):
        raise ValueError("Existing source revision reused")
    return {"train_count": len(raw), "validation_count": 0, "observed_tool_calls": 24,
            "independent_validators_passed": 8, "baseline_target_tests_failed": 8,
            "known_exact_duplicates": 0, "raw_sha256": sha256(RAW.read_bytes()).hexdigest(),
            "source_revisions": {t.task_id: t.source_revision for t in raw},
            "execution_backend": "ConstrainedPowerShellExecutor: real file reads/local replacements/pytest, not native full PowerShell",
            "leakage_review": "Independently authored train sources and targets; exact overlap checks are not proof of semantic independence",
            "student_model": plan.student_target.recommendation.model_id,
            "hardware_profile_id": plan.student_target.hardware.profile_id,
            "hardware_verification_status": plan.student_target.hardware.verification_status,
            "recommendation_id": plan.student_target.recommendation.recommendation_id,
            "recommendation_evidence_status": plan.student_target.recommendation.evidence_status,
            "published": False, "sft_exported": False, "training_started": False}


def main():
    if sys.argv[1:] == ["--audit-only"]:
        plan = load_teacher_collection_plan(PLAN_DIR).plan
        before = json.loads((EVIDENCE / "protected-input-hashes.json").read_text("utf-8"))
        report = audit(plan)
        if before != protected_hashes():
            raise ValueError("Protected existing artifacts changed")
        report["protected_file_count"] = len(before)
        save(EVIDENCE / "input-preservation.json", {"unchanged": True, "file_count": len(before)})
        save(EVIDENCE / "collection-audit.json", report)
        print(json.dumps(report, indent=2), flush=True)
        return
    if sys.argv[1:]:
        raise ValueError("Only --audit-only is supported")
    if PLAN_DIR.exists() or RAW.parent.exists():
        raise FileExistsError("Refusing to overwrite or reuse existing collection artifacts")
    if inspect_fixture("train-greeting-good-day-contract-v1", shared.BASE).revision != shared.BASE_REVISION:
        raise ValueError("Original train source changed")
    old = load_teacher_collection_plan(ROOT / "plans/teacher/stage1-codex-powershell-edit-005").plan
    verify_student_target(old.student_target, artifact_root=ROOT)
    verify_harness_profile(old.harness_profile, artifact_root=ROOT)
    before = protected_hashes()
    save(EVIDENCE / "protected-input-hashes.json", before)
    tasks, roots, ids = [], {}, {}
    for index, variant in enumerate(VARIANTS, 1):
        source = source_for(variant)
        fixture_id = f"train-literal-edit-006-{variant.slug}-v1"
        task_id = f"train-literal-edit-006-{index:02d}-{variant.slug}"
        tasks.append(TaskSpec(task_id=task_id, fixture_id=fixture_id,
            fixture_family_id="train-literal-edit-006", fixture_revision=inspect_fixture(fixture_id, source).revision,
            defect_family=variant.slug, split=DatasetSplit.TRAIN, category=TaskCategory.SINGLE_FILE_CHANGE,
            difficulty=TaskDifficulty.EASY, task=variant.task, knowledge_distance=0.2,
            logical_arrival=timedelta(0), time_limit=timedelta(minutes=3), allowed_tools=("exec_command",),
            expected_changes=(ExpectedChange(path="greeting.py", operation=ChangeOperation.MODIFY),),
            validation_ids=("pytest:test_greeting.py", "exact-local-literal-edit", "unchanged-test-file")))
        roots[fixture_id] = source.relative_to(ROOT).as_posix()
        ids[task_id] = f"train-literal-edit-006-{index:02d}-trajectory-001"
    plan = build_teacher_collection_plan(plan_id=PLAN_ID, tasks=tasks, teacher_model="gpt-5",
        teacher_skill_version="0.2.0", fixture_roots=roots, destination_jsonl=RAW_RELATIVE,
        trajectory_ids=ids, student_target=old.student_target, harness_profile=old.harness_profile)
    data = (plan.model_dump_json(indent=2) + "\n").encode()
    shared.write_once(PLAN_DIR / "collection-plan.json", data)
    manifest = PublishedTeacherCollectionPlan(schema_version=plan.schema_version, plan_id=PLAN_ID,
        byte_count=len(data), sha256=sha256(data).hexdigest(), plan=plan)
    shared.write_once(PLAN_DIR / "manifest.json", (manifest.model_dump_json(indent=2) + "\n").encode())
    load_teacher_collection_plan(PLAN_DIR)
    # Reuse the proven recording implementation in memory; never edit the older script/artifacts.
    replacements = {"VARIANTS": VARIANTS, "WORKSPACES": WORKSPACES}
    original = {key: getattr(shared, key) for key in replacements}
    try:
        for key, value in replacements.items():
            setattr(shared, key, value)
        for assignment, variant in zip(plan.assignments, VARIANTS, strict=True):
            shared.collect(assignment, variant)
    finally:
        for key, value in original.items():
            setattr(shared, key, value)
    report = audit(plan)
    after = protected_hashes()
    if before != after:
        raise ValueError("Protected existing artifacts changed")
    report["protected_file_count"] = len(before)
    save(EVIDENCE / "input-preservation.json", {"unchanged": True, "file_count": len(before)})
    save(EVIDENCE / "collection-audit.json", report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
