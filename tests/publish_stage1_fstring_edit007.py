"""Independently review and publish only the six-record f-string increment."""

import json
import os
import subprocess
import sys
from hashlib import sha256

import collect_stage1_fstring_edit007 as collection
import test_stage1_fstring_edit007_collection as checks
from capability_capsule.eval.dataset_integrity import load_teacher_dataset_publication
from capability_capsule.eval.dataset_pipeline import curate_teacher_dataset
from capability_capsule.eval.dataset_publication import publish_teacher_dataset
from capability_capsule.eval.fixture_provenance import verify_task_fixture
from capability_capsule.eval.jsonl import load_jsonl
from capability_capsule.eval.records import TeacherTrajectory
from capability_capsule.eval.student_target import verify_student_target
from capability_capsule.eval.teacher_collection_plan_publication import load_teacher_collection_plan

ROOT = collection.ROOT
DATASET_ID = collection.PLAN_ID
OUTPUT = ROOT / "datasets/teacher/published" / DATASET_ID
EVIDENCE = ROOT / "tests/fstring007-publication-audit.json"
RAW_SHA256 = "fdfb8418c2372572574673a9a85ca5e69939d4751c236d6041c54da275e2a457"


def protected_hashes():
    hashes = collection.protected_hashes()
    prefix = OUTPUT.relative_to(ROOT).as_posix() + "/"
    hashes = {path: digest for path, digest in hashes.items() if not path.startswith(prefix)}
    for directory in (collection.PLAN_DIR, collection.ARTIFACTS, collection.RAW.parent):
        for path in directory.rglob("*"):
            if path.is_file() and not any(part in {"__pycache__", ".pytest_cache"} for part in path.parts):
                hashes[path.relative_to(ROOT).as_posix()] = sha256(path.read_bytes()).hexdigest()
    return hashes


def review():
    if sha256(collection.RAW.read_bytes()).hexdigest() != RAW_SHA256:
        raise ValueError("Raw differs from reviewed collection")
    plan = load_teacher_collection_plan(collection.PLAN_DIR).plan
    verify_student_target(plan.student_target, artifact_root=ROOT)
    checks.test_fstring007_six_pinned_train_assignments_complete_and_raw_unique()
    checks.test_fstring007_raw_contains_real_bounded_calls_and_no_invented_recovery()
    checks.test_fstring007_preserves_dynamic_expressions_and_multi_input_tests()
    checks.test_fstring007_preserves_existing_artifacts_and_stops_at_raw_review()
    checks.test_fstring007_normalization_allows_literal_suffix_but_rejects_parameter_loss()
    validations = []
    for assignment, variant in zip(plan.assignments, collection.VARIANTS, strict=True):
        verify_task_fixture(assignment.task, ROOT / assignment.authorized_fixture_root)
        workspace = collection.WORKSPACES / variant.slug
        result = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "test_greeting.py"],
            cwd=workspace, env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
            capture_output=True, text=True, timeout=60, check=False)
        if result.returncode:
            raise ValueError(f"Independent validation failed: {variant.slug}: {result.stdout}{result.stderr}")
        validations.append({"trajectory_id": assignment.trajectory_id, "task_id": assignment.task.task_id,
            "source_revision": assignment.task.fixture_revision, "split": "train",
            "exit_code": result.returncode, "output": result.stdout + result.stderr})
        print(f"reviewed {variant.slug}: independent multi-input pytest passed", flush=True)
    raw = load_jsonl(collection.RAW, TeacherTrajectory)
    dataset = curate_teacher_dataset(raw, tasks=[a.task for a in plan.assignments])
    if (len(dataset.train), len(dataset.validation), len(dataset.duplicates)) != (6, 0, 0):
        raise ValueError("Unexpected partition counts")
    return dataset, validations


def main():
    if OUTPUT.exists() or EVIDENCE.exists():
        raise FileExistsError("Refusing to overwrite publication or evidence")
    before = protected_hashes()
    dataset, validations = review()
    if before != protected_hashes():
        raise ValueError("Protected inputs changed during review")
    manifest = publish_teacher_dataset(dataset, output_root=OUTPUT.parent, dataset_id=DATASET_ID)
    verified = load_teacher_dataset_publication(OUTPUT)
    if verified.dataset != dataset or (OUTPUT / "train.jsonl").read_bytes() != collection.RAW.read_bytes():
        raise ValueError("Published records differ from raw")
    if before != protected_hashes():
        raise ValueError("Protected inputs changed during publication")
    evidence = {"dataset_id": DATASET_ID, "raw_sha256": RAW_SHA256,
        "manifest_sha256": sha256((OUTPUT / "manifest.json").read_bytes()).hexdigest(),
        "train_count": 6, "validation_count": 0, "duplicate_count": 0,
        "raw_train_byte_identical": True, "protected_file_count": len(before),
        "protected_input_hashes": before, "unchanged": True, "independent_validations": validations,
        "known_exact_validation_overlap": False, "teacher_skill_version": "0.2.0",
        "hardware_profile_id": verified.dataset.train and load_teacher_collection_plan(collection.PLAN_DIR).plan.student_target.hardware.profile_id,
        "scope": "Independent six-record increment curation/publication only; not SFT, training or Student evaluation",
        "leakage_scope": "Known exact task/revision/source/meaningful-target overlap reviewed, not semantic independence",
        "historical_collection_evidence_preserved": True}
    collection.save(EVIDENCE, evidence)
    print(manifest.model_dump_json(indent=2), flush=True)
    print(f"Published 6 train / 0 validation / 0 duplicates; {len(before)} protected files byte-unchanged", flush=True)


if __name__ == "__main__":
    main()
