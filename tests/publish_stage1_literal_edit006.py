"""Review and publish only the eight-record literal-edit-006 increment."""

import json
import os
import subprocess
import sys
from hashlib import sha256

import collect_stage1_literal_edit006 as collection
import test_stage1_literal_edit006_collection as checks
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
EVIDENCE = ROOT / "tests/literal006-publication-audit.json"
RAW_SHA256 = "7864fdc8366f5be433257790734b6cdcc094f7457172fc5a4bbc652ada4785ee"


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
        raise ValueError("Raw differs from the independently reviewed collection")
    plan = load_teacher_collection_plan(collection.PLAN_DIR).plan
    verify_student_target(plan.student_target, artifact_root=ROOT)
    # Existing checks cover schema/envelopes, precise patch, structure and known validation overlap.
    checks.test_literal006_complete_pinned_train_assignments_and_raw()
    checks.test_literal006_observed_commands_are_literal_only_and_results_are_real()
    checks.test_literal006_preserves_python_structure_tests_and_pinned_snapshots()
    checks.test_literal006_preserves_existing_artifacts_and_records_collection_scope()
    checks.test_literal006_has_no_known_exact_overlap_with_validation()
    validations = []
    for assignment, variant in zip(plan.assignments, collection.VARIANTS, strict=True):
        verify_task_fixture(assignment.task, ROOT / assignment.authorized_fixture_root)
        workspace = collection.WORKSPACES / variant.slug
        validated = subprocess.run(
            [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "test_greeting.py"],
            cwd=workspace, env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
            capture_output=True, text=True, timeout=60, check=False)
        if validated.returncode != 0:
            raise ValueError(f"Independent validation failed: {variant.slug}: {validated.stdout}{validated.stderr}")
        validations.append({"task_id": assignment.task.task_id, "source_revision": assignment.task.fixture_revision,
                            "exit_code": validated.returncode, "output": validated.stdout + validated.stderr})
        print(f"reviewed {variant.slug}: independent pytest passed", flush=True)
    raw = load_jsonl(collection.RAW, TeacherTrajectory)
    dataset = curate_teacher_dataset(raw, tasks=[a.task for a in plan.assignments])
    if (len(dataset.train), len(dataset.validation), len(dataset.duplicates)) != (8, 0, 0):
        raise ValueError("Unexpected curated partition counts")
    return dataset, validations


def main():
    if OUTPUT.exists() or EVIDENCE.exists():
        raise FileExistsError("Refusing to overwrite a publication or its review evidence")
    before = protected_hashes()
    dataset, validations = review()
    if before != protected_hashes():
        raise ValueError("Protected inputs changed during review")
    manifest = publish_teacher_dataset(dataset, output_root=OUTPUT.parent, dataset_id=DATASET_ID)
    verified = load_teacher_dataset_publication(OUTPUT)
    if verified.dataset != dataset or (OUTPUT / "train.jsonl").read_bytes() != collection.RAW.read_bytes():
        raise ValueError("Publication differs from reviewed raw")
    if before != protected_hashes():
        raise ValueError("Protected inputs changed during publication")
    evidence = {"dataset_id": DATASET_ID, "raw_sha256": RAW_SHA256,
                "manifest_sha256": sha256((OUTPUT / "manifest.json").read_bytes()).hexdigest(),
                "train_count": 8, "validation_count": 0, "duplicate_count": 0,
                "raw_train_byte_identical": True, "protected_file_count": len(before),
                "protected_input_hashes": before, "unchanged": True, "independent_validations": validations,
                "scope": "Raw increment curation/publication only; not SFT, training or Student evaluation",
                "leakage_scope": "No known exact task/revision/group overlap; not proof of semantic independence"}
    collection.save(EVIDENCE, evidence)
    print(manifest.model_dump_json(indent=2), flush=True)
    print(f"Published: 8 train, 0 validation, 0 duplicates; {len(before)} inputs byte-unchanged", flush=True)


if __name__ == "__main__":
    main()
