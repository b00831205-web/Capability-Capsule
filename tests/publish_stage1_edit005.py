"""Publish the reviewed edit-005 increment using existing dataset APIs."""

from hashlib import sha256
from pathlib import Path

from capability_capsule.eval.curation import trajectory_fingerprint
from capability_capsule.eval.dataset_integrity import load_teacher_dataset_publication
from capability_capsule.eval.dataset_pipeline import curate_teacher_dataset
from capability_capsule.eval.dataset_publication import publish_teacher_dataset
from capability_capsule.eval.dataset_validation import validate_teacher_dataset
from capability_capsule.eval.fixture_provenance import verify_task_fixture
from capability_capsule.eval.harness_profile import verify_harness_profile
from capability_capsule.eval.jsonl import load_jsonl
from capability_capsule.eval.records import TeacherTrajectory
from capability_capsule.eval.student_target import verify_student_target
from capability_capsule.eval.teacher_collection_plan_publication import load_teacher_collection_plan


ROOT = Path(__file__).resolve().parents[1]
DATASET_ID = "stage1-codex-powershell-edit-005"


def main() -> None:
    raw_path = ROOT / "datasets/teacher" / DATASET_ID / "raw.jsonl"
    raw_bytes = raw_path.read_bytes()
    if sha256(raw_bytes).hexdigest() != "86e537f9ec6d1f0459c98f6efa82f92f0a7ea43fc4e13423d702e6afb2abcda2":
        raise ValueError("Raw artifact differs from the reviewed version")
    plan = load_teacher_collection_plan(ROOT / "plans/teacher" / DATASET_ID).plan
    verify_student_target(plan.student_target, artifact_root=ROOT)
    harness = verify_harness_profile(plan.harness_profile, artifact_root=ROOT)
    raw = load_jsonl(raw_path, TeacherTrajectory)
    validate_teacher_dataset(raw, tasks=[a.task for a in plan.assignments], harness_profile=harness)
    for assignment in plan.assignments:
        verify_task_fixture(assignment.task, ROOT / assignment.authorized_fixture_root)
    output_root = ROOT / "datasets/teacher/published"
    old_fingerprints = {
        trajectory_fingerprint(record)
        for path in output_root.glob("*/train.jsonl")
        if path.parent.name != DATASET_ID
        for record in load_jsonl(path, TeacherTrajectory)
    }
    if old_fingerprints.intersection(trajectory_fingerprint(record) for record in raw):
        raise ValueError("New increment duplicates an existing published trajectory")
    dataset = curate_teacher_dataset(raw, tasks=[a.task for a in plan.assignments])
    if (len(dataset.train), len(dataset.validation), len(dataset.duplicates)) != (12, 0, 0):
        raise ValueError("Unexpected curated partition counts")
    manifest = publish_teacher_dataset(dataset, output_root=output_root, dataset_id=DATASET_ID)
    verified = load_teacher_dataset_publication(output_root / DATASET_ID)
    if verified.dataset != dataset or raw_path.read_bytes() != raw_bytes:
        raise ValueError("Publication differs from reviewed raw or raw changed")
    print(manifest.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
