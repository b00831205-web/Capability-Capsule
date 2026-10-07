"""Immutable independent increment, exact raw bytes and protected old artifacts."""

import json
from hashlib import sha256

import pytest
import publish_stage1_fstring_edit007 as publication
from capability_capsule.eval.dataset_integrity import load_teacher_dataset_publication
from capability_capsule.eval.dataset_publication import publish_teacher_dataset
from capability_capsule.eval.jsonl import load_jsonl
from capability_capsule.eval.records import TeacherTrajectory


def test_fstring007_publication_contains_only_six_raw_records_and_four_files():
    verified = load_teacher_dataset_publication(publication.OUTPUT)
    raw = load_jsonl(publication.collection.RAW, TeacherTrajectory)
    assert verified.dataset.train == raw and len(raw) == 6
    assert verified.dataset.validation == verified.dataset.duplicates == ()
    assert [a.record_count for a in verified.manifest.artifacts] == [6, 0, 0]
    assert (publication.OUTPUT / "train.jsonl").read_bytes() == publication.collection.RAW.read_bytes()
    assert verified.manifest.artifacts[0].sha256 == publication.RAW_SHA256
    assert (publication.OUTPUT / "validation.jsonl").read_bytes() == b""
    assert (publication.OUTPUT / "duplicates.jsonl").read_bytes() == b""
    assert {p.name for p in publication.OUTPUT.iterdir()} == {"train.jsonl", "validation.jsonl", "duplicates.jsonl", "manifest.json"}


def test_fstring007_publication_records_independent_validation_and_preserves_old_inputs():
    evidence = json.loads(publication.EVIDENCE.read_text("utf-8"))
    assert evidence["unchanged"] and evidence["raw_train_byte_identical"]
    assert evidence["protected_file_count"] == len(evidence["protected_input_hashes"])
    current = publication.protected_hashes()
    assert all(current.get(path) == digest for path, digest in evidence["protected_input_hashes"].items())
    assert evidence["manifest_sha256"] == sha256((publication.OUTPUT / "manifest.json").read_bytes()).hexdigest()
    assert len(evidence["independent_validations"]) == 6
    assert all(v["exit_code"] == 0 and "passed" in v["output"] for v in evidence["independent_validations"])
    plan = publication.load_teacher_collection_plan(publication.collection.PLAN_DIR).plan
    assert [v["trajectory_id"] for v in evidence["independent_validations"]] == [a.trajectory_id for a in plan.assignments]
    assert not evidence["known_exact_validation_overlap"]
    for path in (publication.collection.RAW, publication.collection.EVIDENCE / "collection-audit.json",
                 publication.ROOT / "datasets/teacher/published/stage1-codex-powershell-literal-edit-006/manifest.json",
                 publication.ROOT / "runs/training/stage1-codex-qwen35-2b-010-literal006-64step-retry1/checkpoint.json",
                 publication.ROOT / "runs/evaluation/stage1-codex-qwen35-2b-010-literal006-64step-retry1-parameter-diagnostic-v1/diagnostic-summary.json"):
        assert path.relative_to(publication.ROOT).as_posix() in evidence["protected_input_hashes"]
    historical = json.loads((publication.collection.EVIDENCE / "collection-audit.json").read_text("utf-8"))
    assert not historical["published"]  # Historical collection phase stays unchanged.


def test_fstring007_publication_refuses_overwrite():
    verified = load_teacher_dataset_publication(publication.OUTPUT)
    before = {p.name: p.read_bytes() for p in publication.OUTPUT.iterdir()}
    with pytest.raises(FileExistsError):
        publish_teacher_dataset(verified.dataset, output_root=publication.OUTPUT.parent, dataset_id=publication.DATASET_ID)
    with pytest.raises(FileExistsError):
        publication.main()
    assert {p.name: p.read_bytes() for p in publication.OUTPUT.iterdir()} == before
