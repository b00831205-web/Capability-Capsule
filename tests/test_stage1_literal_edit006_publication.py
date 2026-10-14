"""Verify immutable publication and preservation of its reviewed inputs."""

import json
from hashlib import sha256

import pytest
import publish_stage1_literal_edit006 as publication
from capability_capsule.eval.dataset_integrity import load_teacher_dataset_publication
from capability_capsule.eval.dataset_publication import publish_teacher_dataset
from capability_capsule.eval.jsonl import load_jsonl
from capability_capsule.eval.records import TeacherTrajectory


def test_literal006_publication_matches_raw_and_empty_partitions():
    verified = load_teacher_dataset_publication(publication.OUTPUT)
    raw = load_jsonl(publication.collection.RAW, TeacherTrajectory)
    assert verified.dataset.train == raw
    assert len(raw) == 8
    assert verified.dataset.validation == verified.dataset.duplicates == ()
    assert [a.record_count for a in verified.manifest.artifacts] == [8, 0, 0]
    assert (publication.OUTPUT / "train.jsonl").read_bytes() == publication.collection.RAW.read_bytes()
    assert verified.manifest.artifacts[0].sha256 == publication.RAW_SHA256
    assert (publication.OUTPUT / "validation.jsonl").read_bytes() == b""
    assert (publication.OUTPUT / "duplicates.jsonl").read_bytes() == b""
    assert {p.name for p in publication.OUTPUT.iterdir()} == {"train.jsonl", "validation.jsonl", "duplicates.jsonl", "manifest.json"}


def test_literal006_publication_preserves_inputs_and_records_independent_validators():
    evidence = json.loads(publication.EVIDENCE.read_text("utf-8"))
    assert evidence["unchanged"] and evidence["raw_train_byte_identical"]
    assert evidence["protected_file_count"] == len(evidence["protected_input_hashes"])
    current = publication.protected_hashes()
    assert all(current.get(path) == digest for path, digest in evidence["protected_input_hashes"].items())
    assert evidence["manifest_sha256"] == sha256((publication.OUTPUT / "manifest.json").read_bytes()).hexdigest()
    assert len(evidence["independent_validations"]) == 8
    assert all(v["exit_code"] == 0 and "passed" in v["output"] for v in evidence["independent_validations"])
    for relative in (publication.collection.RAW.relative_to(publication.ROOT).as_posix(),
                     "runs/training/stage1-codex-qwen35-2b-008-edit005-64step/checkpoint.json",
                     "runs/evaluation/stage1-codex-qwen35-2b-008-edit005-64step-v2-diagnostic/evaluation-summary.json"):
        assert relative in evidence["protected_input_hashes"]


def test_literal006_publication_refuses_overwrite():
    verified = load_teacher_dataset_publication(publication.OUTPUT)
    before = {path.name: path.read_bytes() for path in publication.OUTPUT.iterdir()}
    with pytest.raises(FileExistsError):
        publish_teacher_dataset(verified.dataset, output_root=publication.OUTPUT.parent, dataset_id=publication.DATASET_ID)
    with pytest.raises(FileExistsError):
        publication.main()
    assert {path.name: path.read_bytes() for path in publication.OUTPUT.iterdir()} == before
