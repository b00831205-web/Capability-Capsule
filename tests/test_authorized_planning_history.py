"""Later doc edits need exact recorded before/after hashes; artifacts stay immutable."""

import json
from hashlib import sha256

import authorized_planning_history as history


def test_only_recorded_planning_edit_can_differ(tmp_path, monkeypatch):
    monkeypatch.setattr(history, "ROOT", tmp_path)
    document = "MVP_ROADMAP.md"
    (tmp_path / document).write_text("new", encoding="utf-8")
    before = sha256(b"old").hexdigest()
    after = sha256(b"new").hexdigest()
    update = tmp_path / "update.json"
    update.write_text(json.dumps({"explicitly_authorized": True,
        "before": {document: before}, "after": {document: after}}), encoding="utf-8")
    assert history.input_matches(document, before, after, update_path=update)
    assert not history.input_matches("runs/training/adapter.bin", before, after, update_path=update)
    assert not history.input_matches(document, "unrelated-old-hash", after, update_path=update)
    (tmp_path / document).write_text("unrecorded follow-up edit", encoding="utf-8")
    assert not history.input_matches(document, before, after, update_path=update)


def test_missing_update_is_not_permission(tmp_path):
    assert not history.input_matches("MVP_ROADMAP.md", "old", "new", update_path=tmp_path / "missing.json")
    assert history.input_matches("runs/training/adapter.bin", "same", "same", update_path=tmp_path / "missing.json")


def test_fixed_train_diagnostic_receipt_extends_documents_only(tmp_path, monkeypatch):
    monkeypatch.setattr(history, "ROOT", tmp_path)
    name = "MVP_ROADMAP.md"
    (tmp_path / name).write_text("diagnosed", encoding="utf-8")
    first, middle, final = [sha256(value).hexdigest() for value in (b"initial", b"repaired", b"diagnosed")]
    initial = tmp_path / "first.json"
    initial.write_text(json.dumps({"explicitly_authorized":True,
        "before":{name:first}, "after":{name:middle}}), encoding="utf-8")
    receipt = tmp_path / "runs/evaluation/stage1-codex-qwen35-2b-012-mix006007-56step-train-fit-diagnostic-evalfix-v2/planning-update.json"
    receipt.parent.mkdir(parents=True)
    value = {"explicitly_authorized":True, "before":{name:middle}, "after":{name:final}}
    receipt.write_text(json.dumps(value), encoding="utf-8")
    assert history.input_matches(name, first, final, update_path=initial)
    assert not history.input_matches("datasets/teacher/raw.jsonl", first, final, update_path=initial)
    receipt.write_text(json.dumps({**value,"explicitly_authorized":False}), encoding="utf-8")
    assert not history.input_matches(name, first, final, update_path=initial)
    receipt.write_text(json.dumps({**value,"before":{name:"unrelated"}}), encoding="utf-8")
    assert not history.input_matches(name, first, final, update_path=initial)


def test_eval_repair_receipt_only_extends_authorized_document_hashes(tmp_path, monkeypatch):
    monkeypatch.setattr(history, "ROOT", tmp_path)
    name = "MVP_ROADMAP.md"
    (tmp_path / name).write_text("repaired", encoding="utf-8")
    first, middle, final = [sha256(value).hexdigest() for value in (b"initial", b"diagnosed", b"repaired")]
    initial = tmp_path / "first.json"
    initial.write_text(json.dumps({"explicitly_authorized": True,
        "before": {name: first}, "after": {name: middle}}), encoding="utf-8")
    receipt = tmp_path / "tests/eval-v2-planning-update.json"
    receipt.parent.mkdir()
    value = {"explicitly_authorized": True, "before": {name: middle}, "after": {name: final}}
    receipt.write_text(json.dumps(value), encoding="utf-8")
    assert history.input_matches(name, first, final, update_path=initial)
    assert not history.input_matches("src/capability_capsule/eval/coding_checkpoint.py", first, final, update_path=initial)
    receipt.write_text(json.dumps({**value, "before": {name: "unrelated"}}), encoding="utf-8")
    assert not history.input_matches(name, first, final, update_path=initial)
    receipt.write_text(json.dumps({**value, "explicitly_authorized": False}), encoding="utf-8")
    assert not history.input_matches(name, first, final, update_path=initial)


def test_recovery_update_requires_unbroken_hash_chain(tmp_path, monkeypatch):
    monkeypatch.setattr(history, "ROOT", tmp_path)
    document = "MVP_ROADMAP.md"
    (tmp_path / document).write_text("final", encoding="utf-8")
    first = sha256(b"first").hexdigest()
    middle = sha256(b"middle").hexdigest()
    final = sha256(b"final").hexdigest()
    update = tmp_path / "planning-update.json"
    update.write_text(json.dumps({"explicitly_authorized": True,
        "before": {document: first}, "after": {document: middle}}), encoding="utf-8")
    receipt = tmp_path / "recovery-20261009/planning-update.json"
    receipt.parent.mkdir()
    receipt.write_text(json.dumps({"explicitly_authorized": True,
        "before": {document: middle}, "after": {document: final}}), encoding="utf-8")
    assert history.input_matches(document, first, final, update_path=update)
    receipt.write_text(json.dumps({"explicitly_authorized": True,
        "before": {document: "broken"}, "after": {document: final}}), encoding="utf-8")
    assert not history.input_matches(document, first, final, update_path=update)


def test_reload_receipt_requires_explicit_authorization_and_matching_previous_hash(tmp_path, monkeypatch):
    monkeypatch.setattr(history, "ROOT", tmp_path)
    document = "MVP_ROADMAP.md"
    (tmp_path / document).write_text("reloaded", encoding="utf-8")
    initial = sha256(b"initial").hexdigest()
    middle = sha256(b"trained").hexdigest()
    final = sha256(b"reloaded").hexdigest()
    update = tmp_path / "first.json"
    update.write_text(json.dumps({"explicitly_authorized": True,
        "before": {document: initial}, "after": {document: middle}}), encoding="utf-8")
    receipt = tmp_path / "runs/training/stage1-codex-qwen35-2b-012-mix006007-56step/reload-planning-update-20261010.json"
    receipt.parent.mkdir(parents=True)
    value = {"explicitly_authorized": True, "before": {document: middle}, "after": {document: final}}
    receipt.write_text(json.dumps(value), encoding="utf-8")
    assert history.input_matches(document, initial, final, update_path=update)
    receipt.write_text(json.dumps({**value, "explicitly_authorized": False}), encoding="utf-8")
    assert not history.input_matches(document, initial, final, update_path=update)
    receipt.write_text(json.dumps({**value, "before": {document: "unrelated"}}), encoding="utf-8")
    assert not history.input_matches(document, initial, final, update_path=update)
