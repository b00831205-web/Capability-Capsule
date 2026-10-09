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
