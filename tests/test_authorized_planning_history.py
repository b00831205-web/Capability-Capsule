"""Later doc edits need exact recorded before/after hashes; artifacts stay immutable."""

import json
from hashlib import sha256

import authorized_planning_history as history


def test_precision_memory_waiver_receipt_cannot_authorize_data_changes(tmp_path, monkeypatch):
    monkeypatch.setattr(history,"ROOT",tmp_path)
    name = "MVP_ROADMAP.md"
    (tmp_path/name).write_text("waived-precision",encoding="utf-8")
    first,middle,final = [sha256(x).hexdigest() for x in (b"initial",b"blocked",b"waived-precision")]
    initial = tmp_path/"initial.json"
    initial.write_text(json.dumps({"explicitly_authorized":True,"before":{name:first},"after":{name:middle}}),encoding="utf-8")
    receipt = tmp_path/"runs/evaluation/stage1-codex-qwen35-2b-012-first-edit-precision-memory-override-v1/planning-update.json"
    receipt.parent.mkdir(parents=True)
    value = {"explicitly_authorized":True,"before":{name:middle},"after":{name:final}}
    receipt.write_text(json.dumps(value),encoding="utf-8")
    assert history.input_matches(name,first,final,update_path=initial)
    assert not history.input_matches("datasets/teacher/train.jsonl",first,final,update_path=initial)
    receipt.write_text(json.dumps({**value,"before":{name:"unrelated"}}),encoding="utf-8")
    assert not history.input_matches(name,first,final,update_path=initial)


def test_precision_diagnostic_receipt_only_extends_planning_documents(tmp_path, monkeypatch):
    monkeypatch.setattr(history,"ROOT",tmp_path)
    name = "MVP_ROADMAP.md"
    (tmp_path/name).write_text("precision",encoding="utf-8")
    first,middle,final = [sha256(x).hexdigest() for x in (b"initial",b"forward",b"precision")]
    initial = tmp_path/"initial.json"
    initial.write_text(json.dumps({"explicitly_authorized":True,"before":{name:first},"after":{name:middle}}),encoding="utf-8")
    receipt = tmp_path/"runs/evaluation/stage1-codex-qwen35-2b-012-first-edit-precision-v1/planning-update.json"
    receipt.parent.mkdir(parents=True)
    value = {"explicitly_authorized":True,"before":{name:middle},"after":{name:final}}
    receipt.write_text(json.dumps(value),encoding="utf-8")
    assert history.input_matches(name,first,final,update_path=initial)
    assert not history.input_matches("runs/training/adapter.safetensors",first,final,update_path=initial)
    receipt.write_text(json.dumps({**value,"explicitly_authorized":False}),encoding="utf-8")
    assert not history.input_matches(name,first,final,update_path=initial)


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


def test_memory_waiver_doc_receipt_does_not_authorize_data_or_system_changes(tmp_path, monkeypatch):
    monkeypatch.setattr(history,"ROOT",tmp_path)
    name = "MVP_ROADMAP.md"
    (tmp_path / name).write_text("waived",encoding="utf-8")
    first,middle,final = [sha256(x).hexdigest() for x in (b"initial",b"blocked",b"waived")]
    initial = tmp_path / "initial.json"
    initial.write_text(json.dumps({"explicitly_authorized":True,"before":{name:first},"after":{name:middle}}),encoding="utf-8")
    receipt = tmp_path / "runs/evaluation/stage1-codex-mix006007-teacher-forced-adapters-011-012-memory-override-v1/planning-update.json"
    receipt.parent.mkdir(parents=True)
    value = {"explicitly_authorized":True,"before":{name:middle},"after":{name:final}}
    receipt.write_text(json.dumps(value),encoding="utf-8")
    assert history.input_matches(name,first,final,update_path=initial)
    assert not history.input_matches("datasets/teacher/raw.jsonl",first,final,update_path=initial)
    assert not history.input_matches("system/pagefile.json",first,final,update_path=initial)
    receipt.write_text(json.dumps({**value,"before":{name:"unrelated"}}),encoding="utf-8")
    assert not history.input_matches(name,first,final,update_path=initial)


def test_adapter_only_receipt_extends_documents_not_reused_base_scores(tmp_path, monkeypatch):
    monkeypatch.setattr(history,"ROOT",tmp_path)
    name = "MVP_ROADMAP.md"
    (tmp_path / name).write_text("adapters",encoding="utf-8")
    first,middle,final = [sha256(x).hexdigest() for x in (b"initial",b"base",b"adapters")]
    initial = tmp_path / "initial.json"
    initial.write_text(json.dumps({"explicitly_authorized":True,"before":{name:first},"after":{name:middle}}),encoding="utf-8")
    receipt = tmp_path / "runs/evaluation/stage1-codex-mix006007-teacher-forced-adapters-011-012-v1/planning-update.json"
    receipt.parent.mkdir(parents=True)
    value = {"explicitly_authorized":True,"before":{name:middle},"after":{name:final}}
    receipt.write_text(json.dumps(value),encoding="utf-8")
    assert history.input_matches(name,first,final,update_path=initial)
    assert not history.input_matches("runs/evaluation/base/summary.json",first,final,update_path=initial)
    receipt.write_text(json.dumps({**value,"explicitly_authorized":False}),encoding="utf-8")
    assert not history.input_matches(name,first,final,update_path=initial)


def test_second_retry_receipt_does_not_authorize_old_artifact_edits(tmp_path, monkeypatch):
    monkeypatch.setattr(history,"ROOT",tmp_path)
    name = "MVP_ROADMAP.md"
    (tmp_path / name).write_text("renewed",encoding="utf-8")
    first,middle,final = [sha256(x).hexdigest() for x in (b"initial",b"blocked",b"renewed")]
    initial = tmp_path / "initial.json"
    initial.write_text(json.dumps({"explicitly_authorized":True,"before":{name:first},"after":{name:middle}}),encoding="utf-8")
    receipt = tmp_path / "runs/evaluation/stage1-codex-mix006007-teacher-forced-base-011-012-v1-retry2-20261013/planning-update.json"
    receipt.parent.mkdir(parents=True)
    value = {"explicitly_authorized":True,"before":{name:middle},"after":{name:final}}
    receipt.write_text(json.dumps(value),encoding="utf-8")
    assert history.input_matches(name,first,final,update_path=initial)
    assert not history.input_matches("artifacts/sft/train.jsonl",first,final,update_path=initial)
    receipt.write_text(json.dumps({**value,"explicitly_authorized":False}),encoding="utf-8")
    assert not history.input_matches(name,first,final,update_path=initial)


def test_retry_receipt_extends_only_explicit_document_chain(tmp_path, monkeypatch):
    monkeypatch.setattr(history,"ROOT",tmp_path)
    name = "MVP_ROADMAP.md"
    (tmp_path / name).write_text("blocked",encoding="utf-8")
    first,middle,final = [sha256(x).hexdigest() for x in (b"initial",b"interrupted",b"blocked")]
    initial = tmp_path / "initial.json"
    initial.write_text(json.dumps({"explicitly_authorized":True,"before":{name:first},"after":{name:middle}}),encoding="utf-8")
    receipt = tmp_path / "runs/evaluation/stage1-codex-mix006007-teacher-forced-base-011-012-v1-retry-20261013/planning-update.json"
    receipt.parent.mkdir(parents=True)
    value = {"explicitly_authorized":True,"before":{name:middle},"after":{name:final}}
    receipt.write_text(json.dumps(value),encoding="utf-8")
    assert history.input_matches(name,first,final,update_path=initial)
    assert not history.input_matches("runs/evaluation/old/summary.json",first,final,update_path=initial)
    receipt.write_text(json.dumps({**value,"before":{name:"wrong"}}),encoding="utf-8")
    assert not history.input_matches(name,first,final,update_path=initial)


def test_forward_only_receipt_cannot_authorize_artifact_changes(tmp_path, monkeypatch):
    monkeypatch.setattr(history,"ROOT",tmp_path)
    name = "MVP_ROADMAP.md"
    (tmp_path / name).write_text("forward",encoding="utf-8")
    first,middle,final = [sha256(x).hexdigest() for x in (b"initial",b"diagnosed",b"forward")]
    initial = tmp_path / "initial.json"
    initial.write_text(json.dumps({"explicitly_authorized":True,"before":{name:first},"after":{name:middle}}),encoding="utf-8")
    receipt = tmp_path / "runs/evaluation/stage1-codex-mix006007-teacher-forced-base-011-012-v1/planning-update.json"
    receipt.parent.mkdir(parents=True)
    value = {"explicitly_authorized":True,"before":{name:middle},"after":{name:final}}
    receipt.write_text(json.dumps(value),encoding="utf-8")
    assert history.input_matches(name,first,final,update_path=initial)
    assert not history.input_matches("runs/training/adapter.safetensors",first,final,update_path=initial)
    receipt.write_text(json.dumps({**value,"explicitly_authorized":False}),encoding="utf-8")
    assert not history.input_matches(name,first,final,update_path=initial)


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
