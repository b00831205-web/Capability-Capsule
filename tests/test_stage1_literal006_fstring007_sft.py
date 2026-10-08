"""Exact immutable mix, complete assistant targets and no automatic training."""

import json
from hashlib import sha256

import pytest
import prepare_stage1_literal006_fstring007_sft as mix
from teacher_replay_history import distinct_history
from capability_capsule.eval.dataset_integrity import load_teacher_dataset_publication
from capability_capsule.training.sft import SFTExportManifest, load_sft_examples


def report():
    return json.loads(mix.EVIDENCE.read_text("utf-8"))


def test_mix_contains_exactly_original_eight_plus_six_records_once():
    datasets = [load_teacher_dataset_publication(module.OUTPUT).dataset for module, _, _ in mix.SOURCES]
    published = load_teacher_dataset_publication(mix.PUBLICATION)
    assert published.dataset.train == datasets[0].train + datasets[1].train
    assert len({r.trajectory_id for r in published.dataset.train}) == 14
    assert published.dataset.validation == published.dataset.duplicates == ()
    assert [a.record_count for a in published.manifest.artifacts] == [14, 0, 0]
    assert {p.name for p in mix.PUBLICATION.iterdir()} == {"train.jsonl", "validation.jsonl", "duplicates.jsonl", "manifest.json"}
    assert (mix.PUBLICATION / "train.jsonl").read_bytes() == b"".join(
        (module.OUTPUT / "train.jsonl").read_bytes() for module, _, _ in mix.SOURCES)
    for module, digest, _ in mix.SOURCES:
        assert sha256((module.OUTPUT / "manifest.json").read_bytes()).hexdigest() == digest


def test_mix_sft_pins_model_schema_source_and_unchanged_contract():
    manifest = SFTExportManifest.model_validate_json((mix.OUTPUT / "manifest.json").read_bytes())
    assert manifest.schema_version == "0.3"
    assert manifest.assistant_turn_policy == "coalesce_adjacent_assistant_messages"
    assert manifest.source_dataset_id == mix.DATASET_ID
    assert manifest.source_dataset_digest == report()["source_dataset_digest"]
    assert manifest.tokenizer_id == f"{mix.baseline.MODEL_ID}@{mix.baseline.REVISION}"
    assert manifest.chat_contract == mix.baseline.contract()
    assert manifest.chat_contract_sha256 == mix.baseline.CONTRACT_DIGEST
    assert manifest.max_length == 4096
    assert [a.record_count for a in manifest.artifacts] == [14, 0]
    assert {p.name for p in mix.OUTPUT.iterdir()} == {"train.jsonl", "validation.jsonl", "manifest.json"}
    for artifact in manifest.artifacts:
        payload = (mix.OUTPUT / artifact.filename).read_bytes()
        assert len(payload) == artifact.byte_count
        assert sha256(payload).hexdigest() == artifact.sha256


def test_all_fourteen_decoded_targets_cover_every_real_edit_and_mask_other_roles():
    data = report()
    records = load_teacher_dataset_publication(mix.PUBLICATION).dataset.train
    examples = load_sft_examples(mix.OUTPUT / "train.jsonl")
    assert len(records) == len(examples) == len(data["examples"]) == 14
    assert load_sft_examples(mix.OUTPUT / "validation.jsonl") == ()
    assert data["truncated_examples"] == data["validation_count"] == data["duplicate_count"] == 0
    assert data["audited_tool_calls"] == 45
    assert data["generation_prefix_checks"] == 59
    assert data["total_tokens"] == sum(len(e.input_ids) for e in examples)
    assert data["trainable_tokens"] == sum(sum(x != -100 for x in e.labels) for e in examples)
    assert data["max_sequence_length"] == max(len(e.input_ids) for e in examples) < 4096
    for record, example, audit in zip(records, examples, data["examples"], strict=True):
        assert record.trajectory_id == example.source_trajectory_id == audit["trajectory_id"]
        assert record.source_revision == example.source_revision == audit["source_revision"]
        assert audit["token_count"] == len(example.input_ids)
        assert audit["trainable_tokens"] == sum(x != -100 for x in example.labels)
        calls = [call for message in record.messages for call in message.tool_calls]
        spans = audit["decoded_assistant_turns"]
        assert len(calls) == audit["tool_calls"]
        assert len(spans) == audit["generation_prefix_checks"] == len(calls) + 1
        for span, call in zip(spans[:-1], calls, strict=True):
            assert span.count("<tool_call>") == 1 and call.arguments["cmd"] in span
            assert "<|im_end|>" not in span.split("<tool_call>", 1)[0]
        assert '"envelope"' not in "".join(spans)
        assert any(x == -100 for x in example.labels)
    # Old examples must not change through mixing or re-encoding.
    assert examples[:8] == load_sft_examples(mix.baseline.OUTPUT / "train.jsonl")


def test_mix_preserves_all_old_artifacts_and_records_independent_validation():
    data = report()
    assert data["unchanged"] and not data["training_started"]
    current = mix.protected_hashes()
    assert data["protected_file_count"] == len(data["protected_input_hashes"])
    assert all(current.get(path) == digest for path, digest in data["protected_input_hashes"].items())
    assert sum(len(s["independent_validations"]) for s in data["sources"]) == 14
    assert all(v["exit_code"] == 0 and "passed" in v["output"]
        for source in data["sources"] for v in source["independent_validations"])
    assert data["source_dataset_digest"] == sha256((mix.PUBLICATION / "manifest.json").read_bytes()).hexdigest()
    assert data["sft_manifest_sha256"] == sha256((mix.OUTPUT / "manifest.json").read_bytes()).hexdigest()


def test_mix_refuses_overwrite_and_tampered_source_manifest(monkeypatch):
    before = {str(p): p.read_bytes() for root in (mix.PUBLICATION, mix.OUTPUT) for p in root.iterdir()}
    monkeypatch.setattr(mix.sys, "argv", ["prepare.py"])
    with pytest.raises(FileExistsError):
        mix.main()
    assert {str(p): p.read_bytes() for root in (mix.PUBLICATION, mix.OUTPUT) for p in root.iterdir()} == before
    module, _, count = mix.SOURCES[0]
    monkeypatch.setattr(mix, "SOURCES", ((module, "0" * 64, count),))
    with pytest.raises(ValueError, match="Source manifest changed"):
        mix.sources()


def test_history_allows_exact_origin_references_but_rejects_conflicting_ids():
    record = load_teacher_dataset_publication(mix.PUBLICATION).dataset.train[0]
    assert distinct_history([record], [record]) == []
    changed = record.model_copy(update={"task": record.task + " changed"})
    with pytest.raises(ValueError, match="Conflicting content"):
        distinct_history([record], [changed])
    different_id = record.model_copy(update={"trajectory_id": "independent-duplicate"})
    assert distinct_history([record], [different_id]) == [different_id]
