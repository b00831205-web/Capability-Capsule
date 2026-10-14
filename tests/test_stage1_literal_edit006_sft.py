"""Verify pinned, untruncated SFT and the complete eight-example decoded audit."""

import json
from hashlib import sha256

import pytest
import export_stage1_literal_edit006_sft as export
from capability_capsule.eval.dataset_integrity import load_teacher_dataset_publication
from capability_capsule.training.sft import SFTExportManifest, load_sft_examples


def read(path):
    return json.loads(path.read_text("utf-8"))


def test_literal006_sft_pins_source_model_contract_and_artifacts():
    manifest = SFTExportManifest.model_validate_json((export.OUTPUT / "manifest.json").read_bytes())
    assert manifest.schema_version == "0.3"
    assert manifest.assistant_turn_policy == "coalesce_adjacent_assistant_messages"
    assert manifest.export_id == export.EXPORT_ID
    assert manifest.source_dataset_id == export.publication.DATASET_ID
    assert manifest.source_dataset_digest == export.SOURCE_DIGEST
    assert manifest.tokenizer_id == f"{export.MODEL_ID}@{export.REVISION}"
    assert manifest.max_length == 4096
    assert manifest.chat_contract == export.contract()
    assert manifest.chat_contract_sha256 == export.CONTRACT_DIGEST
    assert not manifest.chat_contract.enable_thinking
    assert [a.record_count for a in manifest.artifacts] == [8, 0]
    assert {p.name for p in export.OUTPUT.iterdir()} == {"train.jsonl", "validation.jsonl", "manifest.json"}
    for artifact in manifest.artifacts:
        payload = (export.OUTPUT / artifact.filename).read_bytes()
        assert len(payload) == artifact.byte_count
        assert sha256(payload).hexdigest() == artifact.sha256


def test_literal006_sft_preserves_source_identity_and_audits_all_labels():
    source = load_teacher_dataset_publication(export.publication.OUTPUT).dataset.train
    examples = load_sft_examples(export.OUTPUT / "train.jsonl")
    report = read(export.EVIDENCE)
    assert load_sft_examples(export.OUTPUT / "validation.jsonl") == ()
    assert report["composition"] == "literal-edit-006-only"
    assert report["train_count"] == len(source) == len(examples) == len(report["examples"]) == 8
    assert report["validation_count"] == report["truncated_examples"] == 0
    assert report["audited_tool_calls"] == 24 and report["generation_prefix_checks"] == 32
    assert report["total_tokens"] == sum(len(e.input_ids) for e in examples)
    assert report["trainable_tokens"] == sum(sum(label != -100 for label in e.labels) for e in examples)
    assert report["max_sequence_length"] == max(len(e.input_ids) for e in examples) < 4096
    for trajectory, example, audited in zip(source, examples, report["examples"], strict=True):
        assert example.source_trajectory_id == trajectory.trajectory_id == audited["trajectory_id"]
        assert example.source_revision == trajectory.source_revision == audited["source_revision"]
        assert audited["token_count"] == len(example.input_ids)
        assert audited["trainable_tokens"] == sum(label != -100 for label in example.labels)
        assert audited["generation_prefix_checks"] == 4
        spans = audited["decoded_assistant_turns"]
        calls = [call for message in trajectory.messages for call in message.tool_calls]
        assert len(spans) == 4 and len(calls) == 3
        for span, call in zip(spans[:3], calls, strict=True):
            assert span.count("<tool_call>") == 1
            assert call.arguments["cmd"] in span
            assert "<|im_end|>" not in span.split("<tool_call>", 1)[0]
        assert '"envelope"' not in "".join(spans)
        assert "passed" in spans[-1]
        assert any(label == -100 for label in example.labels)


def test_literal006_sft_preserves_previous_inputs_and_refuses_overwrite():
    report = read(export.EVIDENCE)
    assert report["unchanged"] and not report["training_started"]
    assert report["protected_file_count"] == len(report["protected_input_hashes"])
    current = export.protected_hashes()
    assert all(current.get(path) == digest for path, digest in report["protected_input_hashes"].items())
    assert report["manifest_sha256"] == sha256((export.OUTPUT / "manifest.json").read_bytes()).hexdigest()
    before = {path.name: path.read_bytes() for path in export.OUTPUT.iterdir()}
    # main checks argv before checking existing files; pytest itself supplies arguments.
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(export.sys, "argv", ["export_stage1_literal_edit006_sft.py"])
        with pytest.raises(FileExistsError):
            export.main()
    assert {path.name: path.read_bytes() for path in export.OUTPUT.iterdir()} == before
