"""Publish an immutable 8+6 mix and audit/export SFT; never train."""

import json
import sys
from hashlib import sha256

import publish_stage1_literal_edit006 as literal
import publish_stage1_fstring_edit007 as fstring
import export_stage1_literal_edit006_sft as baseline
from capability_capsule.eval.dataset_integrity import load_teacher_dataset_publication
from capability_capsule.eval.dataset_pipeline import curate_teacher_dataset
from capability_capsule.eval.dataset_publication import publish_teacher_dataset
from capability_capsule.eval.dataset_validation import validate_teacher_dataset
from capability_capsule.eval.harness_profile import verify_harness_profile
from capability_capsule.training.sft import export_sft_dataset, load_sft_examples

ROOT = literal.ROOT
DATASET_ID = "stage1-codex-powershell-literal006-fstring007-mix-v1"
EXPORT_ID = DATASET_ID + "-qwen35-2b-v1"
PUBLICATION = ROOT / "datasets/teacher/published" / DATASET_ID
OUTPUT = ROOT / "artifacts/sft" / EXPORT_ID
EVIDENCE = ROOT / "tests/literal006-fstring007-mix-sft-audit.json"
SOURCES = (
    (literal, "bb07ba632ecb914db4b1f65d93551b1e3c1b205a57fc4fb134a304f4ba38fd6b", 8),
    (fstring, "609685d5b2fa0bb764d11eafde057fefca2e12d1ed9b83097e0ad8cf0fcf9117", 6),
)


def protected_hashes():
    hashes = {**literal.protected_hashes(), **fstring.protected_hashes()}
    prefixes = tuple(p.relative_to(ROOT).as_posix() + "/" for p in (PUBLICATION, OUTPUT))
    return {path: digest for path, digest in hashes.items() if not path.startswith(prefixes)}


def sources():
    records, tasks, plans, references = [], [], [], []
    for module, digest, count in SOURCES:
        if sha256((module.OUTPUT / "manifest.json").read_bytes()).hexdigest() != digest:
            raise ValueError("Source manifest changed")
        published = load_teacher_dataset_publication(module.OUTPUT).dataset
        reviewed, validations = module.review()
        if published != reviewed or len(published.train) != count:
            raise ValueError("Reviewed source differs from publication")
        plan = module.load_teacher_collection_plan(module.collection.PLAN_DIR).plan
        records.extend(published.train)
        tasks.extend(a.task for a in plan.assignments)
        plans.append(plan)
        references.append({"dataset_id": module.DATASET_ID, "manifest_sha256": digest,
                           "train_count": count, "independent_validations": validations})
    if plans[0].student_target != plans[1].student_target or plans[0].harness_profile != plans[1].harness_profile:
        raise ValueError("Incompatible Student/harness references")
    harness = verify_harness_profile(plans[0].harness_profile, artifact_root=ROOT)
    validate_teacher_dataset(records, tasks=tasks, harness_profile=harness)
    dataset = curate_teacher_dataset(records, tasks=tasks)
    if dataset.train != tuple(records) or dataset.validation or dataset.duplicates or len(records) != 14:
        raise ValueError("Mix reordered, duplicated or lost records")
    return dataset, references


def audit_example(trajectory, tokenizer, contract):
    # Reuse prefix/mask checks but allow three or four real tool calls.
    from capability_capsule.training.sft import _chat_messages, encode_sft_trajectory
    example = encode_sft_trajectory(trajectory, tokenizer=tokenizer, chat_contract=contract, max_length=4096)
    messages = _chat_messages(trajectory, chat_contract=contract)
    def encode(items, generation=False):
        return tuple(tokenizer.apply_chat_template(items, tools=list(contract.tools), tokenize=True,
            add_generation_prompt=generation, enable_thinking=False, truncation=False, return_dict=True)["input_ids"])
    if encode(messages) != example.input_ids or len(example.input_ids) >= 4096:
        raise ValueError("Truncation or tokenization mismatch")
    previous, spans = 0, []
    for end, message in enumerate(messages, 1):
        if not any(m["role"] == "user" for m in messages[:end]):
            continue
        ids = encode(messages[:end])
        if ids != example.input_ids[:len(ids)] or len(ids) < previous:
            raise ValueError("Chat prefix mismatch")
        labels = example.labels[previous:len(ids)]
        if message["role"] == "assistant":
            if labels != example.input_ids[previous:len(ids)]:
                raise ValueError("Assistant target masked")
            generation = encode(messages[:end-1], True)
            if generation != example.input_ids[:len(generation)]:
                raise ValueError("Inference prefix mismatch")
            spans.append(tokenizer.decode(labels, skip_special_tokens=False))
        elif any(label != -100 for label in labels):
            raise ValueError("Non-assistant target unmasked")
        previous = len(ids)
    if previous != len(example.input_ids):
        raise ValueError("Unaudited token tail")
    calls = [call for message in trajectory.messages for call in message.tool_calls]
    if len(calls) not in {3, 4} or len(spans) != len(calls) + 1:
        raise ValueError("Unexpected tool/assistant boundaries")
    for span, call in zip(spans[:-1], calls, strict=True):
        if span.count("<tool_call>") != 1 or call.arguments["cmd"] not in span:
            raise ValueError("Tool command missing from target")
        if "<|im_end|>" in span.split("<tool_call>", 1)[0]:
            raise ValueError("Narration prematurely terminates")
    if '"envelope"' in "".join(spans):
        raise ValueError("Tool result leaked into targets")
    return example, {"trajectory_id": trajectory.trajectory_id, "source_revision": trajectory.source_revision,
        "token_count": len(example.input_ids), "trainable_tokens": sum(x != -100 for x in example.labels),
        "tool_calls": len(calls), "generation_prefix_checks": len(spans), "decoded_assistant_turns": spans}


def main():
    if sys.argv[1:]:
        raise ValueError("No arguments supported")
    if any(path.exists() for path in (PUBLICATION, OUTPUT, EVIDENCE)):
        raise FileExistsError("Refusing to overwrite mix, export or evidence")
    before = protected_hashes()
    dataset, references = sources()
    contract = baseline.contract()
    print("Loading pinned offline tokenizer; auditing all fourteen trajectories", flush=True)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(baseline.MODEL_ID, revision=baseline.REVISION, local_files_only=True)
    examples, audits = [], []
    for record in dataset.train:
        example, audit = audit_example(record, tokenizer, contract)
        examples.append(example)
        audits.append(audit)
        print(f"Audited {record.trajectory_id}: {audit['token_count']} tokens / {audit['tool_calls']} calls", flush=True)
    if protected_hashes() != before:
        raise ValueError("Protected inputs changed during review")
    publish_teacher_dataset(dataset, output_root=PUBLICATION.parent, dataset_id=DATASET_ID)
    if load_teacher_dataset_publication(PUBLICATION).dataset != dataset:
        raise ValueError("Mix reload differs")
    expected = b"".join((module.OUTPUT / "train.jsonl").read_bytes() for module, _, _ in SOURCES)
    if (PUBLICATION / "train.jsonl").read_bytes() != expected:
        raise ValueError("Mix differs from original record bytes")
    manifest = export_sft_dataset(PUBLICATION, output_root=OUTPUT.parent, export_id=EXPORT_ID,
        tokenizer=tokenizer, tokenizer_id=f"{baseline.MODEL_ID}@{baseline.REVISION}",
        chat_contract=contract, max_length=4096)
    if load_sft_examples(OUTPUT / "train.jsonl") != tuple(examples) or load_sft_examples(OUTPUT / "validation.jsonl"):
        raise ValueError("Saved export differs from audit")
    for artifact in manifest.artifacts:
        payload = (OUTPUT / artifact.filename).read_bytes()
        if len(payload) != artifact.byte_count or sha256(payload).hexdigest() != artifact.sha256:
            raise ValueError("Export integrity mismatch")
    if protected_hashes() != before:
        raise ValueError("Protected inputs changed during publication/export")
    report = {"dataset_id": DATASET_ID, "export_id": EXPORT_ID, "sources": references,
        "composition": "literal006:8 + fstring007:6; one copy each; ordered concatenation",
        "train_count": 14, "validation_count": 0, "duplicate_count": 0, "truncated_examples": 0,
        "source_dataset_digest": sha256((PUBLICATION / "manifest.json").read_bytes()).hexdigest(),
        "sft_manifest_sha256": sha256((OUTPUT / "manifest.json").read_bytes()).hexdigest(),
        "chat_contract_sha256": baseline.CONTRACT_DIGEST,
        "total_tokens": sum(a["token_count"] for a in audits),
        "trainable_tokens": sum(a["trainable_tokens"] for a in audits),
        "max_sequence_length": max(a["token_count"] for a in audits),
        "audited_tool_calls": sum(a["tool_calls"] for a in audits),
        "generation_prefix_checks": sum(a["generation_prefix_checks"] for a in audits),
        "examples": audits, "protected_input_hashes": before, "protected_file_count": len(before),
        "unchanged": True, "training_started": False,
        "limitations": "Same greeting domain; known exact leakage review is not semantic independence or anti-overfitting proof"}
    fstring.collection.save(EVIDENCE, report)
    print(json.dumps({k:v for k,v in report.items() if k not in {"examples", "sources", "protected_input_hashes"}}, indent=2), flush=True)


if __name__ == "__main__":
    main()
