"""Export literal-006 only, with complete decoded labels/prefix integrity audit."""

import json
import sys
from hashlib import sha256

import publish_stage1_literal_edit006 as publication
from capability_capsule.eval.dataset_integrity import load_teacher_dataset_publication
from capability_capsule.training.sft import (
    SFTExportManifest, _chat_messages, encode_sft_trajectory, export_sft_dataset, load_sft_examples,
)

ROOT = publication.ROOT
MODEL_ID = "Qwen/Qwen3.5-2B"
REVISION = "15852e8c16360a2fea060d615a32b45270f8a8fc"
EXPORT_ID = "stage1-codex-powershell-literal-edit-006-qwen35-2b-v1"
OUTPUT = ROOT / "artifacts/sft" / EXPORT_ID
EVIDENCE = ROOT / "tests/literal006-sft-audit.json"
SOURCE_DIGEST = "bb07ba632ecb914db4b1f65d93551b1e3c1b205a57fc4fb134a304f4ba38fd6b"
CONTRACT_DIGEST = "c07de6d8212177e43aba96d819e4eb98b996a71e87ae056e6079ad39fd1c0cf0"


def protected_hashes():
    hashes = publication.protected_hashes()
    for path in publication.OUTPUT.iterdir():
        hashes[path.relative_to(ROOT).as_posix()] = sha256(path.read_bytes()).hexdigest()
    prefix = OUTPUT.relative_to(ROOT).as_posix() + "/"
    return {path: digest for path, digest in hashes.items() if not path.startswith(prefix)}


def contract():
    manifest = SFTExportManifest.model_validate_json(
        (ROOT / "artifacts/sft/stage1-codex-powershell-edit-005-qwen35-2b-v1/manifest.json").read_bytes())
    if manifest.chat_contract is None or manifest.chat_contract.sha256() != CONTRACT_DIGEST:
        raise ValueError("Persisted v2 contract changed")
    return manifest.chat_contract


def audit_example(trajectory, tokenizer, chat_contract):
    example = encode_sft_trajectory(trajectory, tokenizer=tokenizer, chat_contract=chat_contract, max_length=4096)
    messages = _chat_messages(trajectory, chat_contract=chat_contract)
    full = tokenizer.apply_chat_template(messages, tools=list(chat_contract.tools), tokenize=True,
        add_generation_prompt=False, enable_thinking=False, truncation=False, return_dict=True)
    if tuple(full["input_ids"]) != example.input_ids or len(example.input_ids) >= 4096:
        raise ValueError("Truncation or tokenization mismatch")
    previous = 0
    spans, prefix_checks = [], 0
    for end, message in enumerate(messages, 1):
        if not any(item["role"] == "user" for item in messages[:end]):
            continue
        encoded = tokenizer.apply_chat_template(messages[:end], tools=list(chat_contract.tools), tokenize=True,
            add_generation_prompt=False, enable_thinking=False, truncation=False, return_dict=True)
        ids = tuple(encoded["input_ids"])
        if ids != example.input_ids[:len(ids)] or len(ids) < previous:
            raise ValueError("Chat prefix does not match complete example")
        labels = example.labels[previous:len(ids)]
        if message["role"] == "assistant":
            if labels != example.input_ids[previous:len(ids)]:
                raise ValueError("Assistant tokens unexpectedly masked")
            spans.append(tokenizer.decode(labels, skip_special_tokens=False))
            generation = tokenizer.apply_chat_template(messages[:end - 1], tools=list(chat_contract.tools),
                tokenize=True, add_generation_prompt=True, enable_thinking=False, return_dict=True)
            generation_ids = tuple(generation["input_ids"])
            if generation_ids != example.input_ids[:len(generation_ids)]:
                raise ValueError("Inference-generation prefix mismatch")
            prefix_checks += 1
        elif any(label != -100 for label in labels):
            raise ValueError("User/system/tool token leaked into training labels")
        previous = len(ids)
    calls = [call for message in trajectory.messages for call in message.tool_calls]
    if len(calls) != 3 or len(spans) != 4 or prefix_checks != 4:
        raise ValueError("Unexpected assistant/tool boundaries")
    for span, call in zip(spans[:3], calls, strict=True):
        if span.count("<tool_call>") != 1 or call.arguments["cmd"] not in span:
            raise ValueError("Tool request missing from trainable span")
        if "<|im_end|>" in span.split("<tool_call>", 1)[0]:
            raise ValueError("Narration ends before its tool call")
    if '"envelope"' in "".join(spans):
        raise ValueError("Tool envelope leaked into assistant targets")
    return example, {"trajectory_id": trajectory.trajectory_id, "source_revision": trajectory.source_revision,
                     "token_count": len(example.input_ids), "trainable_tokens": sum(label != -100 for label in example.labels),
                     "generation_prefix_checks": prefix_checks, "decoded_assistant_turns": spans}


def main():
    if sys.argv[1:]:
        raise ValueError("No arguments supported")
    if OUTPUT.exists() or EVIDENCE.exists():
        raise FileExistsError("Refusing to overwrite an SFT version or audit")
    before = protected_hashes()
    if sha256((publication.OUTPUT / "manifest.json").read_bytes()).hexdigest() != SOURCE_DIGEST:
        raise ValueError("Published source manifest differs from reviewed version")
    dataset = load_teacher_dataset_publication(publication.OUTPUT).dataset
    if (len(dataset.train), len(dataset.validation), len(dataset.duplicates)) != (8, 0, 0):
        raise ValueError("Unexpected source composition")
    chat_contract = contract()
    print("Loading pinned local tokenizer; source and v2 contract verified.", flush=True)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=REVISION, local_files_only=True)
    examples, audits = [], []
    for trajectory in dataset.train:
        example, audited = audit_example(trajectory, tokenizer, chat_contract)
        examples.append(example)
        audits.append(audited)
        print(f"Decoded {trajectory.trajectory_id}: {len(example.input_ids)} tokens; four prefixes matched", flush=True)
    if protected_hashes() != before:
        raise ValueError("Protected inputs changed during audit")
    manifest = export_sft_dataset(publication.OUTPUT, output_root=OUTPUT.parent, export_id=EXPORT_ID,
        tokenizer=tokenizer, tokenizer_id=f"{MODEL_ID}@{REVISION}", chat_contract=chat_contract, max_length=4096)
    if load_sft_examples(OUTPUT / "train.jsonl") != tuple(examples) or load_sft_examples(OUTPUT / "validation.jsonl"):
        raise ValueError("Persisted export differs from decoded audit")
    for artifact in manifest.artifacts:
        payload = (OUTPUT / artifact.filename).read_bytes()
        if len(payload) != artifact.byte_count or sha256(payload).hexdigest() != artifact.sha256:
            raise ValueError("SFT artifact integrity mismatch")
    if protected_hashes() != before:
        raise ValueError("Protected inputs changed during export")
    report = {"export_id": EXPORT_ID, "composition": "literal-edit-006-only", "train_count": 8,
        "validation_count": 0, "source_dataset_digest": SOURCE_DIGEST, "chat_contract_sha256": CONTRACT_DIGEST,
        "total_tokens": sum(len(e.input_ids) for e in examples),
        "trainable_tokens": sum(a["trainable_tokens"] for a in audits),
        "max_sequence_length": max(len(e.input_ids) for e in examples), "audited_tool_calls": 24,
        "generation_prefix_checks": 32, "truncated_examples": 0, "examples": audits,
        "protected_input_hashes": before, "protected_file_count": len(before), "unchanged": True,
        "manifest_sha256": sha256((OUTPUT / "manifest.json").read_bytes()).hexdigest(), "training_started": False}
    publication.collection.save(EVIDENCE, report)
    print(json.dumps({k: v for k, v in report.items() if k not in {"examples", "protected_input_hashes"}}, indent=2), flush=True)


if __name__ == "__main__":
    main()
