"""Audit edit-005 in memory; --export writes its new immutable SFT version."""

import json
import sys
from hashlib import sha256
from pathlib import Path

from transformers import AutoTokenizer

from capability_capsule.eval.dataset_integrity import load_teacher_dataset_publication
from capability_capsule.training.sft import (
    SFTExportManifest,
    encode_sft_trajectory,
    export_sft_dataset,
    load_sft_examples,
)


ROOT = Path(__file__).resolve().parents[1]
MODEL_ID = "Qwen/Qwen3.5-2B"
REVISION = "15852e8c16360a2fea060d615a32b45270f8a8fc"
EXPORT_ID = "stage1-codex-powershell-edit-005-qwen35-2b-v1"


def main() -> None:
    if sys.argv[1:] not in ([], ["--export"]):
        raise ValueError("Only --export is supported")
    publication = ROOT / "datasets/teacher/published/stage1-codex-powershell-edit-005"
    dataset = load_teacher_dataset_publication(publication).dataset
    prior_manifest = SFTExportManifest.model_validate_json(
        (ROOT / "artifacts/sft/stage1-codex-powershell-contract-004-qwen35-2b-v2/manifest.json").read_bytes()
    )
    contract = prior_manifest.chat_contract
    if contract is None or (len(dataset.train), len(dataset.validation)) != (12, 0):
        raise ValueError("Unexpected contract or source partition")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=REVISION, local_files_only=True)
    examples = []
    tool_count = 0
    for trajectory in dataset.train:
        example = encode_sft_trajectory(
            trajectory, tokenizer=tokenizer, chat_contract=contract, max_length=4096,
        )
        if len(example.input_ids) >= 4096:
            raise ValueError("Potentially truncated trajectory")
        spans = []
        current = []
        for label in example.labels:
            if label == -100:
                if current:
                    spans.append(tokenizer.decode(current, skip_special_tokens=False))
                    current = []
            else:
                current.append(label)
        if current:
            spans.append(tokenizer.decode(current, skip_special_tokens=False))
        calls = [call for message in trajectory.messages for call in message.tool_calls]
        if len(calls) != 3 or len(spans) != 4:
            raise ValueError("Unexpected tool count or assistant-turn boundaries")
        for span, call in zip(spans[:3], calls, strict=True):
            if span.count("<tool_call>") != 1 or call.arguments["cmd"] not in span:
                raise ValueError("Tool call is missing from its trainable assistant turn")
            if "<|im_end|>" in span.split("<tool_call>", 1)[0]:
                raise ValueError("Narration ends before its tool call")
        if '"envelope"' in "".join(spans):
            raise ValueError("Tool-result envelope leaked into assistant labels")
        examples.append(example)
        tool_count += len(calls)
    print(json.dumps({
        "export_id": EXPORT_ID,
        "composition": "edit-005-only",
        "train_examples": len(examples),
        "validation_examples": 0,
        "total_tokens": sum(len(item.input_ids) for item in examples),
        "trainable_tokens": sum(sum(label != -100 for label in item.labels) for item in examples),
        "max_sequence_length": max(len(item.input_ids) for item in examples),
        "audited_trainable_tool_calls": tool_count,
        "chat_contract_sha256": contract.sha256(),
    }, indent=2), flush=True)
    if sys.argv[1:] == ["--export"]:
        manifest = export_sft_dataset(
            publication, output_root=ROOT / "artifacts/sft", export_id=EXPORT_ID,
            tokenizer=tokenizer, tokenizer_id=f"{MODEL_ID}@{REVISION}",
            chat_contract=contract, max_length=4096,
        )
        export_dir = ROOT / "artifacts/sft" / EXPORT_ID
        if load_sft_examples(export_dir / "train.jsonl") != tuple(examples):
            raise ValueError("Persisted examples differ from the decoded audit")
        if load_sft_examples(export_dir / "validation.jsonl"):
            raise ValueError("Unexpected validation examples")
        for artifact in manifest.artifacts:
            payload = (export_dir / artifact.filename).read_bytes()
            if len(payload) != artifact.byte_count or sha256(payload).hexdigest() != artifact.sha256:
                raise ValueError("SFT artifact integrity mismatch")
        print(manifest.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
