"""Tests for immutable Qwen-compatible SFT export."""

from __future__ import annotations

from collections import UserDict
from hashlib import sha256
from pathlib import Path
from typing import Any, ClassVar

import pytest

from capability_capsule.eval.dataset_pipeline import CuratedTeacherDataset
from capability_capsule.eval.dataset_publication import publish_teacher_dataset
from capability_capsule.eval.records import (
    DatasetSplit,
    MessageRole,
    TeacherTrajectory,
    ToolCallRecord,
    TrajectoryMessage,
)
from capability_capsule.training.sft import (
    SFTChatContract,
    SFTExample,
    SFTExportManifest,
    encode_sft_trajectory,
    export_sft_dataset,
    load_sft_examples,
)


class RecordingTokenizer:
    """Small chat-template double exposing the Transformers contract we require."""

    def __init__(self, *, assistant_masks: list[int] | None = None) -> None:
        self.assistant_masks = assistant_masks or [0, 0, 1, 1, 0, 1]
        self.conversations: list[list[dict[str, Any]]] = []
        self.keyword_arguments: list[dict[str, Any]] = []

    def apply_chat_template(
        self,
        conversation: list[dict[str, Any]],
        **kwargs: Any,
    ) -> dict[str, list[int]]:
        self.conversations.append(conversation)
        self.keyword_arguments.append(kwargs)
        return {
            "input_ids": [10, 11, 12, 13, 14, 15],
            "attention_mask": [1, 1, 1, 1, 1, 1],
            "assistant_masks": self.assistant_masks,
        }


class PrefixFallbackTokenizer:
    """Template double without generation blocks or a native assistant mask."""

    _prefix_lengths: ClassVar[dict[int, int]] = {
        1: 1,
        2: 2,
        3: 4,
        4: 5,
        5: 7,
    }

    def __init__(self, *, include_zero_mask: bool = True) -> None:
        self.include_zero_mask = include_zero_mask
        self.conversations: list[list[dict[str, Any]]] = []
        self.keyword_arguments: list[dict[str, Any]] = []

    def apply_chat_template(
        self,
        conversation: list[dict[str, Any]],
        **kwargs: Any,
    ) -> Any:
        self.conversations.append(conversation)
        self.keyword_arguments.append(kwargs)
        length = self._prefix_lengths[len(conversation)]
        input_ids = list(range(10, 10 + length))

        if kwargs.get("return_dict"):
            result = {
                "input_ids": input_ids,
                "attention_mask": [1] * length,
            }
            if self.include_zero_mask:
                result["assistant_masks"] = [0] * length
            return result

        return UserDict({"input_ids": input_ids})


def make_chat_contract() -> SFTChatContract:
    return SFTChatContract(
        contract_id="powershell-current-workspace-cmd-v1",
        system_prompt=(
            "You are editing files in the current workspace. "
            "Use the declared PowerShell tool, modify only requested files, "
            "and run the supplied tests."
        ),
        tools=(
            {
                "type": "function",
                "function": {
                    "name": "exec_command",
                    "description": (
                        "Run one PowerShell command in the current workspace."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "cmd": {"type": "string"},
                        },
                        "required": ["cmd"],
                        "additionalProperties": False,
                    },
                },
            },
        ),
        enable_thinking=False,
    )


def make_trajectory(
    trajectory_id: str,
    split: DatasetSplit,
) -> TeacherTrajectory:
    return TeacherTrajectory(
        trajectory_id=trajectory_id,
        task_id=f"task-{trajectory_id}",
        split=split,
        teacher_model="gpt-5",
        teacher_skill_version="0.2.0",
        task="Read greeting.py, make the requested change, and validate it.",
        source_revision=f"revision-{trajectory_id}",
        messages=(
            TrajectoryMessage(
                role=MessageRole.USER,
                content="Change the greeting.",
            ),
            TrajectoryMessage(
                role=MessageRole.ASSISTANT,
                content="I will inspect the file.",
                tool_calls=(
                    ToolCallRecord(
                        name="exec_command",
                        arguments={"cmd": "sed -n '1,80p' greeting.py"},
                    ),
                ),
            ),
            TrajectoryMessage(
                role=MessageRole.TOOL,
                content='return f"Hello, {name}"',
                tool_name="exec_command",
            ),
            TrajectoryMessage(
                role=MessageRole.ASSISTANT,
                content="The greeting was updated and validation passed.",
            ),
        ),
        tags=("smoke-pilot",),
    )


def test_encode_trajectory_uses_chat_template_and_masks_non_assistant_tokens() -> None:
    tokenizer = RecordingTokenizer()
    trajectory = make_trajectory("train-001", DatasetSplit.TRAIN)

    example = encode_sft_trajectory(
        trajectory,
        tokenizer=tokenizer,
        chat_contract=make_chat_contract(),
        max_length=2048,
    )

    assert example.source_trajectory_id == "train-001"
    assert example.source_revision == "revision-train-001"
    assert example.input_ids == (10, 11, 12, 13, 14, 15)
    assert example.attention_mask == (1, 1, 1, 1, 1, 1)
    assert example.labels == (-100, -100, 12, 13, -100, 15)

    rendered = tokenizer.conversations[0]
    assert rendered[0] == {
        "role": "system",
        "content": (
            "You are editing files in the current workspace. "
            "Use the declared PowerShell tool, modify only requested files, "
            "and run the supplied tests."
        ),
    }
    assert rendered[1] == {"role": "user", "content": "Change the greeting."}
    assert rendered[2]["role"] == "assistant"
    assert rendered[2]["tool_calls"] == [
        {
            "type": "function",
            "function": {
                "name": "exec_command",
                "arguments": {"cmd": "sed -n '1,80p' greeting.py"},
            },
        }
    ]
    assert rendered[3] == {
        "role": "tool",
        "content": 'return f"Hello, {name}"',
        "name": "exec_command",
    }
    assert tokenizer.keyword_arguments[0] == {
        "tools": list(make_chat_contract().tools),
        "tokenize": True,
        "add_generation_prompt": False,
        "enable_thinking": False,
        "return_dict": True,
        "return_assistant_tokens_mask": True,
        "truncation": True,
        "max_length": 2048,
    }


def test_encode_trajectory_rejects_template_without_trainable_assistant_tokens() -> None:
    tokenizer = RecordingTokenizer(assistant_masks=[0, 0, 0, 0, 0, 0])

    with pytest.raises(ValueError, match="assistant tokens"):
        encode_sft_trajectory(
            make_trajectory("train-001", DatasetSplit.TRAIN),
            tokenizer=tokenizer,
            chat_contract=make_chat_contract(),
            max_length=2048,
        )


@pytest.mark.parametrize("include_zero_mask", [False, True])
def test_encode_trajectory_derives_assistant_spans_from_message_prefixes(
    include_zero_mask: bool,
) -> None:
    tokenizer = PrefixFallbackTokenizer(
        include_zero_mask=include_zero_mask,
    )
    example = encode_sft_trajectory(
        make_trajectory("train-001", DatasetSplit.TRAIN),
        tokenizer=tokenizer,
        chat_contract=make_chat_contract(),
        max_length=2048,
    )

    assert example.input_ids == (10, 11, 12, 13, 14, 15, 16)
    assert example.labels == (-100, -100, 12, 13, -100, 15, 16)
    assert all(
        call["tools"] == list(make_chat_contract().tools)
        for call in tokenizer.keyword_arguments
    )
    assert all(
        call["enable_thinking"] is False
        for call in tokenizer.keyword_arguments
    )
    assert all(
        any(message["role"] == "user" for message in conversation)
        for conversation in tokenizer.conversations
    )


def test_encode_trajectory_rejects_arguments_outside_chat_contract() -> None:
    trajectory = make_trajectory(
        "train-extra-argument",
        DatasetSplit.TRAIN,
    )
    tool_call = trajectory.messages[1].tool_calls[0].model_copy(
        update={
            "arguments": {
                "cmd": "Get-Content -LiteralPath greeting.py -Raw",
                "workdir": "E:/capsule/tmp/workspace",
            }
        }
    )
    messages = list(trajectory.messages)
    messages[1] = messages[1].model_copy(
        update={"tool_calls": (tool_call,)}
    )
    trajectory = trajectory.model_copy(
        update={"messages": tuple(messages)}
    )

    with pytest.raises(
        ValueError,
        match=r"outside the SFT contract: workdir",
    ):
        encode_sft_trajectory(
            trajectory,
            tokenizer=RecordingTokenizer(),
            chat_contract=make_chat_contract(),
            max_length=2048,
        )


def test_export_sft_dataset_is_immutable_and_traceable(tmp_path: Path) -> None:
    curated = CuratedTeacherDataset(
        train=(make_trajectory("train-001", DatasetSplit.TRAIN),),
        validation=(
            make_trajectory("validation-001", DatasetSplit.VALIDATION),
        ),
        duplicates=(),
    )
    publication_root = tmp_path / "teacher"
    publish_teacher_dataset(
        curated,
        output_root=publication_root,
        dataset_id="teacher-smoke-001",
    )
    output_root = tmp_path / "sft"

    manifest = export_sft_dataset(
        publication_root / "teacher-smoke-001",
        output_root=output_root,
        export_id="qwen-smoke-001",
        tokenizer=RecordingTokenizer(),
        tokenizer_id="Qwen/Qwen3.5-0.8B@revision-001",
        chat_contract=make_chat_contract(),
        max_length=2048,
    )

    export_dir = output_root / "qwen-smoke-001"
    assert manifest.export_id == "qwen-smoke-001"
    assert manifest.source_dataset_id == "teacher-smoke-001"
    assert len(manifest.source_dataset_digest) == 64
    assert manifest.tokenizer_id == "Qwen/Qwen3.5-0.8B@revision-001"
    assert manifest.max_length == 2048
    assert manifest.schema_version == "0.2"
    assert manifest.chat_contract == make_chat_contract()
    assert manifest.chat_contract_sha256 == make_chat_contract().sha256()
    assert [artifact.filename for artifact in manifest.artifacts] == [
        "train.jsonl",
        "validation.jsonl",
    ]
    assert [artifact.record_count for artifact in manifest.artifacts] == [1, 1]
    assert (export_dir / "manifest.json").is_file()
    assert load_sft_examples(export_dir / "train.jsonl") == (
        SFTExample(
            source_trajectory_id="train-001",
            source_revision="revision-train-001",
            input_ids=(10, 11, 12, 13, 14, 15),
            attention_mask=(1, 1, 1, 1, 1, 1),
            labels=(-100, -100, 12, 13, -100, 15),
        ),
    )

    with pytest.raises(FileExistsError, match="qwen-smoke-001"):
        export_sft_dataset(
            publication_root / "teacher-smoke-001",
            output_root=output_root,
            export_id="qwen-smoke-001",
            tokenizer=RecordingTokenizer(),
            tokenizer_id="Qwen/Qwen3.5-0.8B@revision-001",
            chat_contract=make_chat_contract(),
            max_length=2048,
        )


def test_checked_in_stage1_qwen_export_is_complete_and_untruncated() -> None:
    repository_root = Path(__file__).resolve().parents[1]
    export_dir = (
        repository_root
        / "artifacts"
        / "sft"
        / "stage1-codex-001-qwen35-2b-v2"
    )
    manifest = SFTExportManifest.model_validate_json(
        (export_dir / "manifest.json").read_text(encoding="utf-8")
    )
    train = load_sft_examples(export_dir / "train.jsonl")
    validation = load_sft_examples(export_dir / "validation.jsonl")
    examples = train + validation

    assert manifest.source_dataset_id == "stage1-codex-001"
    assert manifest.source_dataset_digest == sha256(
        (
            repository_root
            / "datasets"
            / "teacher"
            / "published"
            / "stage1-codex-001"
            / "manifest.json"
        ).read_bytes()
    ).hexdigest()
    assert manifest.tokenizer_id == (
        "Qwen/Qwen3.5-2B@15852e8c16360a2fea060d615a32b45270f8a8fc"
    )
    assert manifest.max_length == 4096
    assert len(train) == 8
    assert len(validation) == 2
    assert max(len(example.input_ids) for example in examples) == 2167
    assert all(len(example.input_ids) < manifest.max_length for example in examples)


def test_checked_in_stage1_powershell_qwen_export_is_untruncated() -> None:
    root = Path(__file__).resolve().parents[1]
    export_dir = (
        root
        / "artifacts"
        / "sft"
        / "stage1-codex-powershell-002-qwen35-2b-v1"
    )
    manifest = SFTExportManifest.model_validate_json(
        (export_dir / "manifest.json").read_bytes()
    )
    train = load_sft_examples(export_dir / "train.jsonl")
    validation = load_sft_examples(export_dir / "validation.jsonl")
    examples = train + validation

    assert manifest.source_dataset_id == "stage1-codex-powershell-002"
    assert manifest.max_length == 4096
    assert len(train) == 16
    assert len(validation) == 2
    assert max(len(example.input_ids) for example in examples) == 2167
    assert all(len(example.input_ids) < manifest.max_length for example in examples)

    for artifact in manifest.artifacts:
        payload = (export_dir / artifact.filename).read_bytes()
        assert artifact.byte_count == len(payload)
        assert artifact.sha256 == sha256(payload).hexdigest()


def test_checked_in_powershell_edit_qwen_export_is_untruncated() -> None:
    root = Path(__file__).resolve().parents[1]
    export_dir = (
        root
        / "artifacts"
        / "sft"
        / "stage1-codex-powershell-edit-003-qwen35-2b-v1"
    )
    manifest = SFTExportManifest.model_validate_json(
        (export_dir / "manifest.json").read_bytes()
    )
    train = load_sft_examples(export_dir / "train.jsonl")
    validation = load_sft_examples(export_dir / "validation.jsonl")
    examples = train + validation

    assert manifest.source_dataset_id == "stage1-codex-powershell-edit-003"
    assert manifest.source_dataset_digest == sha256(
        (
            root
            / "datasets"
            / "teacher"
            / "published"
            / "stage1-codex-powershell-edit-003"
            / "manifest.json"
        ).read_bytes()
    ).hexdigest()
    assert manifest.max_length == 4096
    assert len(train) == 24
    assert len(validation) == 2
    assert max(len(example.input_ids) for example in examples) == 2986
    assert all(len(example.input_ids) < manifest.max_length for example in examples)


def test_checked_in_contract_004_qwen_export_pins_the_eval_contract() -> None:
    root = Path(__file__).resolve().parents[1]
    export_dir = (
        root
        / "artifacts"
        / "sft"
        / "stage1-codex-powershell-contract-004-qwen35-2b-v1"
    )
    manifest = SFTExportManifest.model_validate_json(
        (export_dir / "manifest.json").read_bytes()
    )
    train = load_sft_examples(export_dir / "train.jsonl")
    validation = load_sft_examples(export_dir / "validation.jsonl")

    assert manifest.schema_version == "0.2"
    assert manifest.source_dataset_id == "stage1-codex-powershell-contract-004"
    assert manifest.source_dataset_digest == sha256(
        (
            root
            / "datasets"
            / "teacher"
            / "published"
            / "stage1-codex-powershell-contract-004"
            / "manifest.json"
        ).read_bytes()
    ).hexdigest()
    assert manifest.chat_contract.contract_id == (
        "stage1-codex-validation-v2-cmd-only-v1"
    )
    tool_parameters = manifest.chat_contract.tools[0]["function"]["parameters"]
    assert tool_parameters["required"] == ["cmd"]
    assert tool_parameters["additionalProperties"] is False
    assert manifest.chat_contract_sha256 == (
        "c07de6d8212177e43aba96d819e4eb98b996a71e87ae056e6079ad39fd1c0cf0"
    )
    assert len(train) == 8
    assert validation == ()
    assert max(len(example.input_ids) for example in train) == 838
    assert all(len(example.input_ids) < manifest.max_length for example in train)
    for artifact in manifest.artifacts:
        payload = (export_dir / artifact.filename).read_bytes()
        assert artifact.byte_count == len(payload)
        assert artifact.sha256 == sha256(payload).hexdigest()
