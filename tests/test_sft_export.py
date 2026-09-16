"""Tests for immutable Qwen-compatible SFT export."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from capability_capsule.training.sft import (
    SFTExample,
    encode_sft_trajectory,
    export_sft_dataset,
    load_sft_examples,
)

from capability_capsule.eval.dataset_pipeline import CuratedTeacherDataset
from capability_capsule.eval.dataset_publication import publish_teacher_dataset
from capability_capsule.eval.records import (
    DatasetSplit,
    MessageRole,
    TeacherTrajectory,
    ToolCallRecord,
    TrajectoryMessage,
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

    _prefix_lengths = {1: 2, 2: 4, 3: 5, 4: 7}

    def apply_chat_template(
        self,
        conversation: list[dict[str, Any]],
        **kwargs: Any,
    ) -> Any:
        length = self._prefix_lengths[len(conversation)]
        input_ids = list(range(10, 10 + length))

        if kwargs.get("return_dict"):
            return {
                "input_ids": input_ids,
                "attention_mask": [1] * length,
                "assistant_masks": [0] * length,
            }

        return input_ids


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
        max_length=2048,
    )

    assert example.source_trajectory_id == "train-001"
    assert example.source_revision == "revision-train-001"
    assert example.input_ids == (10, 11, 12, 13, 14, 15)
    assert example.attention_mask == (1, 1, 1, 1, 1, 1)
    assert example.labels == (-100, -100, 12, 13, -100, 15)

    rendered = tokenizer.conversations[0]
    assert rendered[0] == {"role": "user", "content": "Change the greeting."}
    assert rendered[1]["role"] == "assistant"
    assert rendered[1]["tool_calls"] == [
        {
            "type": "function",
            "function": {
                "name": "exec_command",
                "arguments": {"cmd": "sed -n '1,80p' greeting.py"},
            },
        }
    ]
    assert rendered[2] == {
        "role": "tool",
        "content": 'return f"Hello, {name}"',
        "name": "exec_command",
    }
    assert tokenizer.keyword_arguments[0] == {
        "tokenize": True,
        "add_generation_prompt": False,
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
            max_length=2048,
        )


def test_encode_trajectory_derives_assistant_spans_from_message_prefixes() -> None:
    example = encode_sft_trajectory(
        make_trajectory("train-001", DatasetSplit.TRAIN),
        tokenizer=PrefixFallbackTokenizer(),
        max_length=2048,
    )

    assert example.input_ids == (10, 11, 12, 13, 14, 15, 16)
    assert example.labels == (-100, -100, 12, 13, -100, 15, 16)


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
        max_length=2048,
    )

    export_dir = output_root / "qwen-smoke-001"
    assert manifest.export_id == "qwen-smoke-001"
    assert manifest.source_dataset_id == "teacher-smoke-001"
    assert len(manifest.source_dataset_digest) == 64
    assert manifest.tokenizer_id == "Qwen/Qwen3.5-0.8B@revision-001"
    assert manifest.max_length == 2048
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
            max_length=2048,
        )
