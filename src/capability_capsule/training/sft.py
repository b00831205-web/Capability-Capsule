"""Immutable Qwen-compatible SFT export with assistant-only loss labels."""

from __future__ import annotations

import shutil
from collections.abc import Sequence
from hashlib import sha256
from pathlib import Path
from tempfile import mkdtemp
from typing import Any, Literal, Protocol, Self, cast

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

from capability_capsule.eval.dataset_integrity import (
    load_teacher_dataset_publication,
)
from capability_capsule.eval.jsonl import load_jsonl
from capability_capsule.eval.records import TeacherTrajectory


class ChatTemplateTokenizer(Protocol):
    """Tokenizer surface required by the exporter."""

    def apply_chat_template(
        self,
        conversation: list[dict[str, Any]],
        **kwargs: Any,
    ) -> Any:
        """Render and tokenize one chat trajectory."""


class SFTExample(BaseModel):
    """One tokenized trajectory with assistant-only training labels."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["0.1"] = "0.1"
    source_trajectory_id: str = Field(min_length=1)
    source_revision: str = Field(min_length=1)
    input_ids: tuple[int, ...] = Field(min_length=1)
    attention_mask: tuple[int, ...] = Field(min_length=1)
    labels: tuple[int, ...] = Field(min_length=1)

    @field_validator("source_trajectory_id", "source_revision")
    @classmethod
    def reject_blank_source_fields(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("SFT source fields must not be blank")
        return value

    @model_validator(mode="after")
    def validate_token_vectors(self) -> Self:
        lengths = {
            len(self.input_ids),
            len(self.attention_mask),
            len(self.labels),
        }
        if len(lengths) != 1:
            raise ValueError(
                "input_ids, attention_mask, and labels must have equal lengths"
            )

        if any(token_id < 0 for token_id in self.input_ids):
            raise ValueError("input_ids must be non-negative")

        if any(value not in {0, 1} for value in self.attention_mask):
            raise ValueError("attention_mask values must be zero or one")

        if all(label == -100 for label in self.labels):
            raise ValueError("SFT example must contain trainable assistant tokens")

        for token_id, label in zip(self.input_ids, self.labels, strict=True):
            if label != -100 and label != token_id:
                raise ValueError(
                    "Unmasked labels must equal their corresponding input token"
                )

        return self


class SFTArtifact(BaseModel):
    """Integrity metadata for one exported SFT split."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    filename: Literal["train.jsonl", "validation.jsonl"]
    record_count: int = Field(ge=0)
    byte_count: int = Field(ge=0)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class SFTExportManifest(BaseModel):
    """Traceable identity of one tokenizer-specific SFT export."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["0.1"] = "0.1"
    export_id: str = Field(min_length=1)
    source_dataset_id: str = Field(min_length=1)
    source_dataset_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    tokenizer_id: str = Field(min_length=1)
    max_length: int = Field(gt=0)
    artifacts: tuple[SFTArtifact, ...] = Field(
        min_length=2,
        max_length=2,
    )

    @field_validator(
        "export_id",
        "source_dataset_id",
        "tokenizer_id",
    )
    @classmethod
    def reject_blank_identifiers(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("SFT export identifiers must not be blank")
        return value

    @field_validator("artifacts")
    @classmethod
    def require_canonical_artifact_order(
        cls,
        value: tuple[SFTArtifact, ...],
    ) -> tuple[SFTArtifact, ...]:
        filenames = tuple(artifact.filename for artifact in value)
        if filenames != ("train.jsonl", "validation.jsonl"):
            raise ValueError(
                "SFT artifacts must contain train and validation "
                "in canonical order"
            )
        return value


def _validate_export_id(export_id: str) -> str:
    if (
        not export_id.strip()
        or export_id != export_id.strip()
        or export_id in {".", ".."}
        or "/" in export_id
        or "\\" in export_id
        or "\0" in export_id
    ):
        raise ValueError(
            "export_id must be a safe, non-blank directory name"
        )
    return export_id


def _chat_messages(
    trajectory: TeacherTrajectory,
) -> list[dict[str, Any]]:
    rendered: list[dict[str, Any]] = []

    for message in trajectory.messages:
        item: dict[str, Any] = {
            "role": message.role.value,
            "content": message.content,
        }

        if message.tool_calls:
            item["tool_calls"] = [
                {
                    "type": "function",
                    "function": {
                        "name": tool_call.name,
                        "arguments": tool_call.arguments,
                    },
                }
                for tool_call in message.tool_calls
            ]

        if message.tool_name is not None:
            item["name"] = message.tool_name

        rendered.append(item)

    return rendered


def _token_vector(value: Any, *, field_name: str) -> tuple[int, ...]:
    if hasattr(value, "tolist"):
        value = value.tolist()

    if (
        isinstance(value, Sequence)
        and not isinstance(value, (str, bytes, bytearray))
        and len(value) == 1
        and isinstance(value[0], Sequence)
        and not isinstance(value[0], (str, bytes, bytearray))
    ):
        value = value[0]

    if not isinstance(value, Sequence) or isinstance(
        value,
        (str, bytes, bytearray),
    ):
        raise ValueError(f"Tokenizer {field_name} must be a token sequence")

    result: list[int] = []
    for item in value:
        if not isinstance(item, int):
            raise ValueError(
                f"Tokenizer {field_name} must contain only integers"
            )
        result.append(int(item))

    if not result:
        raise ValueError(f"Tokenizer {field_name} must not be empty")

    return tuple(result)

def _derive_assistant_mask_from_prefixes(
        messages: list[dict[str, Any]],
        *,
        tokenizer: ChatTemplateTokenizer,
        input_ids: tuple[int, ...],
        max_length: int,
) -> tuple[int, ...]:
    """Derive assistant spans when the chat template has no generation blocks."""

    assistant_mask = [0] * len(input_ids)
    previous_prefix: tuple[int, ...] = ()

    for end, message in enumerate(messages, start=1):
        rendered = tokenizer.apply_chat_template(
            messages[:end],
            tokenize = True,
            add_generation_prompt = False,
            truncation = True,
            max_length = max_length,
        )

        if hasattr(rendered, "get"):
            rendered = rendered.get("input_ids")

        prefix_ids = _token_vector(rendered, field_name="prefix input_ids")
        if len(prefix_ids) < len(previous_prefix):
            raise ValueError(
                "Chat-template prefixes must grow monotonically"
            )

        if (len(prefix_ids) > len(input_ids) or input_ids[: len(prefix_ids)] != prefix_ids):
            raise ValueError(
                "Chat-template prefix tokens do not match full conversation"
            )

        if message["role"] == "assistant":
            for position in range(len(previous_prefix), len(prefix_ids)):
                assistant_mask[position] = 1

        previous_prefix = prefix_ids

    return tuple(assistant_mask)


def encode_sft_trajectory(
    trajectory: TeacherTrajectory,
    *,
    tokenizer: ChatTemplateTokenizer,
    max_length: int,
) -> SFTExample:
    """Tokenize one trajectory and mask every non-assistant token."""

    if max_length <= 0:
        raise ValueError("max_length must be positive")

    if trajectory.source_revision is None:
        raise ValueError(
            "SFT export requires a trajectory source revision"
        )

    messages = _chat_messages(trajectory)
    encoded = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=False,
        return_dict=True,
        return_assistant_tokens_mask=True,
        truncation=True,
        max_length=max_length,
    )

    if not isinstance(encoded, dict):
        try:
            encoded = dict(encoded)
        except (TypeError, ValueError) as error:
            raise ValueError(
                "Tokenizer must return a mapping when return_dict is enabled"
            ) from error

    input_ids = _token_vector(
        encoded.get("input_ids"),
        field_name="input_ids",
    )
    attention_mask = _token_vector(
        encoded.get("attention_mask"),
        field_name="attention_mask",
    )

    assistant_mask_value = encoded.get("assistant_masks")
    if assistant_mask_value is None:
        assistant_mask_value = encoded.get("assistant_tokens_mask")

    if assistant_mask_value is None:
        assistant_mask = _derive_assistant_mask_from_prefixes(
            messages,
            tokenizer = tokenizer,
            input_ids= input_ids,
            max_length= max_length,
        )

    else:
        assistant_mask = _token_vector(
            assistant_mask_value,
            field_name= "assistant mask"
        )
        if not any(assistant_mask):
            assistant_mask = _derive_assistant_mask_from_prefixes(
                messages,
                tokenizer= tokenizer,
                input_ids= input_ids,
                max_length= max_length,
            )

    if not (
        len(input_ids)
        == len(attention_mask)
        == len(assistant_mask)
    ):
        raise ValueError(
            "Tokenizer output vectors must have equal lengths"
        )

    labels = tuple(
        token_id if is_assistant else -100
        for token_id, is_assistant in zip(
            input_ids,
            assistant_mask,
            strict=True,
        )
    )

    if all(label == -100 for label in labels):
        raise ValueError(
            "Chat template produced no trainable assistant tokens"
        )

    return SFTExample(
        source_trajectory_id=trajectory.trajectory_id,
        source_revision=trajectory.source_revision,
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
    )


def _write_examples(
    path: Path,
    examples: Sequence[SFTExample],
) -> SFTArtifact:
    digest = sha256()
    byte_count = 0

    with path.open("xb") as stream:
        for example in examples:
            encoded = example.model_dump_json().encode("utf-8") + b"\n"
            stream.write(encoded)
            digest.update(encoded)
            byte_count += len(encoded)

    return SFTArtifact(
        filename=cast(
            Literal["train.jsonl", "validation.jsonl"],
            path.name,
        ),
        record_count=len(examples),
        byte_count=byte_count,
        sha256=digest.hexdigest(),
    )


def load_sft_examples(path: Path) -> tuple[SFTExample, ...]:
    """Load and validate one exported SFT JSONL split."""

    return load_jsonl(Path(path), SFTExample)


def export_sft_dataset(
    publication_dir: Path,
    *,
    output_root: Path,
    export_id: str,
    tokenizer: ChatTemplateTokenizer,
    tokenizer_id: str,
    max_length: int,
) -> SFTExportManifest:
    """Export one verified Teacher publication without overwriting."""

    validated_export_id = _validate_export_id(export_id)

    if not tokenizer_id.strip():
        raise ValueError("tokenizer_id must not be blank")

    if max_length <= 0:
        raise ValueError("max_length must be positive")

    publication_dir = Path(publication_dir)
    verified = load_teacher_dataset_publication(publication_dir)
    publication_manifest = publication_dir / "manifest.json"
    source_dataset_digest = sha256(
        publication_manifest.read_bytes()
    ).hexdigest()

    train_examples = tuple(
        encode_sft_trajectory(
            trajectory,
            tokenizer=tokenizer,
            max_length=max_length,
        )
        for trajectory in verified.dataset.train
    )
    validation_examples = tuple(
        encode_sft_trajectory(
            trajectory,
            tokenizer=tokenizer,
            max_length=max_length,
        )
        for trajectory in verified.dataset.validation
    )

    output_root = Path(output_root)
    export_dir = output_root / validated_export_id

    if export_dir.exists():
        raise FileExistsError(
            f"SFT export {validated_export_id!r} already exists"
        )

    output_root.mkdir(parents=True, exist_ok=True)
    staging_dir = Path(
        mkdtemp(
            prefix=f".{validated_export_id}.",
            dir=output_root,
        )
    )

    try:
        artifacts = (
            _write_examples(
                staging_dir / "train.jsonl",
                train_examples,
            ),
            _write_examples(
                staging_dir / "validation.jsonl",
                validation_examples,
            ),
        )
        manifest = SFTExportManifest(
            export_id=validated_export_id,
            source_dataset_id=verified.manifest.dataset_id,
            source_dataset_digest=source_dataset_digest,
            tokenizer_id=tokenizer_id,
            max_length=max_length,
            artifacts=artifacts,
        )
        manifest_bytes = (
            manifest.model_dump_json(indent=2).encode("utf-8")
            + b"\n"
        )
        (staging_dir / "manifest.json").write_bytes(manifest_bytes)
        staging_dir.rename(export_dir)
        return manifest
    finally:
        if staging_dir.exists():
            shutil.rmtree(staging_dir)