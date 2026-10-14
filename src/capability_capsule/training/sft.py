"""Immutable Qwen-compatible SFT export with assistant-only loss labels."""

from __future__ import annotations

import shutil
import json
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


class  SFTChatContract(BaseModel):
    """Exact Student-visible system prompt and tool contract"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["0.1"] = "0.1"
    contract_id: str = Field(min_length=1)
    system_prompt: str = Field(min_length=1)
    tools: tuple[dict[str, Any], ...] = Field(min_length=1)
    enable_thinking: bool = False

    @field_validator("contract_id", "system_prompt")
    @classmethod
    def reject_blank_content_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("SFT chat-contract fields must not be blank")
        return value

    @field_validator("tools")
    @classmethod
    def validate_tools(cls, value: tuple[dict[str, Any], ...]) -> tuple[dict[str, Any], ...]:
        names: list[str] = []

        for tool in value:
            if tool.get("type") != "function":
                raise ValueError("SFT tools must use function definitions")

            function = tool.get("function")
            if not isinstance(function, dict):
                raise ValueError("SFT function tools must contain function metadata")

            name = function.get("name")
            if not isinstance(name, str) or not name.strip():
                raise ValueError("SFT function tools must have a non-blank name")

            parameters = function.get("parameters")
            if not isinstance(parameters, dict):
                raise ValueError(
                    f"SFT too {name!r} must contain a parameter schema"
                )

            if parameters.get("type") != "object":
                raise ValueError(f"SFT tool {name!r} parameters must describe an object")

            properties = parameters.get("properties")
            if not isinstance(properties, dict):
                raise ValueError(f"SFT tool {name!r} parameters must contain properties")

            required = parameters.get("required", [])
            if (not isinstance(required, list) or not all(isinstance(item, str) for item in required) or len(required) != len(set(required))):
                raise ValueError(f"SFT tool {name!r} required fields must be unique strings")

            if set(required) - set(properties):
                raise ValueError(f"SFT tool {name!r} requires undeclared properties")

            if parameters.get("additionalProperties") is not False:
                raise ValueError(f"SFT tool {name!r} must reject additional properties")

            names.append(name)

        if len(names) != len(set(names)):
            raise ValueError("SFT tool name must be unique")

        return value

    def sha256(self) -> str:
        payload = json.dumps(
            self.model_dump(mode = "json"),
            ensure_ascii= False,
            sort_keys= True,
            separators= (",", ":")
        ).encode("utf-8")
        return sha256(payload).hexdigest()

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

    schema_version: Literal["0.1", "0.2", "0.3"] = "0.1"
    export_id: str = Field(min_length=1)
    source_dataset_id: str = Field(min_length=1)
    source_dataset_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    tokenizer_id: str = Field(min_length=1)
    max_length: int = Field(gt=0)
    chat_contract: SFTChatContract | None = None
    chat_contract_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    assistant_turn_policy: Literal["coalesce_adjacent_assistant_messages"] | None = None
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

    @model_validator(mode="after")
    def validate_chat_contract(self) -> Self:
        if self.schema_version == "0.1":
            if self.chat_contract is not None or self.chat_contract_sha256 is not None or self.assistant_turn_policy is not None:
                raise ValueError("SFT manifest schema 0.1 cannot contain a chat contract or assistant-turn policy")

            return self

        if self.chat_contract is None:
            raise ValueError(f"SFT manifest schema {self.schema_version} requires a chat contract")

        if self.chat_contract_sha256 is None:
            raise ValueError(f"SFT manifest schema {self.schema_version} requires a chat-contract digest")

        actual_digest = self.chat_contract.sha256()
        if self.chat_contract_sha256 != actual_digest:
            raise ValueError("SFT chat-contract SHA-256 mismatch")

        if self.schema_version == "0.2":
            if self.assistant_turn_policy is not None:
                raise ValueError("SFT manifest schema 0.2 cannot contain an assistant-turn policy")

            return self

        if self.assistant_turn_policy is None:
            raise ValueError(
                "SFT manifest schema 0.3 requires an assistant-turn policy"
            )
        return self


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

def _merge_assistant_content(previous: str, current: str) -> str:
    """Combine observable text belonging to one logical assistant turn."""

    if previous and current:
        return f"{previous}\n\n{current}"

    return previous or current

def _append_normalized_chat_message(
        rendered: list[dict[str, Any]],
        item: dict[str, Any]
) -> None:
    """Coalesce adjacent assistant records into one chat-template turn."""

    if item["role"] != "assistant" or not rendered or rendered[-1]["role"] != "assistant":
        rendered.append(item)

        return

    previous = rendered[-1]
    previous["content"] = _merge_assistant_content(
        previous.get("content", ""),
        item.get("content", "")
    )
    current_tool_calls = item.get("tool_calls", [])
    if current_tool_calls:
        previous.setdefault("tool_calls", []).extend(current_tool_calls)


def _chat_messages(
    trajectory: TeacherTrajectory,
    *,
    chat_contract: SFTChatContract
) -> list[dict[str, Any]]:
    rendered: list[dict[str, Any]] = [
        {
            "role": "system",
            "content": chat_contract.system_prompt,
        }
    ]

    for message in trajectory.messages:
        if message.role.value == "system":
            raise ValueError("Teacher trajectories must not supply their own system message")
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

        _append_normalized_chat_message(rendered, item)

    return rendered

def _matches_json_type(value: object, expected_type: str) -> bool:
    if expected_type == "string":
        return isinstance(value, str)
    if expected_type == "boolean":
        return isinstance(value, bool)
    if expected_type == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected_type == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if expected_type == "object":
        return isinstance(value, dict)
    if expected_type == "array":
        return isinstance(value, list)
    if expected_type == "null":
        return value is None
    raise ValueError(f"Unsupported SFT tool JSON type {expected_type!r}")

def _tool_contracts(
    chat_contract: SFTChatContract,
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}

    for tool in chat_contract.tools:
        function = tool["function"]
        result[function["name"]] = function

    return result

def _validate_trajectory_tool_contract(
        trajectory: TeacherTrajectory,
        *,
        chat_contract: SFTChatContract,
) -> None:
    tools = _tool_contracts(chat_contract)

    for message in trajectory.messages:
        for tool_call in message.tool_calls:
            function = tools.get(tool_call.name)
            if function is None:
                raise ValueError(
                    f"Trajectory {trajectory.trajectory_id!r} calls under tool {tool_call.name!r}"
                )

            parameters = function["parameters"]
            properties = parameters["properties"]
            required = set(parameters.get("required", []))
            arguments = tool_call.arguments

            missing = required - set(arguments)
            if missing:
                raise ValueError(
                    f"Trajectory {trajectory.trajectory_id!r} tool {tool_call.name!r} is missing arguments: {", ".join(sorted(missing))}"
                )

            if parameters.get("additionalProperties") is False:
                unexpected = set(arguments) - set(properties)
                if unexpected:
                    raise ValueError(
                        f"Trajectory {trajectory.trajectory_id!r} tool {tool_call.name!r} contains arguments outside the SFT contract: {", ".join(sorted(unexpected))}"
                    )

            for argument_name, argument_value in arguments.items():
                property_schema = properties.get(argument_name)
                if not isinstance(property_schema, dict):
                    continue

                expected_type = property_schema.get("type")
                if expected_type is None:
                    continue

                if not isinstance(expected_type, str):
                    raise ValueError(
                        f"SFT tool {tool_call.name!r} argument schema for {argument_name!r} must contain one JSON type"
                    )

                if not _matches_json_type(argument_value, expected_type):
                    raise ValueError(
                        f"Trajectory {trajectory.trajectory_id!r} tool {tool_call.name!r} argument {argument_name!r} must have JSON type {expected_type!r}"
                    )

        if message.tool_name is not None and message.tool_name not in tools:
            raise ValueError(
                f"Trajectory {trajectory.trajectory_id!r} contains a result for undeclared tool {message.tool_name!r}"
            )

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
        chat_contract: SFTChatContract,
        input_ids: tuple[int, ...],
        max_length: int,
) -> tuple[int, ...]:
    """Derive assistant spans when the chat template has no generation blocks."""

    assistant_mask = [0] * len(input_ids)
    previous_prefix: tuple[int, ...] = ()

    for end, message in enumerate(messages, start=1):
        prefix_messages = messages[:end]
        if not any(item["role"] == "user" for item in prefix_messages):
            continue

        rendered = tokenizer.apply_chat_template(
            prefix_messages,
            tools = list(chat_contract.tools),
            tokenize = True,
            add_generation_prompt = False,
            enable_thinking = chat_contract.enable_thinking,
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
    chat_contract: SFTChatContract,
    max_length: int,
) -> SFTExample:
    """Tokenize one trajectory and mask every non-assistant token."""

    if max_length <= 0:
        raise ValueError("max_length must be positive")

    if trajectory.source_revision is None:
        raise ValueError(
            "SFT export requires a trajectory source revision"
        )

    _validate_trajectory_tool_contract(trajectory, chat_contract= chat_contract)
    messages = _chat_messages(trajectory, chat_contract= chat_contract)

    encoded = tokenizer.apply_chat_template(
        messages,
        tools = list(chat_contract.tools),
        tokenize=True,
        add_generation_prompt=False,
        enable_thinking = chat_contract.enable_thinking,
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
            chat_contract= chat_contract,
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
                chat_contract= chat_contract,
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
    chat_contract: SFTChatContract,
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
            chat_contract= chat_contract,
            max_length=max_length,
        )
        for trajectory in verified.dataset.train
    )
    validation_examples = tuple(
        encode_sft_trajectory(
            trajectory,
            tokenizer=tokenizer,
            chat_contract= chat_contract,
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
            schema_version= "0.3",
            export_id=validated_export_id,
            source_dataset_id=verified.manifest.dataset_id,
            source_dataset_digest=source_dataset_digest,
            tokenizer_id=tokenizer_id,
            max_length=max_length,
            chat_contract= chat_contract,
            chat_contract_sha256= chat_contract.sha256(),
            assistant_turn_policy=("coalesce_adjacent_assistant_messages"),
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
