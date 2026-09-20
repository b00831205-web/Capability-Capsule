"""Bounded coding-fixture evaluation for trained checkpoints."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path, PurePosixPath
from time import monotonic
from typing import Any, Literal, Protocol, Self

from pydantic import BaseModel, ConfigDict, Field, JsonValue, field_validator, model_validator

from capability_capsule.eval.fixture_provenance import inspect_fixture, verify_task_fixture
from capability_capsule.eval.records import CaseResult, DatasetSplit, RunStatus, ToolCallRecord
from capability_capsule.eval.tasks import TaskSpec
from capability_capsule.eval.evaluation_suite import EvaluationSuiteIdentity


class CodingValidatorSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    validator_id: str = Field(min_length=1)
    kind: Literal["pytest", "python_call", "exact_text"]
    target: str = Field(min_length=1)
    function: str | None = None
    arguments: tuple[JsonValue, ...] = ()
    expected: JsonValue | None = None

    @model_validator(mode="after")
    def validate_shape(self) -> Self:
        if self.kind == "python_call" and not self.function:
            raise ValueError("python_call validator requires function")
        if self.kind != "python_call" and self.function is not None:
            raise ValueError("Only python_call validators use function")
        if self.kind == "pytest" and (self.arguments or self.expected is not None):
            raise ValueError("pytest validator does not use arguments or expected")
        if self.kind in {"python_call", "exact_text"} and self.expected is None:
            raise ValueError(f"{self.kind} validator requires expected")
        return self


class CodingEvaluationCase(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["0.1"] = "0.1"
    case_id: str = Field(min_length=1)
    fixture_root: str = Field(min_length=1)
    task: TaskSpec
    validators: tuple[CodingValidatorSpec, ...] = Field(min_length=1)

    @field_validator("fixture_root")
    @classmethod
    def require_safe_relative_fixture_root(cls, value: str) -> str:
        normalized = PurePosixPath(value)
        if (
            not value.strip()
            or value != value.strip()
            or normalized.is_absolute()
            or ".." in normalized.parts
            or "\\" in value
            or ":" in value
        ):
            raise ValueError("fixture_root must be a safe relative path")
        return value

    @model_validator(mode="after")
    def require_validation_split(self) -> Self:
        if self.task.split is not DatasetSplit.VALIDATION:
            raise ValueError("Checkpoint coding cases must use validation split")
        if self.task.allowed_tools != ("exec_command",):
            raise ValueError("Checkpoint coding cases require only exec_command")
        return self


class CodingEvaluationSuite(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["0.1"] = "0.1"
    capability_id: str = Field(min_length=1)
    evaluation_suite_id: str = Field(min_length=1)
    evaluation_split: Literal[DatasetSplit.VALIDATION] = DatasetSplit.VALIDATION
    system_prompt: str = Field(min_length=1)
    cases: tuple[CodingEvaluationCase, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def require_unique_cases(self) -> Self:
        case_ids = tuple(case.case_id for case in self.cases)
        if len(set(case_ids)) != len(case_ids):
            raise ValueError("Coding evaluation case IDs must be unique")
        return self

    def identity(self) -> EvaluationSuiteIdentity:
        payload = [case.model_dump(mode="json", exclude_defaults=True) for case in self.cases]
        encoded = json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return EvaluationSuiteIdentity(
            capability_id=self.capability_id,
            evaluation_suite_id=self.evaluation_suite_id,
            evaluation_split=DatasetSplit.VALIDATION,
            case_count=len(self.cases),
            evaluation_suite_digest=sha256(encoded).hexdigest(),
        )


def load_coding_evaluation_suite(path: Path) -> CodingEvaluationSuite:
    return CodingEvaluationSuite.model_validate_json(Path(path).read_bytes())


def copy_verified_evaluation_fixture(
    case: CodingEvaluationCase,
    *,
    artifact_root: Path,
    destination: Path,
) -> None:
    source = (Path(artifact_root) / case.fixture_root).resolve(strict=True)
    verify_task_fixture(case.task, source)
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, destination, ignore=shutil.ignore_patterns("__pycache__", ".pytest_cache", "*.pyc"))
    copied = inspect_fixture(case.task.fixture_id, destination)
    if copied.revision != case.task.fixture_revision:
        raise ValueError("Copied evaluation fixture revision mismatch")


class GeneratedTurn(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    completion: str
    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    peak_rss_mb: float = Field(ge=0)


class ParsedTurn(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    content: str = ""
    tool_calls: tuple[ToolCallRecord, ...] = ()

    @model_validator(mode="after")
    def require_content_or_tool(self) -> Self:
        if not self.content.strip() and not self.tool_calls:
            raise ValueError("Assistant completion is empty")
        return self


_TOOL_CALL = re.compile(
    r"<tool_call>\s*<function=([^>\s]+)>\s*"
    r"<parameter=cmd>\s*(.*?)\s*</parameter>\s*"
    r"</function>\s*</tool_call>",
    re.DOTALL,
)


def parse_qwen_tool_completion(completion: str) -> ParsedTurn:
    matches = tuple(_TOOL_CALL.finditer(completion))
    if not matches:
        if "<tool_call>" in completion or "<function=" in completion:
            raise ValueError("Malformed Qwen tool call")
        content = completion.replace("<|im_end|>", "").strip()
        return ParsedTurn(content=content)
    if len(matches) != 1:
        raise ValueError("Parallel tool calls are not supported")
    match = matches[0]
    remainder = (completion[: match.start()] + completion[match.end() :])
    remainder = remainder.replace("<|im_end|>", "").strip()
    command = match.group(2).strip()
    if not command:
        raise ValueError("exec_command cmd must not be blank")
    return ParsedTurn(
        content=remainder,
        tool_calls=(ToolCallRecord(name=match.group(1), arguments={"cmd": command}),),
    )


class ToolExecutionResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    exit_code: int
    output: str = ""
    authorized: bool = True
    duration_ms: float = Field(ge=0)

    def envelope(self) -> str:
        return json.dumps(
            {
                "envelope": "codex-exec-command-v1",
                "exit_code": self.exit_code,
                "output": self.output,
            },
            ensure_ascii=False,
            separators=(",", ":"),
        )


class ToolExecutor(Protocol):
    def execute(self, command: str, workspace: Path) -> ToolExecutionResult: ...


class TurnGenerator(Protocol):
    def generate(
        self,
        messages: Sequence[dict[str, Any]],
        tools: Sequence[dict[str, Any]],
    ) -> GeneratedTurn: ...


def _peak_process_rss_mb() -> float:
    try:
        import resource

        value = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        return value / 1024.0
    except (ImportError, AttributeError):
        return 0.0


class TransformersPeftTurnGenerator:
    """Generate deterministic turns from one hash-verified PEFT checkpoint."""

    def __init__(self, checkpoint_path: Path, *, max_new_tokens: int = 256) -> None:
        if max_new_tokens < 1:
            raise ValueError("max_new_tokens must be positive")
        from transformers import AutoTokenizer

        from capability_capsule.training.lora import (
            SavedAdapterCheckpoint,
            reload_lora_adapter,
        )
        from capability_capsule.training.transformers_peft import (
            TransformersPeftAdapterLoader,
        )

        checkpoint_path = Path(checkpoint_path).resolve(strict=True)
        checkpoint = SavedAdapterCheckpoint.model_validate_json(
            checkpoint_path.read_bytes()
        )
        self._model = reload_lora_adapter(
            checkpoint_path,
            loader=TransformersPeftAdapterLoader(),
        )
        self._tokenizer = AutoTokenizer.from_pretrained(
            checkpoint.base_model_id,
            revision=checkpoint.base_model_revision,
            local_files_only=True,
        )
        self._max_new_tokens = max_new_tokens

    def generate(
        self,
        messages: Sequence[dict[str, Any]],
        tools: Sequence[dict[str, Any]],
    ) -> GeneratedTurn:
        import torch

        encoded = self._tokenizer.apply_chat_template(
            list(messages),
            tools=list(tools),
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
            enable_thinking=False,
        )
        input_ids = encoded["input_ids"]
        attention_mask = encoded["attention_mask"]
        with torch.inference_mode():
            generated = self._model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=self._max_new_tokens,
                do_sample=False,
                pad_token_id=self._tokenizer.pad_token_id,
                eos_token_id=self._tokenizer.eos_token_id,
            )
        completion_ids = generated[0, input_ids.shape[-1] :]
        completion = self._tokenizer.decode(
            completion_ids,
            skip_special_tokens=False,
        )
        return GeneratedTurn(
            completion=completion,
            input_tokens=int(input_ids.shape[-1]),
            output_tokens=int(completion_ids.shape[-1]),
            peak_rss_mb=_peak_process_rss_mb(),
        )


_READ = re.compile(
    r"^(?:cat|type)\s+['\"]?(greeting\.py|test_greeting\.py)['\"]?$|"
    r"^Get-Content\s+(?:(?:-LiteralPath|-Path)\s+)?['\"]?"
    r"(greeting\.py|test_greeting\.py)['\"]?(?:\s+-Raw)?$",
    re.IGNORECASE,
)
_REPLACE = re.compile(
    r"\.Replace\(\s*(['\"])(.*?)\1\s*,\s*(['\"])(.*?)\3\s*\)",
    re.DOTALL,
)
_SET_CONTENT = re.compile(
    r"Set-Content\s+(?:(?:-LiteralPath|-Path)\s+)?['\"]?greeting\.py['\"]?",
    re.IGNORECASE,
)
_PYTEST = re.compile(
    r"^(?:python(?:\.exe)?\s+-m\s+pytest|pytest)\s+-q(?:\s+test_greeting\.py)?$",
    re.IGNORECASE,
)


class ConstrainedPowerShellExecutor:
    """Execute only the small, path-confined command subset used by this suite."""

    def execute(self, command: str, workspace: Path) -> ToolExecutionResult:
        started = monotonic()
        command = command.strip()
        workspace = Path(workspace).resolve(strict=True)
        read = _READ.fullmatch(command)
        if read:
            relative = read.group(1) or read.group(2)
            path = workspace / relative
            try:
                output = path.read_text(encoding="utf-8")
                exit_code = 0
            except OSError as error:
                output = f"{type(error).__name__}: {error}"
                exit_code = 1
            return ToolExecutionResult(
                exit_code=exit_code,
                output=output,
                duration_ms=(monotonic() - started) * 1000,
            )

        replacement = _REPLACE.search(command)
        if replacement and _SET_CONTENT.search(command):
            if any(token in command for token in ("..", ":", "/", "\\")):
                return self._denied(started, "Path escape syntax is not authorized")
            old, new = replacement.group(2), replacement.group(4)
            path = workspace / "greeting.py"
            content = path.read_text(encoding="utf-8")
            if old not in content:
                return ToolExecutionResult(
                    exit_code=1,
                    output="Replacement source text was not found",
                    duration_ms=(monotonic() - started) * 1000,
                )
            path.write_text(content.replace(old, new), encoding="utf-8", newline="")
            return ToolExecutionResult(
                exit_code=0,
                output="",
                duration_ms=(monotonic() - started) * 1000,
            )

        if _PYTEST.fullmatch(command):
            completed = subprocess.run(
                [sys.executable, "-m", "pytest", "-q", "test_greeting.py"],
                cwd=workspace,
                capture_output=True,
                text=True,
                timeout=60,
                check=False,
            )
            return ToolExecutionResult(
                exit_code=completed.returncode,
                output=(completed.stdout + completed.stderr)[-8000:],
                duration_ms=(monotonic() - started) * 1000,
            )

        return self._denied(started, "Command is outside the authorized evaluation subset")

    @staticmethod
    def _denied(started: float, message: str) -> ToolExecutionResult:
        return ToolExecutionResult(
            exit_code=126,
            output=message,
            authorized=False,
            duration_ms=(monotonic() - started) * 1000,
        )


class ValidatorOutcome(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    validator_id: str
    passed: bool
    detail: str


def run_coding_validators(
    case: CodingEvaluationCase,
    workspace: Path,
) -> tuple[ValidatorOutcome, ...]:
    workspace = Path(workspace).resolve(strict=True)
    outcomes: list[ValidatorOutcome] = []
    for validator in case.validators:
        target = (workspace / validator.target).resolve(strict=True)
        try:
            target.relative_to(workspace)
        except ValueError as error:
            raise ValueError("Validator target escapes workspace") from error

        if validator.kind == "pytest":
            completed = subprocess.run(
                [sys.executable, "-m", "pytest", "-q", validator.target],
                cwd=workspace,
                capture_output=True,
                text=True,
                timeout=60,
                check=False,
            )
            outcomes.append(
                ValidatorOutcome(
                    validator_id=validator.validator_id,
                    passed=completed.returncode == 0,
                    detail=(completed.stdout + completed.stderr)[-8000:],
                )
            )
            continue

        if validator.kind == "exact_text":
            actual = target.read_text(encoding="utf-8")
            passed = actual == validator.expected
            outcomes.append(
                ValidatorOutcome(
                    validator_id=validator.validator_id,
                    passed=passed,
                    detail="exact text matched" if passed else "exact text mismatch",
                )
            )
            continue

        namespace: dict[str, Any] = {}
        exec(compile(target.read_bytes(), str(target), "exec"), namespace)
        value = namespace[validator.function](*validator.arguments)  # type: ignore[index]
        passed = value == validator.expected
        outcomes.append(
            ValidatorOutcome(
                validator_id=validator.validator_id,
                passed=passed,
                detail=f"returned {value!r}",
            )
        )
    return tuple(outcomes)


class CodingEvaluationOutcome(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    result: CaseResult
    completions: tuple[str, ...]
    tool_results: tuple[ToolExecutionResult, ...]
    validators: tuple[ValidatorOutcome, ...]


def evaluate_coding_case(
    case: CodingEvaluationCase,
    *,
    workspace: Path,
    generator: TurnGenerator,
    executor: ToolExecutor,
    system_prompt: str,
    tools: Sequence[dict[str, Any]],
    experiment_id: str,
    run_id: str,
    model_id: str,
    hardware_id: str,
    cycle_position: int,
    max_tool_rounds: int = 6,
    now: Callable[[], datetime] = lambda: datetime.now(UTC),
) -> CodingEvaluationOutcome:
    if max_tool_rounds < 1:
        raise ValueError("max_tool_rounds must be positive")
    started_at = now()
    started = monotonic()
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": case.task.task},
    ]
    completions: list[str] = []
    tool_results: list[ToolExecutionResult] = []
    input_tokens = 0
    output_tokens = 0
    peak_rss_mb = 0.0
    invalid_tool_calls = 0
    status = RunStatus.FAILED
    error_type: str | None = None
    reached_final = False

    try:
        for _ in range(max_tool_rounds + 1):
            generated = generator.generate(messages, tools)
            input_tokens += generated.input_tokens
            output_tokens += generated.output_tokens
            peak_rss_mb = max(peak_rss_mb, generated.peak_rss_mb)
            completions.append(generated.completion)
            parsed = parse_qwen_tool_completion(generated.completion)
            if not parsed.tool_calls:
                reached_final = True
                status = RunStatus.COMPLETED
                break
            if len(tool_results) >= max_tool_rounds:
                error_type = "ToolRoundLimitExceeded"
                break
            tool_call = parsed.tool_calls[0]
            arguments = tool_call.arguments
            command = arguments.get("cmd")
            if tool_call.name != "exec_command" or not isinstance(command, str):
                invalid_tool_calls += 1
                execution = ToolExecutionResult(
                    exit_code=126,
                    output="Invalid exec_command request",
                    authorized=False,
                    duration_ms=0,
                )
            else:
                execution = executor.execute(command, workspace)
                if not execution.authorized:
                    invalid_tool_calls += 1
            tool_results.append(execution)
            messages.append(
                {
                    "role": "assistant",
                    "content": parsed.content,
                    "tool_calls": [
                        {
                            "type": "function",
                            "function": {
                                "name": tool_call.name,
                                "arguments": dict(arguments),
                            },
                        }
                    ],
                }
            )
            messages.append(
                {
                    "role": "tool",
                    "name": tool_call.name,
                    "content": execution.envelope(),
                }
            )
    except Exception as error:
        error_type = type(error).__name__

    validators = run_coding_validators(case, workspace)
    passed = all(outcome.passed for outcome in validators)
    success = reached_final and passed and invalid_tool_calls == 0
    duration_ms = (monotonic() - started) * 1000
    result = CaseResult(
        experiment_id=experiment_id,
        run_id=run_id,
        case_id=case.case_id,
        model_id=model_id,
        condition_id="trained",
        hardware_id=hardware_id,
        repetition=1,
        cycle_position=cycle_position,
        status=status,
        success=success,
        time_bounded_success=(
            success and duration_ms <= case.task.time_limit.total_seconds() * 1000
        ),
        score=(1.0 if passed else 0.0) if status is RunStatus.COMPLETED else None,
        started_at=started_at,
        duration_ms=duration_ms,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        peak_rss_mb=peak_rss_mb,
        tool_call_count=len(tool_results),
        invalid_tool_call_count=invalid_tool_calls,
        error_type=error_type,
    )
    return CodingEvaluationOutcome(
        result=result,
        completions=tuple(completions),
        tool_results=tuple(tool_results),
        validators=validators,
    )
