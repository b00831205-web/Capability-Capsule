"""Opt-in executor/termination fixes; the hash-pinned historical evaluator stays intact.

This is an evaluator implementation version, not the existing fixed-v2 task suite.
New runs must pin this module as well as its legacy dependency and use a new identity.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from time import monotonic
from typing import Any, Literal, cast

from capability_capsule.eval import coding_checkpoint as legacy

IMPLEMENTATION_ID: Literal["coding-checkpoint-literal-safe-termination-v2"] = (
    "coding-checkpoint-literal-safe-termination-v2"
)
FinishReason = Literal["eos", "length", "unknown"]
_TOOL_MARKUP = (
    "<tool_call>",
    "</tool_call>",
    "<function=",
    "</function>",
    "<parameter=",
    "</parameter>",
)


class GeneratedTurn(legacy.GeneratedTurn):
    finish_reason: FinishReason = "unknown"


def generation_finish_reason(
    token_ids: Sequence[int],
    eos_token_ids: int | Sequence[int] | None,
    max_new_tokens: int,
) -> FinishReason:
    """Check actual generated tokens, not EOS-looking text or length alone."""
    if max_new_tokens < 1:
        raise ValueError("max_new_tokens must be positive")
    eos = {eos_token_ids} if isinstance(eos_token_ids, int) else set(eos_token_ids or ())
    if token_ids and token_ids[-1] in eos:
        return "eos"  # Natural EOS on the last budgeted token is still natural.
    if len(token_ids) >= max_new_tokens:
        return "length"
    return "unknown"


class TransformersPeftTurnGenerator(legacy.TransformersPeftTurnGenerator):
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
        if not isinstance(encoded, Mapping):
            raise TypeError("Tokenizer must return a tensor mapping")
        tensors = cast(Mapping[str, Any], encoded)
        input_ids = tensors["input_ids"]
        eos = self._tokenizer.eos_token_id
        with torch.inference_mode():
            generated = self._model.generate(
                input_ids=input_ids,
                attention_mask=tensors["attention_mask"],
                max_new_tokens=self._max_new_tokens,
                do_sample=False,
                pad_token_id=self._tokenizer.pad_token_id,
                eos_token_id=eos,
            )
        completion_ids = generated[0, input_ids.shape[-1] :]
        completion = self._tokenizer.decode(completion_ids, skip_special_tokens=False)
        if not isinstance(completion, str):
            raise TypeError("Tokenizer must decode a single completion string")
        return GeneratedTurn(
            completion=completion,
            input_tokens=int(input_ids.shape[-1]),
            output_tokens=int(completion_ids.shape[-1]),
            peak_rss_mb=legacy._peak_process_rss_mb(),
            finish_reason=generation_finish_reason(
                completion_ids.tolist(), eos, self._max_new_tokens
            ),
        )


# Only literal strings: single quotes support doubled apostrophes. Double quotes
# cannot contain interpolation/escapes. No generated shell code is ever executed.
_LITERAL = r"(?:'(?:[^']|'')*'|\"[^\"$`]*\")"
_ARGS = r"\.Replace\(\s*(?P<old>" + _LITERAL + r")\s*,\s*(?P<new>" + _LITERAL + r")\s*\)"
_GUARDED = re.compile(
    r"\$(?P<source>[A-Za-z][A-Za-z0-9_]*)\s*=\s*Get-Content\s+(?:-Raw\s+)?"
    r"(?:(?:-LiteralPath|-Path)\s+)?greeting\.py(?:\s+-Raw)?\s*;\s*"
    r"\$(?P<target>[A-Za-z][A-Za-z0-9_]*)\s*=\s*\$(?P=source)"
    + _ARGS
    + r"\s*;\s*Set-Content\s+(?:(?:-LiteralPath|-Path)\s+)?greeting\.py\s+-Value\s+\$(?P=target)",
    re.IGNORECASE | re.DOTALL,
)
_PIPE = re.compile(
    r"\(Get-Content\s+(?:-Raw\s+)?(?:(?:-LiteralPath|-Path)\s+)?greeting\.py(?:\s+-Raw)?\)"
    + _ARGS
    + r"\s*\|\s*Set-Content\s+(?:(?:-LiteralPath|-Path)\s+)?greeting\.py",
    re.IGNORECASE | re.DOTALL,
)


def _literal(value: str) -> str:
    return value[1:-1].replace("''", "'") if value.startswith("'") else value[1:-1]


def _confined_file(workspace: Path, name: str) -> Path:
    if name not in {"greeting.py", "test_greeting.py"}:
        raise ValueError("File is outside the authorized subset")
    path = workspace / name
    resolved = path.resolve(strict=True)
    if resolved != path or not resolved.is_file() or resolved.stat().st_nlink != 1:
        raise ValueError("Linked or non-regular fixture files are not authorized")
    return path


class ConstrainedPowerShellExecutor(legacy.ConstrainedPowerShellExecutor):
    """Interpret a fixed command grammar; punctuation inside data is not a path."""

    def execute(self, command: str, workspace: Path) -> legacy.ToolExecutionResult:
        started = monotonic()
        command = command.strip()
        workspace = Path(workspace).resolve(strict=True)
        read = legacy._READ.fullmatch(command)
        replacement = _GUARDED.fullmatch(command) or _PIPE.fullmatch(command)
        pytest = legacy._PYTEST.fullmatch(command)
        if not (read or replacement or pytest):
            return self._denied(started, "Command is outside the authorized evaluation subset")
        names = (
            (read.group(1) or read.group(2),)
            if read
            else ("greeting.py",)
            if replacement
            else ("greeting.py", "test_greeting.py")
        )
        try:
            for name in names:
                _confined_file(workspace, name)
        except ValueError as error:
            return self._denied(started, str(error))
        except OSError as error:
            return legacy.ToolExecutionResult(
                exit_code=1,
                output=f"{type(error).__name__}: {error}",
                duration_ms=(monotonic() - started) * 1000,
            )
        if not replacement:
            return super().execute(command, workspace)
        path = workspace / "greeting.py"
        old, new = _literal(replacement.group("old")), _literal(replacement.group("new"))
        if not old:
            return self._denied(started, "Empty replacement source is not authorized")
        content = path.read_text(encoding="utf-8")
        if old not in content:
            return legacy.ToolExecutionResult(
                exit_code=1,
                output="Replacement source text was not found",
                duration_ms=(monotonic() - started) * 1000,
            )
        path.write_text(content.replace(old, new), encoding="utf-8", newline="")
        return legacy.ToolExecutionResult(exit_code=0, duration_ms=(monotonic() - started) * 1000)


class GenerationLengthExceeded(ValueError):
    pass


class MalformedToolCall(ValueError):
    pass


class UnknownGenerationFinish(ValueError):
    pass


class CodingEvaluationOutcome(legacy.CodingEvaluationOutcome):
    implementation_id: Literal["coding-checkpoint-literal-safe-termination-v2"] = IMPLEMENTATION_ID
    generations: tuple[GeneratedTurn, ...]
    format_errors: tuple[str | None, ...]
    termination_reason: Literal[
        "final_response",
        "generation_length",
        "malformed_tool_call",
        "tool_budget",
        "unknown_finish",
        "error",
    ]


def evaluate_coding_case(
    case: legacy.CodingEvaluationCase,
    *,
    generator: legacy.TurnGenerator,
    executor: legacy.ToolExecutor | None = None,
    **kwargs: Any,
) -> CodingEvaluationOutcome:
    """Reuse grading/budgets, but never execute or accept a truncated generation."""
    generations: list[GeneratedTurn] = []
    format_errors: list[str | None] = []

    class ObservedGenerator:
        def generate(
            self,
            messages: Sequence[dict[str, Any]],
            tools: Sequence[dict[str, Any]],
        ) -> GeneratedTurn:
            turn = GeneratedTurn.model_validate(generator.generate(messages, tools).model_dump())
            generations.append(turn)
            error = None
            try:
                parsed = legacy.parse_qwen_tool_completion(turn.completion)
                if any(marker in parsed.content for marker in _TOOL_MARKUP):
                    raise ValueError("Unmatched tool markup outside the parsed call")
            except ValueError as exc:
                error = str(exc)
            format_errors.append(error)
            if turn.finish_reason == "length":
                raise GenerationLengthExceeded("Generation reached its token limit")
            if turn.finish_reason == "unknown":
                raise UnknownGenerationFinish("Natural EOS was not established")
            if error is not None:
                if any(marker in turn.completion for marker in _TOOL_MARKUP):
                    raise MalformedToolCall(error)
                raise ValueError(error)
            return turn

    outcome = legacy.evaluate_coding_case(
        case,
        generator=ObservedGenerator(),
        executor=executor if executor is not None else ConstrainedPowerShellExecutor(),
        **kwargs,
    )
    reason = {
        "GenerationLengthExceeded": "generation_length",
        "MalformedToolCall": "malformed_tool_call",
        "ToolRoundLimitExceeded": "tool_budget",
        "UnknownGenerationFinish": "unknown_finish",
    }.get(
        outcome.result.error_type or "",
        "final_response" if outcome.result.status.value == "completed" else "error",
    )
    # The legacy loop catches generator errors before counting that turn. Retain
    # its original completion and token telemetry even when no command was run.
    result = outcome.result.model_copy(
        update={
            "input_tokens": sum(turn.input_tokens for turn in generations),
            "output_tokens": sum(turn.output_tokens for turn in generations),
            "peak_rss_mb": max((turn.peak_rss_mb for turn in generations), default=0),
        }
    )
    return CodingEvaluationOutcome.model_validate(
        {
            **outcome.model_dump(),
            "result": result,
            "completions": tuple(turn.completion for turn in generations),
            "generations": tuple(generations),
            "format_errors": tuple(format_errors),
            "termination_reason": reason,
        }
    )
