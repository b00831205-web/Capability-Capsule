"""Synthetic fixtures only: no model load, historical replay or run-directory writes."""

from contextlib import nullcontext
from hashlib import sha256
from types import SimpleNamespace
import sys

import pytest

from capability_capsule.eval import coding_checkpoint as legacy
from capability_capsule.eval import coding_checkpoint_v2 as fixed
from test_coding_checkpoint import make_case, write_fixture


def edit(old, new):
    return ("$old = Get-Content -LiteralPath greeting.py -Raw; "
        f"$new = $old.Replace('{old}', '{new}'); "
        "Set-Content -LiteralPath greeting.py -Value $new")


@pytest.mark.parametrize("text", ["Build: {name}.", "Stay / {name}!", "{count:03d}",
    r"C:\data\name", "../literal/data", "literal; Invoke-Expression nope", "$(literal)", "don''t"])
def test_punctuation_and_shell_looking_literal_data_are_not_paths(tmp_path, text):
    source = tmp_path / "greeting.py"
    source.write_text("original", encoding="utf-8")
    result = fixed.ConstrainedPowerShellExecutor().execute(edit("original", text), tmp_path)
    assert result.authorized and result.exit_code == 0
    assert source.read_text("utf-8") == text.replace("''", "'")


def test_pipe_and_non_interpolated_double_quoted_literals(tmp_path):
    source = tmp_path / "greeting.py"
    source.write_text("old", encoding="utf-8")
    command = '(Get-Content greeting.py -Raw).Replace("old", "new: / data") | Set-Content greeting.py'
    assert fixed.ConstrainedPowerShellExecutor().execute(command, tmp_path).exit_code == 0
    assert source.read_text("utf-8") == "new: / data"


@pytest.mark.parametrize("command", [
    "Get-Content ../greeting.py", "Get-Content C:\\secret.txt", "Get-Content /etc/passwd",
    "Get-Content greeting.py:stream", "Get-Content greeting.py; whoami",
    edit("old", "new").replace("greeting.py", "../greeting.py"),
    edit("old", "new").replace("greeting.py", "test_greeting.py"),
    edit("old", "new") + "; Remove-Item greeting.py",
    '$x = Get-Content greeting.py -Raw; $y = $x.Replace("old", "$env:SECRET"); Set-Content greeting.py -Value $y',
    '$x = Get-Content greeting.py -Raw; $y = $x.Replace("old", "`nnew"); Set-Content greeting.py -Value $y',
    "$x = Get-Content greeting.py -Raw; $y = $x.Replace('old', 'new' + $name); Set-Content greeting.py -Value $y",
])
def test_paths_and_extra_shell_syntax_remain_rejected(tmp_path, command):
    source = tmp_path / "greeting.py"
    source.write_text("old", encoding="utf-8")
    result = fixed.ConstrainedPowerShellExecutor().execute(command, tmp_path)
    assert not result.authorized and result.exit_code == 126
    assert source.read_text("utf-8") == "old"


@pytest.mark.parametrize("link_kind", ["symlink", "hardlink"])
def test_links_cannot_redirect_an_authorized_filename(tmp_path, link_kind):
    outside = tmp_path / "outside.py"
    outside.write_text("original", encoding="utf-8")
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    link = workspace / "greeting.py"
    if link_kind == "symlink":
        link.symlink_to(outside)
    else:
        link.hardlink_to(outside)
    for command in ("Get-Content greeting.py", edit("original", "changed")):
        result = fixed.ConstrainedPowerShellExecutor().execute(command, workspace)
        assert not result.authorized
    assert outside.read_text("utf-8") == "original"


def test_missing_old_text_and_empty_old_text_do_not_write(tmp_path):
    source = tmp_path / "greeting.py"
    source.write_text("original", encoding="utf-8")
    executor = fixed.ConstrainedPowerShellExecutor()
    result = executor.execute(edit("not found", "changed"), tmp_path)
    assert result.authorized and result.exit_code == 1
    assert executor.execute(edit("", "changed"), tmp_path).authorized is False
    assert source.read_text("utf-8") == "original"


def test_doubled_apostrophe_in_source_is_decoded_as_literal(tmp_path):
    source = tmp_path / "greeting.py"
    source.write_text("don't", encoding="utf-8")
    result = fixed.ConstrainedPowerShellExecutor().execute(edit("don''t", "do"), tmp_path)
    assert result.exit_code == 0 and source.read_text("utf-8") == "do"


@pytest.mark.parametrize("tokens,eos,limit,reason", [
    ([1, 99], 99, 2, "eos"), ([1, 99], [98, 99], 2, "eos"),
    ([1, 2], 99, 2, "length"), ([1], 99, 2, "unknown"),
    ([], None, 2, "unknown"), ([1, 99, 2], 99, 3, "length"),
])
def test_finish_reason_uses_last_generated_token(tokens, eos, limit, reason):
    assert fixed.generation_finish_reason(tokens, eos, limit) == reason


def test_zero_generation_budget_is_rejected():
    with pytest.raises(ValueError):
        fixed.generation_finish_reason([], 99, 0)


def test_real_generator_wiring_without_a_model_or_torch(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(inference_mode=nullcontext))

    class Tokens:
        shape = (2,)

        def tolist(self):
            return [7, 99]

    class Matrix:
        def __getitem__(self, key):
            assert key == (0, slice(3, None))
            return Tokens()

    def generate(**kwargs):
        assert kwargs["max_new_tokens"] == 2 and kwargs["do_sample"] is False
        assert kwargs["eos_token_id"] == 99
        return Matrix()

    def template(*args, **kwargs):
        assert kwargs["enable_thinking"] is False
        return {"input_ids": SimpleNamespace(shape=(1, 3)), "attention_mask": object()}

    generator = fixed.TransformersPeftTurnGenerator.__new__(fixed.TransformersPeftTurnGenerator)
    generator._tokenizer = SimpleNamespace(eos_token_id=99, pad_token_id=0,
        apply_chat_template=template, decode=lambda *a, **k: "Done.<|im_end|>")
    generator._model = SimpleNamespace(generate=generate)
    generator._max_new_tokens = 2
    turn = generator.generate([], [])
    assert turn.finish_reason == "eos" and turn.output_tokens == 2 and turn.input_tokens == 3


def turn(text, reason="eos", tokens=10):
    return fixed.GeneratedTurn(completion=text, input_tokens=20, output_tokens=tokens,
        peak_rss_mb=1, finish_reason=reason)


def call(command):
    return "<tool_call><function=exec_command><parameter=cmd>" + command + "</parameter></function></tool_call><|im_end|>"


def evaluate(tmp_path, turns, max_tools=4):
    fixture = tmp_path / "fixture"
    write_fixture(fixture)
    case = make_case(fixture)
    responses = iter(turns)
    generator = SimpleNamespace(generate=lambda *a: next(responses))
    return fixed.evaluate_coding_case(case, workspace=fixture, generator=generator,
        system_prompt="Use exec_command.", tools=(), experiment_id="synthetic", run_id="synthetic",
        model_id="synthetic", hardware_id="synthetic", cycle_position=1, max_tool_rounds=max_tools)


def test_natural_final_keeps_original_grading(tmp_path):
    outcome = evaluate(tmp_path, [turn(call(edit("Hello, ", "Welcome, "))), turn("Done.<|im_end|>")])
    assert outcome.result.success and outcome.termination_reason == "final_response"
    assert outcome.result.tool_call_count == 1
    assert [row.finish_reason for row in outcome.generations] == ["eos", "eos"]


@pytest.mark.parametrize("text", ["I cannot edit files " * 20, call(edit("Hello", "Welcome")),
    "<tool_call><function=exec_command><parameter=cmd>unfinished"])
def test_truncation_never_executes_or_becomes_a_final_response(tmp_path, text):
    outcome = evaluate(tmp_path, [turn(text, "length", 256)])
    assert not outcome.result.success and outcome.termination_reason == "generation_length"
    assert outcome.result.error_type == "GenerationLengthExceeded"
    assert outcome.result.tool_call_count == 0 and outcome.result.output_tokens == 256
    assert outcome.result.input_tokens == 20 and outcome.completions == (text,)
    assert "Hello" in (tmp_path / "fixture/greeting.py").read_text("utf-8")
    assert (outcome.format_errors[0] is not None) == ("unfinished" in text)


def test_unknown_finish_is_not_assumed_natural(tmp_path):
    outcome = evaluate(tmp_path, [turn("Done", "unknown")])
    assert outcome.termination_reason == "unknown_finish" and not outcome.result.success


def test_empty_eos_is_a_generic_error_not_a_malformed_tool_call(tmp_path):
    outcome = evaluate(tmp_path, [turn("   <|im_end|>")])
    assert outcome.termination_reason == "error" and not outcome.result.success
    assert outcome.result.error_type == "ValueError" and outcome.format_errors[0]


@pytest.mark.parametrize("completion", [
    "<tool_call><function=exec_command><parameter=cmd>missing close<|im_end|>",
    call("Get-Content greeting.py") + "<tool_call><function=exec_command>",
    call("Get-Content greeting.py") + call("Get-Content test_greeting.py"),
    "Done.</parameter><|im_end|>",
])
def test_malformed_complete_generations_are_distinct_from_length(tmp_path, completion):
    outcome = evaluate(tmp_path, [turn(completion)])
    assert outcome.termination_reason == "malformed_tool_call"
    assert outcome.result.error_type == "MalformedToolCall" and outcome.result.tool_call_count == 0
    assert outcome.format_errors[0]


def test_tool_budget_is_not_confused_with_generation_length(tmp_path):
    command = call("Get-Content greeting.py")
    outcome = evaluate(tmp_path, [turn(command), turn(command)], max_tools=1)
    assert outcome.termination_reason == "tool_budget"
    assert outcome.result.tool_call_count == 1 and len(outcome.generations) == 2


def test_natural_final_on_last_permitted_generation_is_allowed(tmp_path):
    outcome = evaluate(tmp_path, [turn(call(edit("Hello, ", "Welcome, "))),
        turn("Done.<|im_end|>")], max_tools=1)
    assert outcome.termination_reason == "final_response" and outcome.result.success


def test_eos_does_not_mean_task_success(tmp_path):
    outcome = evaluate(tmp_path, [turn("I cannot edit files.<|im_end|>")])
    assert outcome.termination_reason == "final_response" and not outcome.result.success


def test_legacy_generator_without_finish_metadata_is_not_silently_trusted(tmp_path):
    fixture = tmp_path / "fixture"
    write_fixture(fixture)
    generator = SimpleNamespace(generate=lambda *a: legacy.GeneratedTurn(
        completion="Done.<|im_end|>", input_tokens=1, output_tokens=1, peak_rss_mb=1))
    outcome = fixed.evaluate_coding_case(make_case(fixture), workspace=fixture,
        generator=generator, system_prompt="Use tools", tools=(), experiment_id="synthetic",
        run_id="synthetic", model_id="synthetic", hardware_id="synthetic", cycle_position=1)
    assert outcome.termination_reason == "unknown_finish" and not outcome.result.success


def test_old_evaluator_is_byte_identical_and_not_silently_switched():
    assert sha256(legacy.Path(legacy.__file__).read_bytes()).hexdigest() == "c1b3ec738da80c59da83411adc0884223b36b372041b45390ffa0f9834381356"
    assert legacy.ConstrainedPowerShellExecutor is not fixed.ConstrainedPowerShellExecutor
