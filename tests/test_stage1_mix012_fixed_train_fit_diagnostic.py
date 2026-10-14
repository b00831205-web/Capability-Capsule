"""Synthetic train probes; never load a model or write historical artifacts."""

from types import SimpleNamespace

import pytest

import run_stage1_mix012_fixed_train_fit_diagnostic as probe
from capability_capsule.eval import coding_checkpoint_v2 as fixed


def call(command):
    return "<tool_call><function=exec_command><parameter=cmd>" + command + "</parameter></function></tool_call><|im_end|>"


def run(tmp_path, monkeypatch, turns, expected="new: / {name}!", split="train"):
    (tmp_path / "greeting.py").write_text("old {name}.", encoding="utf-8")
    (tmp_path / "test_greeting.py").write_text("protected", encoding="utf-8")
    monkeypatch.setattr(probe.diagnostic, "pytest_result", lambda p: {
        "exit_code":0 if (p / "greeting.py").read_text() == expected else 1, "output":"synthetic"})
    stream = iter(turns)
    generator = SimpleNamespace(generate=lambda *a: next(stream))
    return probe.run_case({"task_id":"synthetic", "task":"edit source", "split":split},
        {"system_prompt":"test", "tools":[]}, tmp_path, generator, fixed.ConstrainedPowerShellExecutor(), expected)


def turn(text, finish="eos", tokens=10):
    return fixed.GeneratedTurn(completion=text, input_tokens=20, output_tokens=tokens,
        peak_rss_mb=1, finish_reason=finish)


def test_literal_punctuation_and_natural_final_succeed(tmp_path, monkeypatch):
    command = "$old = Get-Content greeting.py -Raw; $new = $old.Replace('old {name}.', 'new: / {name}!'); Set-Content greeting.py -Value $new"
    report = run(tmp_path, monkeypatch, [turn(call(command)), turn("Done.<|im_end|>")])
    assert report["strict_success"] and report["tests_unchanged"]
    assert report["termination_reason"] == "final_response"
    assert report["tool_call_count"] == 1


@pytest.mark.parametrize("text,malformed", [("Done",False), (call("Get-Content greeting.py"),False),
    ("<tool_call><function=exec_command><parameter=cmd>unfinished",True)])
def test_length_never_executes_or_finishes_and_keeps_format_error(tmp_path, monkeypatch, text, malformed):
    report = run(tmp_path, monkeypatch, [turn(text,"length",256)])
    assert report["termination_reason"] == "generation_length"
    assert report["truncated_turn_count"] == 1
    assert report["format_error_count"] == int(malformed)
    assert not report["reached_final"] and not report["strict_success"]
    assert report["tool_call_count"] == 0


def test_malformed_eos_is_not_truncation(tmp_path, monkeypatch):
    report = run(tmp_path, monkeypatch, [turn("<tool_call>unfinished<|im_end|>")])
    assert report["termination_reason"] == "malformed_tool_call"
    assert report["format_error_count"] == 1 and report["truncated_turn_count"] == 0


def test_natural_eos_on_budget_last_token_is_accepted(tmp_path, monkeypatch):
    report = run(tmp_path, monkeypatch, [turn("Done.<|im_end|>",tokens=256)], expected="old {name}.")
    assert report["strict_success"] and report["truncated_turn_count"] == 0


def test_unknown_finish_is_not_final(tmp_path, monkeypatch):
    report = run(tmp_path, monkeypatch, [turn("Done.<|im_end|>","unknown")])
    assert report["termination_reason"] == "unknown_finish" and not report["reached_final"]


def test_same_four_tools_and_five_turns_budget(tmp_path, monkeypatch):
    report = run(tmp_path, monkeypatch, [turn(call("Get-Content greeting.py")) for _ in range(5)])
    assert report["termination_reason"] == "tool_budget"
    assert report["tool_call_count"] == 4 and report["generation_turn_count"] == 5


def test_incorrect_edit_is_separate_from_format_and_truncation(tmp_path, monkeypatch):
    command = "$x = Get-Content greeting.py -Raw; $y = $x.Replace('old', 'wrong'); Set-Content greeting.py -Value $y"
    report = run(tmp_path, monkeypatch, [turn(call(command)), turn("Done.<|im_end|>")])
    assert report["final_edit_incorrect"] and report["edit_attempt_count"] == 1
    assert report["format_error_count"] == report["truncated_turn_count"] == 0
    assert report["reached_final"] and not report["strict_success"]


def test_validation_cannot_be_relabelled_as_train(tmp_path, monkeypatch):
    with pytest.raises(ValueError, match="Train-only"):
        run(tmp_path, monkeypatch, [], split="validation")
