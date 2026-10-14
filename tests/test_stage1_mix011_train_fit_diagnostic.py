"""Free-running train diagnosis without Teacher context or validation relabeling."""

import pytest

import run_stage1_mix011_train_fit_diagnostic as diagnostic
from capability_capsule.eval.coding_checkpoint import ConstrainedPowerShellExecutor, GeneratedTurn


class ScriptedGenerator:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.calls = 0

    def generate(self, messages, tools):
        self.calls += 1
        if self.calls == 1:
            assert messages == [{"role": "system", "content": "System"}, {"role": "user", "content": "Task"}]
        return GeneratedTurn(completion=next(self.responses), input_tokens=1, output_tokens=1, peak_rss_mb=1)


def tool(command):
    return f"<tool_call><function=exec_command><parameter=cmd>{command}</parameter></function></tool_call>"


@pytest.fixture
def workspace(tmp_path):
    (tmp_path / "greeting.py").write_text('def greeting(name):\n    return f"Hello {name}"\n', encoding="utf-8")
    (tmp_path / "test_greeting.py").write_text('from greeting import greeting\ndef test_names():\n    assert greeting("A") == "Welcome A"\n    assert greeting("B") == "Welcome B"\n', encoding="utf-8")
    return tmp_path


def execute(workspace, generator):
    return diagnostic.run_case({"task_id": "train-synthetic", "split": "train", "task": "Task"},
        {"system_prompt": "System", "tools": []}, workspace, generator,
        ConstrainedPowerShellExecutor(), 'def greeting(name):\n    return f"Welcome {name}"\n')


def test_free_running_success_requires_edit_test_and_final(workspace):
    generator = ScriptedGenerator([tool("Get-Content greeting.py"),
        tool("$text = Get-Content -Raw greeting.py; $updated = $text.Replace('Hello', 'Welcome'); Set-Content greeting.py -Value $updated"),
        tool("pytest -q"), "Completed."])
    report = execute(workspace, generator)
    assert report["strict_success"] and report["behavior_pass"] and report["source_changed"]
    assert report["tests_unchanged"] and report["model_invoked_pytest"]
    assert report["tool_call_count"] == 3 and report["generation_turn_count"] == 4
    assert report["split"] == "train" and report["seen_during_training"]


def test_budget_prevents_fifth_execution_and_keeps_failed_source(workspace):
    generator = ScriptedGenerator([tool("Get-ChildItem")] * 5)
    report = execute(workspace, generator)
    assert report["tool_call_count"] == report["denied_command_count"] == 4
    assert report["generation_turn_count"] == 5 and report["error"] == "ToolRoundLimitExceeded"
    assert not report["strict_success"] and not report["source_changed"] and not report["behavior_pass"]


def test_empty_final_claim_does_not_count_as_success(workspace):
    report = execute(workspace, ScriptedGenerator(["Completed."]))
    assert report["reached_final"] and not report["strict_success"] and not report["behavior_pass"]


def test_existing_output_and_attempt_are_never_overwritten(tmp_path, monkeypatch):
    monkeypatch.setattr(diagnostic, "RUN", tmp_path)
    with pytest.raises(FileExistsError):
        diagnostic.prepare()
    diagnostic.save(tmp_path / "inference-attempt.json", {"one_shot": True})
    with pytest.raises(FileExistsError):
        diagnostic.evaluate()


def test_preservation_keeps_explicit_legacy_audit_pin(tmp_path, monkeypatch):
    monkeypatch.setattr(diagnostic, "ROOT", tmp_path)
    monkeypatch.setattr(diagnostic, "RUN", tmp_path / "new-run")
    audit = tmp_path / "tests/audit.json"
    diagnostic.save(audit, {"original": True})
    diagnostic.save(diagnostic.RUN / "protected-input-hashes.json", {"tests/audit.json": diagnostic.digest(audit)})
    monkeypatch.setattr(diagnostic, "protected_hashes", lambda: {})
    assert diagnostic.verify_preservation() == 1
    audit.write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="Protected input changed"):
        diagnostic.verify_preservation()
