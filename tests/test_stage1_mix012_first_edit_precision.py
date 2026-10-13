import json
import pytest
import run_stage1_mix012_first_edit_precision as p


def test_guards_restored_and_budget_frozen():
    assert p.START["physical_free_bytes"] == 14 * 1024**3
    assert set(p.START) == set(p.STOP) == {"physical_free_bytes","commit_free_bytes","wsl_available_bytes","c_free_bytes"}
    assert len(p.CASES) == 4 and p.VARIANTS == ("float32","bfloat16")


@pytest.mark.parametrize("command",[
    "$a = Get-Content -Raw greeting.py; $b = $a.Replace('Hi {name}', 'Hello {name}!'); Set-Content greeting.py -Value $b",
    "(Get-Content -Raw greeting.py).Replace('Hi {name}', 'Hello {name}!') | Set-Content greeting.py",
])
def test_literal_replacement_is_only_in_memory(command):
    original = 'return f"Hi {name}"\n'
    result = p.inspect_command(command,original,'return f"Hello {name}!"\n')
    assert result["exact_reference"] and result["old_found"]
    assert original == 'return f"Hi {name}"\n'


def test_wrong_old_and_missing_punctuation_are_distinct():
    template = "(Get-Content -Raw greeting.py).Replace('%s', '%s') | Set-Content greeting.py"
    result = p.inspect_command(template % ("absent","Hello {name}!"),"Hi {name}","Hello {name}!")
    assert result["grammar_supported"] and not result["old_found"] and not result["exact_reference"]
    result = p.inspect_command(template % ("Hi","Hello"),"Hi {name}","Hello {name}!")
    assert result["old_found"] and not result["exact_reference"]


@pytest.mark.parametrize("command",["Remove-Item greeting.py", "pytest -q", "$(whoami)",
    '(Get-Content -Raw greeting.py).Replace("$env:SECRET", "bad") | Set-Content greeting.py'])
def test_unsupported_commands_never_executed(command,monkeypatch):
    monkeypatch.setattr(p.subprocess,"run",lambda *a,**k: pytest.fail("Unexpected command execution"))
    assert not p.inspect_command(command,"source","reference")["grammar_supported"]


def test_plain_response_or_broken_call_is_not_an_edit():
    assert not p.inspect_completion("Done.","source","reference")["call_format_valid"]
    assert not p.inspect_completion("<tool_call>","source","reference")["call_format_valid"]


def test_frozen_plan_and_existing_inputs():
    plan = p.verify()
    assert [c["task_id"] for c in plan["cases"]] == list(p.CASES)
    assert plan["max_new_tokens"] == 256 and plan["use_cache"] and not plan["do_sample"]
    assert not plan["memory_guard_override"] and not plan["training"] and not plan["command_execution"]
    for case in plan["cases"]:
        assert [m["role"] for m in case["messages"]] == ["system","user","assistant","tool"]
        assert p.canonical_hash(case["messages"]) == case["context_sha256"]
        assert json.loads(case["messages"][-1]["content"])["output"] == case["original_source"]


def test_saved_results_do_not_turn_missing_cases_into_failures():
    report = p.read(p.RUN/"comparison.json")
    assert report["not_task_success_rate"] and report["executed_generated_commands"] == 0
    for variant in p.VARIANTS:
        result = report["variants"][variant]
        assert result["recorded_cases"] == len(result["cases"])
        if result["supervisor"]["status"] == "startup_blocked":
            assert result["recorded_cases"] == 0
    protected = p.read(p.RUN/"protected-input-hashes.json")
    assert all(p.digest(p.ROOT/name) == value for name,value in protected.items())
