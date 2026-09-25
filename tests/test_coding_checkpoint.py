from datetime import timedelta
import json
from pathlib import Path

import pytest

from capability_capsule.eval.coding_checkpoint import (
    CodingEvaluationCase,
    CodingValidatorSpec,
    ConstrainedPowerShellExecutor,
    GeneratedTurn,
    copy_verified_evaluation_fixture,
    evaluate_coding_case,
    load_coding_evaluation_suite,
    parse_qwen_tool_completion,
)
from capability_capsule.eval.fixture_provenance import inspect_fixture
from capability_capsule.eval.records import DatasetSplit, RunStatus
from capability_capsule.eval.jsonl import load_jsonl
from capability_capsule.eval.learning_rate import LearningCurvePoint
from capability_capsule.eval.records import CaseResult
from capability_capsule.eval.tasks import TaskCategory, TaskDifficulty, TaskSpec


def make_case(fixture: Path) -> CodingEvaluationCase:
    revision = inspect_fixture("fixture-001", fixture).revision
    return CodingEvaluationCase(
        case_id="case-001",
        fixture_root="fixture",
        task=TaskSpec(
            task_id="task-001",
            fixture_id="fixture-001",
            fixture_family_id="greeting",
            fixture_revision=revision,
            defect_family="greeting-prefix",
            split=DatasetSplit.VALIDATION,
            category=TaskCategory.SINGLE_FILE_CHANGE,
            difficulty=TaskDifficulty.EASY,
            task="Change Hello to Welcome.",
            knowledge_distance=0.2,
            logical_arrival=timedelta(0),
            time_limit=timedelta(minutes=3),
            allowed_tools=("exec_command",),
            expected_changes=(),
            validation_ids=("python:greeting",),
        ),
        validators=(
            CodingValidatorSpec(
                validator_id="python:greeting",
                kind="python_call",
                target="greeting.py",
                function="greeting",
                arguments=("Ada",),
                expected="Welcome, Ada",
            ),
        ),
    )


def write_fixture(path: Path) -> None:
    path.mkdir(parents=True)
    (path / "greeting.py").write_text(
        'def greeting(name: str) -> str:\n    return f"Hello, {name}"\n',
        encoding="utf-8",
    )


def test_parse_qwen_tool_completion() -> None:
    parsed = parse_qwen_tool_completion(
        "inspect\n<tool_call><function=exec_command><parameter=cmd>"
        "cat greeting.py</parameter></function></tool_call><|im_end|>"
    )
    assert parsed.content == "inspect"
    assert parsed.tool_calls[0].arguments == {"cmd": "cat greeting.py"}


def test_copy_verified_fixture_does_not_modify_source(tmp_path: Path) -> None:
    source = tmp_path / "fixture"
    write_fixture(source)
    case = make_case(source)
    destination = tmp_path / "copy"

    copy_verified_evaluation_fixture(
        case,
        artifact_root=tmp_path,
        destination=destination,
    )
    (destination / "greeting.py").write_text("changed", encoding="utf-8")

    assert "Hello" in (source / "greeting.py").read_text("utf-8")


class Generator:
    def __init__(self) -> None:
        self.turn = 0

    def generate(self, messages, tools) -> GeneratedTurn:
        self.turn += 1
        if self.turn == 1:
            completion = (
                "<tool_call><function=exec_command><parameter=cmd>"
                "(Get-Content greeting.py -Raw).Replace('Hello, ', 'Welcome, ') "
                "| Set-Content greeting.py</parameter></function></tool_call>"
            )
        else:
            completion = "Done.<|im_end|>"
        return GeneratedTurn(
            completion=completion,
            input_tokens=10,
            output_tokens=5,
            peak_rss_mb=100,
        )


def test_evaluate_coding_case_records_real_validator_result(tmp_path: Path) -> None:
    fixture = tmp_path / "fixture"
    write_fixture(fixture)
    case = make_case(fixture)

    outcome = evaluate_coding_case(
        case,
        workspace=fixture,
        generator=Generator(),
        executor=ConstrainedPowerShellExecutor(),
        system_prompt="Use exec_command.",
        tools=({"type": "function"},),
        experiment_id="experiment-001",
        run_id="run-001",
        model_id="capsule-001",
        hardware_id="hardware-001",
        cycle_position=1,
    )

    assert outcome.result.status is RunStatus.COMPLETED
    assert outcome.result.success is True
    assert outcome.result.score == 1.0
    assert outcome.result.tool_call_count == 1
    assert outcome.result.invalid_tool_call_count == 0
    assert outcome.validators[0].passed is True


def test_executor_rejects_paths_outside_fixture(tmp_path: Path) -> None:
    fixture = tmp_path / "fixture"
    write_fixture(fixture)

    result = ConstrainedPowerShellExecutor().execute(
        "Get-Content E:/capsule/secret.txt",
        fixture,
    )

    assert result.authorized is False
    assert result.exit_code == 126


def test_executor_accepts_guarded_training_edit(tmp_path: Path) -> None:
    fixture = tmp_path / "fixture"
    write_fixture(fixture)

    result = ConstrainedPowerShellExecutor().execute(
        "$text = Get-Content -LiteralPath greeting.py -Raw; "
        "$updated = $text.Replace('Hello, ', 'Welcome, '); "
        "Set-Content -LiteralPath greeting.py -Value $updated",
        fixture,
    )

    assert result.authorized is True
    assert result.exit_code == 0
    assert 'return f"Welcome, {name}"' in (fixture / "greeting.py").read_text("utf-8")


def test_executor_accepts_raw_before_path(tmp_path: Path) -> None:
    fixture = tmp_path / "fixture"
    write_fixture(fixture)
    executor = ConstrainedPowerShellExecutor()

    read = executor.execute("Get-Content -Raw greeting.py", fixture)
    edit = executor.execute(
        "$text = Get-Content -Raw greeting.py; "
        "$updated = $text.Replace('Hello, ', 'Welcome, '); "
        "Set-Content -Path greeting.py -Value $updated",
        fixture,
    )

    assert read.authorized is True
    assert read.exit_code == 0
    assert 'return f"Hello, {name}"' in read.output
    assert edit.authorized is True
    assert edit.exit_code == 0
    assert 'return f"Welcome, {name}"' in (fixture / "greeting.py").read_text("utf-8")


@pytest.mark.parametrize(
    "command",
    [
        "Get-Content -Path greeting.py -Raw | Select-String Hello | "
        "ForEach-Object { $_.Replace('Hello', 'Welcome') } | "
        "Set-Content -Path greeting.py",
        "$text = Get-Content -Path greeting.py -Raw; "
        "$updated = $text.Replace('Hello', 'Welcome'); "
        "Set-Content -Path greeting.py -Value $text",
        "$text = Get-Content -Path greeting.py -Raw; "
        "$updated = $text.Replace('Hello', 'Welcome'); "
        "Set-Content -Path greeting.py -Value $updated; Get-Content greeting.py",
    ],
)
def test_executor_rejects_ambiguous_replace_commands(
    tmp_path: Path, command: str
) -> None:
    fixture = tmp_path / "fixture"
    write_fixture(fixture)
    original = (fixture / "greeting.py").read_bytes()

    result = ConstrainedPowerShellExecutor().execute(command, fixture)

    assert result.authorized is False
    assert result.exit_code == 126
    assert (fixture / "greeting.py").read_bytes() == original


@pytest.mark.parametrize("fixture_root", ["../fixture", "/fixture", "C:/fixture", "a\\b"])
def test_case_rejects_unsafe_fixture_root(tmp_path: Path, fixture_root: str) -> None:
    fixture = tmp_path / "fixture"
    write_fixture(fixture)
    payload = make_case(fixture).model_dump()
    payload["fixture_root"] = fixture_root
    with pytest.raises(ValueError, match="safe relative path"):
        CodingEvaluationCase.model_validate(payload)


def test_stage1_checkpoint_evaluation_records_observed_harness_failure() -> None:
    root = Path(__file__).resolve().parents[1]
    run = root / "runs" / "evaluation" / "stage1-codex-qwen35-2b-001"
    results = load_jsonl(run / "case-results.jsonl", CaseResult)
    points = load_jsonl(run / "learning-curve.jsonl", LearningCurvePoint)
    reports = [
        json.loads((run / "case-reports" / f"{result.case_id}.json").read_text("utf-8"))
        for result in results
    ]

    assert len(results) == 2
    assert [result.status for result in results] == [RunStatus.FAILED, RunStatus.FAILED]
    assert [result.success for result in results] == [False, False]
    assert [result.invalid_tool_call_count for result in results] == [1, 3]
    assert all(result.error_type == "ToolRoundLimitExceeded" for result in results)
    assert points[0].checkpoint_id == "stage1-codex-qwen35-2b-001-step-8"
    assert points[0].success_rate == 0.0
    assert points[0].evaluation_suite_digest == (
        "ff3932cd479946ed1b69c42c9a9ffbe5cbb120adebd6a1d5728cc34e9514c894"
    )
    assert "sed -i" in "\n".join(reports[0]["completions"])
    assert "cat > greeting.py <<" in "\n".join(reports[1]["completions"])
    assert all(not validator["passed"] for report in reports for validator in report["validators"])
    assert "Hello" in (
        run / "workspaces" / results[0].case_id / "greeting.py"
    ).read_text("utf-8")


def test_stage1_validation_v2_preserves_fixed_cases_and_adds_unseen_fixture() -> None:
    root = Path(__file__).resolve().parents[1]
    previous = load_coding_evaluation_suite(
        root
        / "runs"
        / "evaluation"
        / "stage1-codex-qwen35-2b-001"
        / "evaluation-suite.json"
    )
    current = load_coding_evaluation_suite(
        root
        / "plans"
        / "evaluation"
        / "stage1-codex-validation-v2"
        / "evaluation-suite.json"
    )
    unseen = current.cases[-1]

    assert current.cases[:2] == previous.cases
    assert unseen.case_id == "validation-powershell-salutation-unseen-001"
    assert inspect_fixture(
        unseen.task.fixture_id,
        root / unseen.fixture_root,
    ).revision == unseen.task.fixture_revision
    assert current.identity().evaluation_suite_digest == (
        "3e756195c1585c57c4dcc8a3fef40cb2653a67bc57820c6156602f357301b9bb"
    )


def test_stage1_validation_v3_changes_only_generic_command_policy() -> None:
    root = Path(__file__).resolve().parents[1]
    v2 = load_coding_evaluation_suite(
        root / "plans/evaluation/stage1-codex-validation-v2/evaluation-suite.json"
    )
    v3 = load_coding_evaluation_suite(
        root
        / "plans/evaluation/stage1-codex-validation-v3-command-policy/evaluation-suite.json"
    )

    assert v3.cases == v2.cases
    assert v3.identity().evaluation_suite_digest == v2.identity().evaluation_suite_digest
    assert v3.evaluation_suite_id != v2.evaluation_suite_id
    assert v3.system_prompt.startswith(v2.system_prompt)
    added_policy = v3.system_prompt[len(v2.system_prompt):]
    assert ".Replace" in added_policy
    assert "Set-Content" in added_policy
    assert "Welcome" not in added_policy
    assert "PowerShell ready" not in added_policy


def test_stage1_powershell_checkpoint_evaluation_records_fixed_and_unseen_failure() -> None:
    root = Path(__file__).resolve().parents[1]
    run = (
        root
        / "runs"
        / "evaluation"
        / "stage1-codex-qwen35-2b-002-recovery-004-rerun-001"
    )
    results = load_jsonl(run / "case-results.jsonl", CaseResult)
    points = load_jsonl(run / "learning-curve.jsonl", LearningCurvePoint)
    unseen_workspace = run / "workspaces" / "validation-powershell-salutation-unseen-001"

    assert [result.case_id for result in results] == [
        "smoke-validation-greeting-change-codex-001",
        "validation-greeting-exec-exec-002",
        "validation-powershell-salutation-unseen-001",
    ]
    assert [result.success for result in results] == [False, False, False]
    assert [result.invalid_tool_call_count for result in results] == [1, 0, 0]
    assert all(result.error_type == "ToolRoundLimitExceeded" for result in results)
    assert points[0].checkpoint_id == "stage1-codex-qwen35-2b-002-step-16"
    assert points[0].success_rate == 0.0
    assert points[0].cumulative_trajectory_count == 16
    assert points[0].evaluation_suite_digest == (
        "3e756195c1585c57c4dcc8a3fef40cb2653a67bc57820c6156602f357301b9bb"
    )
    assert "Hello, {name}" in (unseen_workspace / "greeting.py").read_text("utf-8")


def test_contract_004_checkpoint_records_protocol_aligned_behavior_failure() -> None:
    root = Path(__file__).resolve().parents[1]
    run = (
        root
        / "runs"
        / "evaluation"
        / "stage1-codex-qwen35-2b-004-contract-v2"
    )
    suite = load_coding_evaluation_suite(run / "evaluation-suite.json")
    results = load_jsonl(run / "case-results.jsonl", CaseResult)
    points = load_jsonl(run / "learning-curve.jsonl", LearningCurvePoint)
    reports = [
        json.loads((run / "case-reports" / f"{result.case_id}.json").read_text("utf-8"))
        for result in results
    ]

    assert suite.identity().evaluation_suite_digest == (
        "3e756195c1585c57c4dcc8a3fef40cb2653a67bc57820c6156602f357301b9bb"
    )
    assert [result.case_id for result in results] == [
        "smoke-validation-greeting-change-codex-001",
        "validation-greeting-exec-exec-002",
        "validation-powershell-salutation-unseen-001",
    ]
    assert [result.success for result in results] == [False, False, False]
    assert [result.invalid_tool_call_count for result in results] == [4, 3, 3]
    assert all(result.error_type == "ToolRoundLimitExceeded" for result in results)
    assert all(result.tool_call_count == 4 for result in results)

    point = points[0]
    assert point.checkpoint_id == "stage1-codex-qwen35-2b-004-contract-step-8"
    assert point.cumulative_trajectory_count == 8
    assert point.cumulative_token_count == 5905
    assert point.success_rate == 0.0
    assert point.evaluation_suite_digest == (
        "3e756195c1585c57c4dcc8a3fef40cb2653a67bc57820c6156602f357301b9bb"
    )

    completions = ["\n".join(report["completions"]) for report in reports]
    assert "Get-ChildItem" in completions[0]
    assert "cat greeting.py" in completions[1]
    assert "cat greeting.py" in completions[2]
    assert "cat > greeting.py <<" in completions[1]
    assert "cat > greeting.py <<" in completions[2]
    assert all(not validator["passed"] for report in reports for validator in report["validators"])


def test_contract_004_higher_exposure_checkpoint_stops_before_tool_use() -> None:
    root = Path(__file__).resolve().parents[1]
    run = (
        root
        / "runs"
        / "evaluation"
        / "stage1-codex-qwen35-2b-005-contract-32step-v2"
    )
    suite = load_coding_evaluation_suite(run / "evaluation-suite.json")
    results = load_jsonl(run / "case-results.jsonl", CaseResult)
    points = load_jsonl(run / "learning-curve.jsonl", LearningCurvePoint)
    reports = [
        json.loads((run / "case-reports" / f"{result.case_id}.json").read_text("utf-8"))
        for result in results
    ]

    assert suite.identity().evaluation_suite_digest == (
        "3e756195c1585c57c4dcc8a3fef40cb2653a67bc57820c6156602f357301b9bb"
    )
    assert len(results) == 3
    assert all(result.status is RunStatus.COMPLETED for result in results)
    assert all(result.success is False for result in results)
    assert all(result.score == 0.0 for result in results)
    assert all(result.tool_call_count == 0 for result in results)
    assert all(result.invalid_tool_call_count == 0 for result in results)
    assert all(result.error_type is None for result in results)
    assert all(
        report["completions"] == [
            "Inspect the current file before editing.<|im_end|>"
        ]
        for report in reports
    )
    assert all(not validator["passed"] for report in reports for validator in report["validators"])

    point = points[0]
    assert point.checkpoint_id == (
        "stage1-codex-qwen35-2b-005-contract-32step-step-32"
    )
    assert point.cumulative_trajectory_count == 8
    assert point.cumulative_token_count == 5905
    assert point.success_rate == 0.0
    assert point.evaluation_suite_digest == (
        "3e756195c1585c57c4dcc8a3fef40cb2653a67bc57820c6156602f357301b9bb"
    )


def test_normalized_turn_checkpoint_uses_tools_but_fails_edit_validation() -> None:
    root = Path(__file__).resolve().parents[1]
    run = root / "runs/evaluation/stage1-codex-qwen35-2b-006-turns-v2-v2"
    suite = load_coding_evaluation_suite(run / "evaluation-suite.json")
    results = load_jsonl(run / "case-results.jsonl", CaseResult)
    points = load_jsonl(run / "learning-curve.jsonl", LearningCurvePoint)
    reports = [
        json.loads((run / "case-reports" / f"{result.case_id}.json").read_text("utf-8"))
        for result in results
    ]

    assert suite.identity().evaluation_suite_digest == (
        "3e756195c1585c57c4dcc8a3fef40cb2653a67bc57820c6156602f357301b9bb"
    )
    assert [result.success for result in results] == [False, False, False]
    assert [result.tool_call_count for result in results] == [4, 4, 4]
    assert [result.invalid_tool_call_count for result in results] == [3, 3, 2]
    assert all(result.error_type == "ToolRoundLimitExceeded" for result in results)
    assert all("Get-Content -Path greeting.py -Raw" in report["completions"][0]
               for report in reports)
    assert all(report["tool_results"][0]["authorized"] is True
               for report in reports)
    assert all(not outcome["passed"] for report in reports
               for outcome in report["validators"])
    assert 'return f"Ada!, {name}!"' in (
        run / "workspaces/validation-powershell-salutation-unseen-001/greeting.py"
    ).read_text("utf-8")

    point = points[0]
    assert point.checkpoint_id == "stage1-codex-qwen35-2b-006-turns-v2-step-32"
    assert point.cumulative_trajectory_count == 8
    assert point.cumulative_token_count == 5705
    assert point.success_rate == 0.0
    assert point.evaluation_suite_digest == (
        "3e756195c1585c57c4dcc8a3fef40cb2653a67bc57820c6156602f357301b9bb"
    )


def test_command_policy_comparison_keeps_cases_fixed_and_records_failed_edit() -> None:
    root = Path(__file__).resolve().parents[1]
    run = root / "runs/evaluation/stage1-codex-qwen35-2b-006-turns-v2-v3-command-policy-raw-order"
    suite = load_coding_evaluation_suite(run / "evaluation-suite.json")
    results = load_jsonl(run / "case-results.jsonl", CaseResult)
    points = load_jsonl(run / "learning-curve.jsonl", LearningCurvePoint)
    manifest = json.loads((run / "run-manifest.json").read_text("utf-8"))
    reports = [
        json.loads((run / "case-reports" / f"{result.case_id}.json").read_text("utf-8"))
        for result in results
    ]

    assert suite.cases == load_coding_evaluation_suite(
        root / "plans/evaluation/stage1-codex-validation-v2/evaluation-suite.json"
    ).cases
    assert manifest["case_digest"] == suite.identity().evaluation_suite_digest
    assert manifest["success_count"] == 0
    assert manifest["max_tool_rounds"] == 4
    assert [result.success for result in results] == [False, False, False]
    assert [result.invalid_tool_call_count for result in results] == [1, 2, 2]
    assert all(report["tool_results"][0]["authorized"] for report in reports)
    assert all("Get-Content -Raw greeting.py" in report["completions"][0] for report in reports)
    assert all(not validator["passed"] for report in reports for validator in report["validators"])
    assert points[0].success_rate == 0.0
    assert points[0].evaluation_suite_id == suite.evaluation_suite_id
    fallback = root / manifest["fixture_sources"][results[0].case_id]
    assert inspect_fixture(suite.cases[0].task.fixture_id, fallback).revision == (
        suite.cases[0].task.fixture_revision
    )
