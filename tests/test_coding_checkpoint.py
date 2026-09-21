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
