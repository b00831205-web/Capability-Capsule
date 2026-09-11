"""End-to-end checks for the committed smoke-pilot task catalog."""

from pathlib import Path

from capability_capsule.eval.fixture_provenance import verify_task_fixture
from capability_capsule.eval.jsonl import load_jsonl
from capability_capsule.eval.records import DatasetSplit
from capability_capsule.eval.tasks import (
    ChangeOperation,
    TaskCategory,
    TaskSpec,
)


_REPOSITORY_ROOT = Path(__file__).parents[1]
_PILOT_ROOT = _REPOSITORY_ROOT / "fixtures" / "smoke-pilot"
_TASKS_PATH = _PILOT_ROOT / "tasks.jsonl"
_FIXTURE_ROOTS = {
    "smoke-cli-search-v1": _PILOT_ROOT / "train-cli-search",
    "smoke-greeting-change-v1": _PILOT_ROOT / "validation-greeting-change",
}


def test_smoke_pilot_tasks_match_authorized_fixture_snapshots() -> None:
    tasks = load_jsonl(_TASKS_PATH, TaskSpec)

    assert [task.task_id for task in tasks] == [
        "smoke-train-cli-search-001",
        "smoke-validation-greeting-change-001",
    ]

    for task in tasks:
        assert task.split is not DatasetSplit.TEST
        assert verify_task_fixture(task, _FIXTURE_ROOTS[task.fixture_id])


def test_smoke_pilot_covers_read_only_train_and_bounded_validation_change() -> None:
    train, validation = load_jsonl(_TASKS_PATH, TaskSpec)

    assert train.split is DatasetSplit.TRAIN
    assert train.category is TaskCategory.CODE_SEARCH
    assert train.expected_changes == ()
    assert train.allowed_tools == ("exec_command",)

    assert validation.split is DatasetSplit.VALIDATION
    assert validation.category is TaskCategory.SINGLE_FILE_CHANGE
    assert validation.expected_changes[0].path == "greeting.py"
    assert validation.expected_changes[0].operation is ChangeOperation.MODIFY
    assert validation.allowed_tools == ("exec_command", "apply_patch")


def test_smoke_pilot_fixture_families_are_disjoint_between_splits() -> None:
    train, validation = load_jsonl(_TASKS_PATH, TaskSpec)

    assert train.fixture_family_id != validation.fixture_family_id
    assert train.split_group != validation.split_group
