import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from capability_capsule.eval.evaluation_suite import (
    EvaluationSuiteIdentity,
    inspect_evaluation_suite,
)
from capability_capsule.eval.records import DatasetSplit


def write_suite(
    path: Path,
    *,
    question: str = "Where is the CLI entry point?",
    indent: int | None = None,
) -> None:
    path.write_text(
        json.dumps(
            [
                {
                    "question": question,
                    "expected_paths": ["src/capability_capsule/cli/__init__.py"],
                }
            ],
            indent=indent,
        ),
        encoding="utf-8",
    )


def test_inspect_evaluation_suite_has_semantic_digest(tmp_path: Path) -> None:
    compact = tmp_path / "compact.json"
    formatted = tmp_path / "formatted.json"
    changed = tmp_path / "changed.json"
    write_suite(compact)
    write_suite(formatted, indent=4)
    write_suite(changed, question="Which command starts the CLI?")

    first = inspect_evaluation_suite(
        compact,
        capability_id="repository-cli-navigation",
        evaluation_suite_id="cli-validation-v1",
        evaluation_split=DatasetSplit.VALIDATION,
    )
    second = inspect_evaluation_suite(
        formatted,
        capability_id="repository-cli-navigation",
        evaluation_suite_id="cli-validation-v1",
        evaluation_split=DatasetSplit.VALIDATION,
    )
    different = inspect_evaluation_suite(
        changed,
        capability_id="repository-cli-navigation",
        evaluation_suite_id="cli-validation-v1",
        evaluation_split=DatasetSplit.VALIDATION,
    )

    assert first.capability_id == "repository-cli-navigation"
    assert first.evaluation_suite_id == "cli-validation-v1"
    assert first.evaluation_split is DatasetSplit.VALIDATION
    assert first.case_count == 1
    assert len(first.evaluation_suite_digest) == 64
    assert first.evaluation_suite_digest == second.evaluation_suite_digest
    assert first.evaluation_suite_digest != different.evaluation_suite_digest


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("capability_id", " "),
        ("evaluation_suite_id", ""),
    ],
)
def test_evaluation_suite_identity_rejects_blank_ids(
    field: str,
    value: str,
) -> None:
    data = {
        "capability_id": "repository-cli-navigation",
        "evaluation_suite_id": "cli-validation-v1",
        "evaluation_split": DatasetSplit.VALIDATION,
        "case_count": 1,
        "evaluation_suite_digest": "a" * 64,
    }
    data[field] = value

    with pytest.raises(ValidationError):
        EvaluationSuiteIdentity.model_validate(data)


def test_inspect_evaluation_suite_rejects_invalid_cases(tmp_path: Path) -> None:
    path = tmp_path / "invalid.json"
    path.write_text("[]", encoding="utf-8")

    with pytest.raises(ValueError):
        inspect_evaluation_suite(
            path,
            capability_id="repository-cli-navigation",
            evaluation_suite_id="cli-validation-v1",
            evaluation_split=DatasetSplit.VALIDATION,
        )
