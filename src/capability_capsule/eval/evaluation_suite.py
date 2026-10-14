"""Stable identities for capability-specific evaluation suites."""

import json
from hashlib import sha256
from pathlib import Path
from typing import Literal

from pydantic import (BaseModel, ConfigDict, Field, field_validator)

from capability_capsule.eval.dataset import load_cases
from capability_capsule.eval.records import DatasetSplit


class EvaluationSuiteIdentity(BaseModel):
    """Semantic identity of one fixed capability evaluation suite."""

    model_config = ConfigDict(extra="forbid", frozen =True)

    schema_version: Literal["0.1"] = "0.1"
    capability_id: str = Field(min_length=1)
    evaluation_suite_id: str = Field(min_length=1)
    evaluation_split: DatasetSplit
    case_count: int = Field(gt=0)
    evaluation_suite_digest: str = Field(pattern = r"^[0-9a-f]{64}$")

    @field_validator(
        "capability_id",
        "evaluation_suite_id",
    )
    @classmethod
    def reject_blank_identifiers(cls, value: str) -> str:
        if not value.strip():
            raise ValueError(
                "Evaluation suite identifiers must not be blank"
            )

        return value


def inspect_evaluation_suite(
        path: Path,
        *,
        capability_id: str,
        evaluation_suite_id: str,
        evaluation_split: DatasetSplit,
) -> EvaluationSuiteIdentity:
    """Validate cases and calculate a formatting-independent digest."""

    cases = load_cases(path)

    canonical_payload = [
        case.model_dump(
            mode="json",
            exclude_defaults=True,
        )
        for case in cases
    ]

    canonical_bytes = json.dumps(
        canonical_payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


    return EvaluationSuiteIdentity(
        capability_id= capability_id,
        evaluation_suite_id= evaluation_suite_id,
        evaluation_split= evaluation_split,
        case_count = len(cases),
        evaluation_suite_digest= sha256(
            canonical_bytes
        ).hexdigest(),
    )
