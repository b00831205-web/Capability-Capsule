"""Process completed training checkpoint into learning-curve records"""

from datetime import datetime
from pathlib import Path
from typing import Literal, Self


from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

from capability_capsule.eval.evaluation_suite import (
    EvaluationSuiteIdentity,
    inspect_evaluation_suite,
)

from capability_capsule.eval.records import (
    CaseResult,
    DatasetSplit,
)
from capability_capsule.eval.training_checkpoint_recording import (
    TrainingCheckpointRecord,
    record_training_checkpoint
)
from capability_capsule.eval.training_evaluation import (
    summarize_training_evaluation,
)
from capability_capsule.eval.training_provenance import TrainingRunManifest
from capability_capsule.telemetry.knowledge_usage import KnowledgeUsageSummary
from capability_capsule.eval.jsonl import load_jsonl
from capability_capsule.telemetry.knowledge_usage import (
    KnowledgeUsageEvent,
    KnowledgeUsageSummary,
    summarize_knowledge_usage,
)

class TrainingCheckpointCompleted(BaseModel):
    """Validated event emitted after one checkpoint evaluation."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["0.2"] = "0.2"
    completed_at: datetime
    training_run: TrainingRunManifest
    experiment_id: str = Field(min_length=1)
    task_family_id: str = Field(min_length=1)
    evaluation_split: DatasetSplit
    evaluation_suite: EvaluationSuiteIdentity
    results: tuple[CaseResult, ...] = Field(min_length=1)
    knowledge_usage: KnowledgeUsageSummary | None = None

    @field_validator(
        "experiment_id",
        "task_family_id"
    )
    @classmethod
    def reject_blank_identifiers(cls, value: str) -> str:
        if not value.strip():
            raise ValueError(
                "Training completion identifiers must not be blank"
            )

        return value

    @field_validator("completed_at")
    @classmethod
    def require_timezone(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError(
                "completed_at must include a timezone"
            )

        return value

    @model_validator(mode="after")
    def validate_result(self) -> Self:
        if self.evaluation_suite.evaluation_split != self.evaluation_split:
            raise ValueError(
                "Evaluation suite evaluation split does not match "
                "the training completion evaluation split"
            )

        for result in self.results:
            if result.experiment_id != self.experiment_id:
                raise ValueError(
                    f"Case result {result.case_id!r} belongs to "
                    f"experiment {result.experiment_id!r}, not "
                    f"{self.experiment_id!r}"
                )

            if (
                result.model_id != self.training_run.output_capsule_id
            ):
                raise ValueError(
                    f"Case result {result.case_id!r} does not "
                    "belong to the training output capsule"
                )

        identities = [
            (
                result.run_id,
                result.case_id,
                result.repetition,
                result.cycle_position,
            )
            for result in self.results
        ]

        if len(set(identities)) != len(identities):
            raise ValueError(
                "Training completion results must be unique"
            )

        return self

def process_training_checkpoint_completion(
        completion: TrainingCheckpointCompleted,
        *,
        ledger_path: Path,
        checkpoint_id: str | None = None,
        target_success_rate: float = 0.8,
        plateau_threshold_percentage_points_per_100_trajectories: float = 1.0,
        plateau_interval_count: int = 2,
) -> TrainingCheckpointRecord:
    """Summarize and record one completed checkpoint evaluation."""

    evaluation = summarize_training_evaluation(
        completion.training_run,
        experiment_id= completion.experiment_id,
        results = completion.results,
    )

    return record_training_checkpoint(
        evaluation,
        ledger_path = ledger_path,
        base_model_id = completion.training_run.base_model_id,
        capability_id= completion.evaluation_suite.capability_id,
        evaluation_suite_id = completion.evaluation_suite.evaluation_suite_id,
        evaluation_suite_digest = completion.evaluation_suite.evaluation_suite_digest,
        checkpoint_id = checkpoint_id,
        task_family_id = completion.task_family_id,
        evaluation_split = completion.evaluation_split,
        knowledge_usage = completion.knowledge_usage,
        target_success_rate = target_success_rate,
        plateau_threshold_percentage_points_per_100_trajectories = plateau_threshold_percentage_points_per_100_trajectories,
        plateau_interval_count = plateau_interval_count,
    )

def _load_checkpoint_knowledge_usage(
        path: Path,
        *,
        capsule_id: str,
        task_family_id: str
) -> KnowledgeUsageSummary:
    """Load knowledge events belonging to one evaluated checkpoint."""

    source = path.resolve(strict=True)

    if not source.is_dir():
        raise NotADirectoryError(source)

    events = tuple(
        KnowledgeUsageEvent.model_validate_json(
            event_path.read_bytes()
        )
        for event_path in sorted(source.glob("*.json"))
        if event_path.is_file()
    )

    if any(
        event.capsule_id != capsule_id for event in events
    ):
        raise ValueError(
            "Knowledge usage events must belong to the task family"
        )

    knowledge_tree_digest = {
        event.knowledge_tree_digest for event in events
    }

    if len(knowledge_tree_digest) > 1:
        raise ValueError(
            "Knowledge usage events must use one knowledge tree"
        )

    return summarize_knowledge_usage(events)

def process_training_checkpoint_files(
        *,
        training_run_path: Path,
        results_path: Path,
        evaluation_suite_path: Path,
        capability_id: str,
        evaluation_suite_id: str,
        ledger_path: Path,
        completed_at: datetime,
        experiment_id: str,
        task_family_id: str,
        evaluation_split: DatasetSplit,
        knowledge_usage_dir: Path | None = None,
        checkpoint_id: str |None = None,
        target_success_rate: float = 0.8,
        plateau_threshold_percentage_points_per_100_trajectories: float = 1.0,
        plateau_interval_count: int = 2,
) -> TrainingCheckpointRecord:
    """Load standard artifactis and process one completed checkpoint."""

    training_run = TrainingRunManifest.model_validate_json(
        training_run_path.read_bytes()
    )
    results = load_jsonl(
        results_path,
        CaseResult,
    )
    evaluation_suite = inspect_evaluation_suite(
        evaluation_suite_path,
        capability_id = capability_id,
        evaluation_suite_id = evaluation_suite_id,
        evaluation_split = evaluation_split
    )
    knowledge_usage: KnowledgeUsageSummary | None = None

    if knowledge_usage_dir is not None:
        knowledge_usage = _load_checkpoint_knowledge_usage(
            knowledge_usage_dir,
            capsule_id = training_run.output_capsule_id,
            task_family_id= task_family_id,
        )

    completion = TrainingCheckpointCompleted(
        completed_at= completed_at,
        training_run = training_run,
        experiment_id= experiment_id,
        task_family_id= task_family_id,
        evaluation_split= evaluation_split,
        evaluation_suite = evaluation_suite,
        results = results,
        knowledge_usage= knowledge_usage,
    )

    return process_training_checkpoint_completion(
        completion,
        ledger_path = ledger_path,
        checkpoint_id = checkpoint_id,
        target_success_rate= target_success_rate,
        plateau_threshold_percentage_points_per_100_trajectories= plateau_threshold_percentage_points_per_100_trajectories,
        plateau_interval_count= plateau_interval_count
    )
