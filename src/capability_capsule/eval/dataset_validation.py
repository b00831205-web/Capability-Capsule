"""Cross-record validation for Teacher trajectory datasets."""

from collections.abc import Sequence

from capability_capsule.eval.curation import validate_split_isolation
from capability_capsule.eval.records import(
    DatasetSplit,
    TeacherTrajectory,
)

from capability_capsule.eval.tasks import TaskSpec


def validate_teacher_dataset(
        trajectories: Sequence[TeacherTrajectory],
        *,
        tasks: Sequence[TaskSpec],
) -> None:
    """Validate Teacher trajectories against their registered task specs"""

    task_by_id: dict[str, TaskSpec] = {}

    for task in tasks:
        if task.task_id in task_by_id:
            raise ValueError(f"Duplicate task ID {task.task_id!r}")

        task_by_id[task.task_id] = task

    seen_trajectory_ids: set[str] = set()
    group_by_task: dict[str, str] = {}

    for trajectory in trajectories:
        if trajectory.trajectory_id in seen_trajectory_ids:
            raise ValueError(
                f"Duplicate trajectory ID {trajectory.trajectory_id!r}"
            )

        seen_trajectory_ids.add(trajectory.trajectory_id)

        if trajectory.split is DatasetSplit.TEST:
            raise ValueError(
                f"Teacher trajectory {trajectory.trajectory_id!r}"
                "must not use the locked test split"
            )

        try:
            task = task_by_id[trajectory.task_id]
        except KeyError as error:
            raise ValueError(
                f"Teacher trajectory reference unknown task "
                f"{trajectory.task_id!r}"
            ) from error

        if trajectory.split is not task.split:
            raise ValueError(
                f"Trajectory split does not match task "
                f"{trajectory.task_id!r}"
            )

        if trajectory.source_revision != task.fixture_revision:
            raise ValueError(
                f"Trajectory source revision does not match task "
                f"{trajectory.task_id!r}"
            )

        if trajectory.task != task.task:
            raise ValueError(
                f"Trajectory text does not match task "
                f"{trajectory.task_id!r}"
            )

        allowed_tools = set(task.allowed_tools)

        for message in trajectory.messages:
            for tool_call in message.tool_calls:
                if tool_call.name not in allowed_tools:
                    raise ValueError(
                        f"Tool {tool_call.name!r} is not allowed for "
                        f"task {trajectory.task_id!r}"
                    )

            if (
                message.tool_name is not None
                and message.tool_name not in allowed_tools
            ):
                    raise ValueError(
                        f"Tool {message.tool_name!r} is not allowed for "
                        f"task {trajectory.task_id!r}"
                    )
        group_by_task[trajectory.task_id] = task.split_group

    validate_split_isolation(
        trajectories,
        group_by_task=group_by_task
    )
        