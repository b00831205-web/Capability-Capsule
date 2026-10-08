"""Collect observable train-only PowerShell edit trajectories for Stage 1."""

import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import timedelta
from hashlib import sha256
from pathlib import Path

from capability_capsule.eval.coding_checkpoint import ConstrainedPowerShellExecutor
from capability_capsule.eval.fixture_provenance import inspect_fixture
from capability_capsule.eval.harness_profile import verify_harness_profile
from capability_capsule.eval.jsonl import load_jsonl
from capability_capsule.eval.records import (
    DatasetSplit,
    TeacherTrajectory,
    ToolCallRecord,
    TrajectoryMessage,
)
from capability_capsule.eval.student_target import StudentTarget, verify_student_target
from capability_capsule.eval.tasks import (
    ChangeOperation,
    ExpectedChange,
    TaskCategory,
    TaskDifficulty,
    TaskSpec,
)
from capability_capsule.eval.teacher_collection import append_teacher_trajectory
from capability_capsule.eval.teacher_collection_plan import (
    TeacherCollectionPlan,
    build_teacher_collection_plan,
)
from capability_capsule.eval.teacher_collection_plan_publication import (
    PublishedTeacherCollectionPlan,
    load_teacher_collection_plan,
)


ROOT = Path(__file__).resolve().parents[1]
PLAN_ID = "stage1-codex-powershell-edit-005"
BASE = ROOT / "artifacts/teacher-fixtures/train-greeting-good-day"
BASE_REVISION = "fc257a990228187dc3e707a0e9acfe8799487eb5b1166b17d75007d89dcb91ce"
SOURCES = ROOT / "artifacts/teacher-fixtures" / PLAN_ID
WORKSPACES = ROOT / "tmp/teacher" / PLAN_ID / "workspaces"
PLAN_DIR = ROOT / "plans/teacher" / PLAN_ID
RAW_RELATIVE = Path("datasets/teacher") / PLAN_ID / "raw.jsonl"
RAW = ROOT / RAW_RELATIVE
OLD_PLAN = ROOT / "plans/teacher/stage1-codex-powershell-contract-004/collection-plan.json"


@dataclass(frozen=True)
class Variant:
    slug: str
    source: str
    old: str
    new: str
    name: str
    expected: str
    task: str
    fallback: str | None = None


VARIANTS = (
    Variant(
        "morning-docstring",
        '"""Keep this module description."""\n\ndef greeting(name: str) -> str:\n    return f"Hello, {name}"\n',
        'return f"Hello, {name}"', 'return f"Morning, {name}!"',
        "Mira", "Morning, Mira!",
        'Change only greeting.py so greeting("Mira") returns "Morning, Mira!". Keep the module description and run the supplied test.',
    ),
    Variant(
        "evening-blank-lines",
        '\n\ndef greeting(name: str) -> str:\n\n    return f"Hello, {name}"\n',
        'return f"Hello, {name}"', 'return f"Good evening, {name}."',
        "Ivo", "Good evening, Ivo.",
        'For Ivo, greeting should say "Good evening, Ivo." Edit greeting.py only and validate.',
    ),
    Variant(
        "meet-single-quotes",
        "def greeting(name: str) -> str:\n    return f'Hello, {name}'\n",
        "return f'Hello, {name}'", "return f'Nice to meet you, {name}!'",
        "Nia", "Nice to meet you, Nia!",
        'Update greeting.py to greet Nia with "Nice to meet you, Nia!"; leave the test file untouched.',
    ),
    Variant(
        "cheers-comment",
        '# The old greeting example is documentation, not the behavior.\ndef greeting(name: str) -> str:\n    return f"Hello, {name}"\n',
        'return f"Hello, {name}"', 'return f"Cheers, {name}!"',
        "Noor", "Cheers, Noor!",
        'Make greeting("Noor") return "Cheers, Noor!" by changing only the behavior in greeting.py, then test.',
    ),
    Variant(
        "all-set-variable",
        'def greeting(name: str) -> str:\n    message = f"Hello, {name}"\n    return message\n',
        'message = f"Hello, {name}"', 'message = f"All set, {name}."',
        "Lena", "All set, Lena.",
        'In greeting.py, update the constructed message so Lena hears "All set, Lena.". Run the provided test.',
    ),
    Variant(
        "glad-prefix",
        'def greeting(name: str) -> str:\n    prefix = "Hello"\n    return f"{prefix}, {name}"\n',
        'prefix = "Hello"', 'prefix = "Glad to see you"',
        "Tari", "Glad to see you, Tari",
        'Change the greeting prefix in greeting.py so the result for Tari is "Glad to see you, Tari"; validate.',
    ),
    Variant(
        "again-indented-comment",
        'def greeting(name: str) -> str:\n    # Preserve this note when changing the return value.\n    return f"Hello, {name}"\n',
        'return f"Hello, {name}"', 'return f"Hi again, {name}!"',
        "Rui", "Hi again, Rui!",
        'Modify greeting.py only: the greeting for Rui must become "Hi again, Rui!". Keep the note.',
    ),
    Variant(
        "ready-format-call",
        'def greeting(name: str) -> str:\n    return "Hello, {}".format(name)\n',
        'return "Hello, {}".format(name)', 'return "Ready for work, {}!".format(name)',
        "Sana", "Ready for work, Sana!",
        'Have greeting.py return "Ready for work, Sana!" for Sana without changing test_greeting.py.',
    ),
    Variant(
        "great-day-salutation",
        'def greeting(name: str) -> str:\n    salutation = "Hello, "\n    return f"{salutation}{name}"\n',
        'salutation = "Hello, "', 'salutation = "Have a great day, "',
        "Emil", "Have a great day, Emil",
        'Update the salutation in greeting.py to produce "Have a great day, Emil" for Emil, then run pytest.',
    ),
    Variant(
        "wonderful-branch",
        'def greeting(name: str) -> str:\n    if not name:\n        return "Hello, stranger"\n    return f"Hello, {name}"\n',
        'return f"Hello, {name}"', 'return f"Wonderful day, {name}!"',
        "Pia", "Wonderful day, Pia!",
        'For a named caller such as Pia, return "Wonderful day, Pia!" from greeting.py; preserve the empty-name behavior.',
        "Hello, stranger",
    ),
    Variant(
        "from-here-concatenation",
        'def greeting(name: str) -> str:\n    return "Hello, " + name\n',
        'return "Hello, " + name', 'return "Greetings from here, " + name',
        "Uma", "Greetings from here, Uma",
        'Change the greeting.py result for Uma to "Greetings from here, Uma" and check the supplied tests.',
    ),
    Variant(
        "onward-uppercase",
        'def greeting(name: str) -> str:\n    return f"Hello, {name.upper()}"\n',
        'return f"Hello, {name.upper()}"', 'return f"Onward, {name.upper()}!"',
        "jules", "Onward, JULES!",
        'Keep the uppercase-name behavior, but make greeting.py return "Onward, JULES!" for jules. Validate.',
    ),
)


def write_once(path: Path, data: bytes) -> None:
    if path.exists():
        if path.read_bytes() != data:
            raise ValueError(f"Existing artifact differs: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def source_for(variant: Variant) -> Path:
    source = SOURCES / variant.slug
    expected_test = (
        "from greeting import greeting\n\n"
        "def test_greeting_target() -> None:\n"
        f"    assert greeting({variant.name!r}) == {variant.expected!r}\n"
    )
    if variant.fallback is not None:
        expected_test += (
            "\ndef test_greeting_preserves_empty_name() -> None:\n"
            f"    assert greeting('') == {variant.fallback!r}\n"
        )
    if not source.exists():
        shutil.copytree(BASE, source, ignore=shutil.ignore_patterns("__pycache__", ".pytest_cache", "*.pyc"))
        (source / "greeting.py").write_text(variant.source, encoding="utf-8", newline="")
        (source / "test_greeting.py").write_text(expected_test, encoding="utf-8", newline="")
    elif (
        (source / "greeting.py").read_text("utf-8") != variant.source
        or (source / "test_greeting.py").read_text("utf-8") != expected_test
    ):
        raise ValueError(f"Existing source snapshot differs: {source}")
    return source


def record_call(messages: list[TrajectoryMessage], command: str, result) -> None:
    messages.append(
        TrajectoryMessage(
            role="assistant",
            tool_calls=(ToolCallRecord(name="exec_command", arguments={"cmd": command}),),
        )
    )
    messages.append(
        TrajectoryMessage(role="tool", tool_name="exec_command", content=result.envelope())
    )


def collect(assignment, variant: Variant) -> None:
    source = ROOT / assignment.authorized_fixture_root
    if inspect_fixture(assignment.task.fixture_id, source).revision != assignment.task.fixture_revision:
        raise ValueError(f"Pinned source revision changed: {source}")
    workspace = WORKSPACES / variant.slug
    if workspace.exists():
        raise FileExistsError(f"Unrecorded workspace already exists: {workspace}")
    workspace.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, workspace)
    if inspect_fixture(assignment.task.fixture_id, workspace).revision != assignment.task.fixture_revision:
        raise ValueError("Disposable copy differs from pinned source")

    executor = ConstrainedPowerShellExecutor()
    messages = [TrajectoryMessage(role="user", content=assignment.task.task)]
    read_command = (
        "Get-Content -Raw greeting.py" if len(messages) and VARIANTS.index(variant) % 2
        else "Get-Content -LiteralPath greeting.py -Raw"
    )
    messages.append(TrajectoryMessage(role="assistant", content="Inspect greeting.py before editing."))
    read = executor.execute(read_command, workspace)
    record_call(messages, read_command, read)
    if read.exit_code != 0 or read.output != variant.source:
        raise ValueError("Observed read did not match the pinned source")

    quote = "'" if "'" not in variant.old + variant.new else '"'
    edit_command = (
        "$text = Get-Content -LiteralPath greeting.py -Raw; "
        f"$updated = $text.Replace({quote}{variant.old}{quote}, {quote}{variant.new}{quote}); "
        "Set-Content -LiteralPath greeting.py -Value $updated"
    )
    messages.append(TrajectoryMessage(role="assistant", content="Replace the target expression without rewriting the file."))
    edit = executor.execute(edit_command, workspace)
    record_call(messages, edit_command, edit)
    if edit.exit_code != 0 or not edit.authorized:
        raise ValueError(f"Guarded edit failed: {edit.output}")
    expected_source = variant.source.replace(variant.old, variant.new)
    if (workspace / "greeting.py").read_text("utf-8") != expected_source:
        raise ValueError("Edited file does not match the intended local replacement")
    if (workspace / "test_greeting.py").read_bytes() != (source / "test_greeting.py").read_bytes():
        raise ValueError("The validator file was modified")

    messages.append(TrajectoryMessage(role="assistant", content="Run the supplied test."))
    tested = executor.execute("pytest -q", workspace)
    record_call(messages, "pytest -q", tested)
    if tested.exit_code != 0 or "passed" not in tested.output:
        raise ValueError(f"Supplied test failed: {tested.output}")
    independent = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "test_greeting.py"],
        cwd=workspace, capture_output=True, text=True, check=False, timeout=60,
    )
    if independent.returncode != 0:
        raise ValueError(f"Independent validator failed: {independent.stdout}{independent.stderr}")
    messages.append(TrajectoryMessage(role="assistant", content="Edited only greeting.py; the supplied test passed."))
    trajectory = TeacherTrajectory(
        trajectory_id=assignment.trajectory_id,
        task_id=assignment.task.task_id,
        split=DatasetSplit.TRAIN,
        teacher_model=assignment.teacher_model,
        teacher_skill_version=assignment.teacher_skill_version,
        task=assignment.task.task,
        messages=tuple(messages),
        source_revision=assignment.task.fixture_revision,
        tags=("cmd-only-contract", "guarded-replace", "train-edit-variation"),
    )
    append_teacher_trajectory(assignment, trajectory, artifact_root=ROOT)
    print(f"{variant.slug}: appended; 3/3 observed tool calls succeeded; independent pytest passed", flush=True)


def main() -> None:
    if Path.cwd().resolve() != ROOT.resolve():
        raise ValueError("Run from the repository root")
    base_before = inspect_fixture("train-greeting-good-day-contract-v1", BASE).revision
    if base_before != BASE_REVISION:
        raise ValueError("Original train fixture differs from the authorized source")
    previous = TeacherCollectionPlan.model_validate_json(OLD_PLAN.read_bytes())
    if previous.student_target is None or previous.harness_profile is None:
        raise ValueError("Pinned Student target and HarnessProfile are required")
    normalized_target = previous.student_target.model_dump(mode="json")
    normalized_target["hardware"]["path"] = normalized_target["hardware"]["path"].replace("\\", "/")
    normalized_target["recommendation"]["path"] = normalized_target["recommendation"]["path"].replace("\\", "/")
    student_target = StudentTarget.model_validate(normalized_target)
    verify_student_target(student_target, artifact_root=ROOT)
    verify_harness_profile(previous.harness_profile, artifact_root=ROOT)

    tasks = []
    roots = {}
    trajectory_ids = {}
    for index, variant in enumerate(VARIANTS, start=1):
        source = source_for(variant)
        fixture_id = f"train-edit-005-{variant.slug}-v1"
        revision = inspect_fixture(fixture_id, source).revision
        task_id = f"train-edit-005-{index:02d}-{variant.slug}"
        task = TaskSpec(
            task_id=task_id,
            fixture_id=fixture_id,
            fixture_family_id="train-greeting-edit-005",
            fixture_revision=revision,
            defect_family=variant.slug,
            split=DatasetSplit.TRAIN,
            category=TaskCategory.SINGLE_FILE_CHANGE,
            difficulty=TaskDifficulty.EASY,
            task=variant.task,
            knowledge_distance=0.2,
            logical_arrival=timedelta(0),
            time_limit=timedelta(minutes=3),
            allowed_tools=("exec_command",),
            expected_changes=(ExpectedChange(path="greeting.py", operation=ChangeOperation.MODIFY),),
            validation_ids=("pytest:test_greeting.py",),
        )
        tasks.append(task)
        roots[fixture_id] = source.relative_to(ROOT).as_posix()
        trajectory_ids[task_id] = f"train-edit-005-{index:02d}-trajectory-001"

    plan = build_teacher_collection_plan(
        plan_id=PLAN_ID,
        tasks=tasks,
        teacher_model=previous.assignments[0].teacher_model,
        teacher_skill_version="0.2.0",
        fixture_roots=roots,
        destination_jsonl=RAW_RELATIVE,
        trajectory_ids=trajectory_ids,
        student_target=student_target,
        harness_profile=previous.harness_profile,
    )
    plan_bytes = (plan.model_dump_json(indent=2) + "\n").encode("utf-8")
    write_once(PLAN_DIR / "collection-plan.json", plan_bytes)
    manifest = PublishedTeacherCollectionPlan(
        schema_version=plan.schema_version,
        plan_id=PLAN_ID,
        byte_count=len(plan_bytes),
        sha256=sha256(plan_bytes).hexdigest(),
        plan=plan,
    )
    write_once(
        PLAN_DIR / "manifest.json",
        (manifest.model_dump_json(indent=2) + "\n").encode("utf-8"),
    )
    load_teacher_collection_plan(PLAN_DIR)
    existing = load_jsonl(RAW, TeacherTrajectory) if RAW.exists() else ()
    existing_ids = {item.trajectory_id for item in existing}
    for assignment, variant in zip(plan.assignments, VARIANTS, strict=True):
        if assignment.trajectory_id not in existing_ids:
            collect(assignment, variant)
    if inspect_fixture("train-greeting-good-day-contract-v1", BASE).revision != BASE_REVISION:
        raise ValueError("Original train fixture changed during collection")
    print(f"complete: {len(load_jsonl(RAW, TeacherTrajectory))} train records", flush=True)


if __name__ == "__main__":
    main()
