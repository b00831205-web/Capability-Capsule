"""Read-only preparation for a separately authorized, train-only free-running probe.

This module never copies fixtures, loads a model, or writes artifacts. It keeps
seen training cases out of the validation-only CodingEvaluationSuite API.
"""

import argparse
import json
from hashlib import sha256
from pathlib import Path

from capability_capsule.eval.fixture_provenance import inspect_fixture

ROOT = Path(__file__).resolve().parents[1]
RUN_ID = "stage1-codex-qwen35-2b-011-mix006007-28step"
CHECKPOINT = ROOT / "runs/training" / RUN_ID / "checkpoint.json"
PUBLICATION = ROOT / "datasets/teacher/published/stage1-codex-powershell-literal006-fstring007-mix-v1"
SFT = ROOT / "artifacts/sft/stage1-codex-powershell-literal006-fstring007-mix-v1-qwen35-2b-v1"
PROPOSED_OUTPUT = ROOT / "runs/evaluation" / (RUN_ID + "-train-fit-diagnostic-v1")
PLAN_IDS = ("stage1-codex-powershell-literal-edit-006", "stage1-codex-powershell-fstring-edit-007")
PINS = {
    CHECKPOINT: "30b10ef7364c6b1ea7cb94a72d2d064b84318a5969d078a2b4e0598ed44bba52",
    PUBLICATION / "manifest.json": "2739762194576d51c89900dc3f7aceade571e1b4ed4a7359f6134198432faa60",
    SFT / "manifest.json": "5d8747cbf4ea5d1f023ecfd12694ec7b839af49701b7b59b6c9c06e27ef823cc",
    ROOT / "src/capability_capsule/eval/coding_checkpoint.py": "c1b3ec738da80c59da83411adc0884223b36b372041b45390ffa0f9834381356",
}


def read(path):
    return json.loads(path.read_text("utf-8"))


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def records(path):
    return [json.loads(line) for line in path.read_text("utf-8").splitlines() if line.strip()]


def verify_bundle(directory):
    manifest = read(directory / "manifest.json")
    for artifact in manifest["artifacts"]:
        path = directory / artifact["filename"]
        if path.parent != directory or digest(path) != artifact["sha256"]:
            raise ValueError("Artifact digest mismatch")
        if len(records(path)) != artifact["record_count"]:
            raise ValueError("Artifact record count mismatch")
    return manifest


def model_messages(contract, task):
    # Never include recorded Teacher answers, tool results, or reference edits.
    return [{"role": "system", "content": contract["system_prompt"]},
            {"role": "user", "content": task["task"]}]


def check_case(record, assignment, example):
    task = assignment["task"]
    if record["split"] != "train" or task["split"] != "train":
        raise ValueError("Train-fit diagnosis requires original train labels")
    if (record["task_id"] != task["task_id"] or record["task"] != task["task"]
            or record["trajectory_id"] != assignment["trajectory_id"]
            or example["source_trajectory_id"] != record["trajectory_id"]
            or example["source_revision"] != record["source_revision"]
            or record["source_revision"] != task["fixture_revision"]):
        raise ValueError("Plan/publication/SFT correspondence mismatch")
    source = (ROOT / assignment["authorized_fixture_root"]).resolve(strict=True)
    source.relative_to((ROOT / "artifacts/teacher-fixtures").resolve())
    if inspect_fixture(task["fixture_id"], source).revision != task["fixture_revision"]:
        raise ValueError("Original fixture changed")
    if set(task["allowed_tools"]) != {"exec_command"}:
        raise ValueError("Unexpected tool contract")
    return {
        "task_id": task["task_id"], "trajectory_id": record["trajectory_id"],
        "split": "train", "seen_during_training": True,
        "task": task["task"], "fixture_id": task["fixture_id"],
        "fixture_root": source.relative_to(ROOT).as_posix(),
        "fixture_revision": task["fixture_revision"],
        "source_sha256": digest(source / "greeting.py"),
        "test_sha256": digest(source / "test_greeting.py"),
        "original_validation_ids": task["validation_ids"],
        "time_limit": task["time_limit"],
    }


def build_plan():
    for path, expected in PINS.items():
        if digest(path) != expected:
            raise ValueError(f"Pinned input changed: {path.name}")
    checkpoint = read(CHECKPOINT)
    for filename, key in (("adapter_model.safetensors", "adapter_model_sha256"),
                          ("adapter_config.json", "adapter_config_sha256")):
        if digest(CHECKPOINT.parent / checkpoint["adapter_directory"] / filename) != checkpoint[key]:
            raise ValueError("Checkpoint adapter changed")
    verify_bundle(PUBLICATION)
    manifest = verify_bundle(SFT)
    if manifest["source_dataset_digest"] != PINS[PUBLICATION / "manifest.json"]:
        raise ValueError("SFT publication mismatch")
    source_records = records(PUBLICATION / "train.jsonl")
    examples = records(SFT / "train.jsonl")
    if len(source_records) != 14 or len(examples) != 14:
        raise ValueError("Expected exactly 14 seen training cases")
    assignments = {}
    plan_hashes = {}
    for plan_id in PLAN_IDS:
        path = ROOT / "plans/teacher" / plan_id / "collection-plan.json"
        plan_hashes[path.relative_to(ROOT).as_posix()] = digest(path)
        for assignment in read(path)["assignments"]:
            task_id = assignment["task"]["task_id"]
            if task_id in assignments:
                raise ValueError("Duplicate task ID")
            assignments[task_id] = assignment
    cases = [check_case(record, assignments[record["task_id"]], example)
             for record, example in zip(source_records, examples, strict=True)]
    if len({case["task_id"] for case in cases}) != 14 or len(assignments) != 14:
        raise ValueError("Task coverage mismatch")
    return {
        "diagnostic_id": PROPOSED_OUTPUT.name, "checkpoint_id": RUN_ID,
        "checkpoint_sha256": PINS[CHECKPOINT], "checkpoint_step": checkpoint["step"],
        "proposed_output": PROPOSED_OUTPUT.relative_to(ROOT).as_posix(),
        "output_already_exists": PROPOSED_OUTPUT.exists(),
        "inference_started": False, "artifact_writes_started": False,
        "scope": "Train-only free-running fit diagnostic, not unseen validation or promotion evidence",
        "case_count": 14, "groups": {PLAN_IDS[0]: 8, PLAN_IDS[1]: 6},
        "repetitions": 1, "max_new_tokens": 256, "max_tool_calls": 4,
        "max_generated_turns": 5, "do_sample": False, "enable_thinking": False,
        "contract": manifest["chat_contract"], "contract_sha256": manifest["chat_contract_sha256"],
        "input_hashes": {**{p.relative_to(ROOT).as_posix(): d for p, d in PINS.items()}, **plan_hashes},
        "cases": cases,
        "grading": ["Original supplied tests pass", "Original reference edit matches",
                    "Supplied test bytes unchanged", "Final response reached without invalid calls"],
        "additional_metrics": ["Actual source changed", "Denied command count", "Model invoked pytest",
                               "Behavior pass independent of exact source", "Time-bounded success"],
        "intent_assessment": "Review stored generated commands; keep human judgment separate from automated grades",
        "recording": "Separate train diagnostic reports; never append these cases as validation ledger points",
        "prerequisite_for_execution": "Separately authorize new output directory and disposable train copies",
        "planning_updates": {
            "MVP_ROADMAP.md": "Record completed 011 training and 0/3 + 0/6 evaluation, paired 010 4/6; next is train-fit diagnosis",
            "PENDING_CONFIGURATION_OPTIMIZATIONS.md": "Track command policy mismatch, literal punctuation rejection, recovery weakness, and unconfirmed undertraining; 56-step trial remains proposed",
        },
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full-plan", action="store_true")
    args = parser.parse_args()
    plan = build_plan()
    if not args.full_plan:
        plan = {key: plan[key] for key in (
            "diagnostic_id", "checkpoint_step", "case_count", "groups", "output_already_exists",
            "inference_started", "artifact_writes_started", "scope", "proposed_output")}
    print(json.dumps(plan, ensure_ascii=False, indent=2))
