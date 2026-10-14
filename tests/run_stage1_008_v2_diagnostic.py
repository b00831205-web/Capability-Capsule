"""Prompt-only closed-loop diagnostic; never rewrite existing experiments."""

import json
import shutil
import sys
from datetime import UTC, datetime
from hashlib import sha256

import run_stage1_008_evaluation as baseline
from run_stage1_007_evaluation import ProgressGenerator, digest, save, TOOLS, EXECUTOR
from capability_capsule.eval.coding_checkpoint import (
    CodingEvaluationCase, CodingEvaluationSuite, ConstrainedPowerShellExecutor,
    TransformersPeftTurnGenerator, copy_verified_evaluation_fixture,
    evaluate_coding_case, load_coding_evaluation_suite,
)
from capability_capsule.eval.fixture_provenance import inspect_fixture
from capability_capsule.eval.jsonl import append_jsonl
from capability_capsule.eval.learning_curve_ledger import append_learning_curve_point
from capability_capsule.eval.learning_rate import LearningCurvePoint
from capability_capsule.training.sft import SFTExportManifest

ROOT = baseline.ROOT
TRAINING = baseline.TRAINING
BASELINE = baseline.RUN
RUN_ID = baseline.RUN_ID + "-v2-diagnostic"
RUN = ROOT / "runs/evaluation" / RUN_ID
GROUPS = {"fixed-v2": "fixed-v3", "heldout-v2": "heldout-v1"}


def protected_hashes():
    hashes = baseline.protected_hashes()
    for path in BASELINE.rglob("*"):
        if path.is_file() and not any(p in {"__pycache__", ".pytest_cache"} for p in path.parts):
            hashes[path.relative_to(ROOT).as_posix()] = digest(path)
    path = ROOT / "plans/evaluation/stage1-codex-validation-v2/evaluation-suite.json"
    hashes[path.relative_to(ROOT).as_posix()] = digest(path)
    return hashes


def prepare():
    if RUN.exists():
        raise FileExistsError(RUN)
    sft_path = baseline.control.SFT / "manifest.json"
    sft = SFTExportManifest.model_validate_json(sft_path.read_bytes())
    contract = sft.chat_contract
    v2 = load_coding_evaluation_suite(ROOT / "plans/evaluation/stage1-codex-validation-v2/evaluation-suite.json")
    if contract is None or contract.system_prompt != v2.system_prompt or tuple(contract.tools) != TOOLS:
        raise ValueError("v2 diagnostic must match the persisted SFT contract exactly")
    before = protected_hashes()
    old_prepared = json.loads((BASELINE / "prepared.json").read_text("utf-8"))
    RUN.mkdir(parents=True)
    save(RUN / "protected-input-hashes.json", before)
    groups, sources = {}, {}
    for group, old_group in GROUPS.items():
        old_directory = BASELINE / old_group
        old_suite = load_coding_evaluation_suite(old_directory / "evaluation-suite.json")
        suite = CodingEvaluationSuite.model_validate({**old_suite.model_dump(mode="json"),
            "evaluation_suite_id": old_suite.evaluation_suite_id + "-v2-training-contract-diagnostic",
            "system_prompt": contract.system_prompt})
        if suite.cases != old_suite.cases or suite.identity().evaluation_suite_digest != old_suite.identity().evaluation_suite_digest:
            raise ValueError("Cases or validators changed")
        directory = RUN / group
        directory.mkdir()
        save(directory / "evaluation-suite.json", suite.model_dump(mode="json"))
        for case in suite.cases:
            relative = old_prepared["fixture_sources"][case.case_id]
            copy_case = CodingEvaluationCase.model_validate({**case.model_dump(mode="json"), "fixture_root": relative})
            copy_verified_evaluation_fixture(copy_case, artifact_root=ROOT,
                destination=directory / "workspaces" / case.case_id)
            sources[case.case_id] = relative
        old_manifest = json.loads((old_directory / "run-manifest.json").read_text("utf-8"))
        if old_manifest["checkpoint_sha256"] != digest(TRAINING / "checkpoint.json") or old_manifest["executor_sha256"] != digest(EXECUTOR):
            raise ValueError("Checkpoint or executor differs from baseline")
        groups[group] = {"suite_sha256": digest(directory / "evaluation-suite.json"),
            "case_digest": suite.identity().evaluation_suite_digest,
            "baseline_run": old_directory.relative_to(ROOT).as_posix()}
    save(RUN / "prepared.json", {"prepared_at": datetime.now(UTC).isoformat(),
        "checkpoint_sha256": digest(TRAINING / "checkpoint.json"), "executor_sha256": digest(EXECUTOR),
        "tools_sha256": sha256(json.dumps(TOOLS, sort_keys=True).encode()).hexdigest(),
        "system_prompt_sha256": sha256(contract.system_prompt.encode()).hexdigest(),
        "sft_manifest_sha256": digest(sft_path), "chat_contract_sha256": contract.sha256(),
        "fixture_sources": sources, "groups": groups,
        "scope": "Prompt-only diagnostic of the same 008 checkpoint and five repeated cases; not fresh heldout or replacement acceptance evidence."})
    print("Pinned identical five cases; only system prompt changes from v3 to SFT v2.", flush=True)


def evaluate():
    prepared = json.loads((RUN / "prepared.json").read_text("utf-8"))
    before = json.loads((RUN / "protected-input-hashes.json").read_text("utf-8"))
    if protected_hashes() != before:
        raise ValueError("Protected inputs changed after preparation")
    for group, pin in prepared["groups"].items():
        directory = RUN / group
        if (directory / "case-results.jsonl").exists():
            raise FileExistsError("Refusing to rerun or overwrite results")
        if digest(directory / "evaluation-suite.json") != pin["suite_sha256"]:
            raise ValueError("Suite changed after preparation")
        for case in load_coding_evaluation_suite(directory / "evaluation-suite.json").cases:
            if inspect_fixture(case.task.fixture_id, directory / "workspaces" / case.case_id).revision != case.task.fixture_revision:
                raise ValueError("Workspace changed after preparation")
    generator = ProgressGenerator(TransformersPeftTurnGenerator(TRAINING / "checkpoint.json", max_new_tokens=256))
    training = json.loads((TRAINING / "training-run.json").read_text("utf-8"))
    checkpoint = json.loads((TRAINING / "checkpoint.json").read_text("utf-8"))
    summaries = {}
    for group, pin in prepared["groups"].items():
        directory = RUN / group
        suite = load_coding_evaluation_suite(directory / "evaluation-suite.json")
        (directory / "case-reports").mkdir()
        results = []
        for position, case in enumerate(suite.cases, 1):
            print(f"Starting {group}: {case.case_id}", flush=True)
            outcome = evaluate_coding_case(case, workspace=directory / "workspaces" / case.case_id,
                generator=generator, executor=ConstrainedPowerShellExecutor(), system_prompt=suite.system_prompt,
                tools=TOOLS, experiment_id=RUN_ID, run_id=f"{RUN_ID}:{group}:{case.case_id}",
                model_id=checkpoint["output_capsule_id"], hardware_id=training["hardware_id"],
                cycle_position=position, max_tool_rounds=4)
            append_jsonl(directory / "case-results.jsonl", outcome.result)
            save(directory / "case-reports" / f"{case.case_id}.json", {
                "result": outcome.result.model_dump(mode="json"), "completions": list(outcome.completions),
                "tool_results": [r.model_dump(mode="json") for r in outcome.tool_results],
                "validators": [v.model_dump(mode="json") for v in outcome.validators]})
            results.append(outcome.result)
            print(f"success={outcome.result.success}, invalid={outcome.result.invalid_tool_call_count}", flush=True)
        count = sum(r.success for r in results)
        stats = training["dataset_stats"]["train"]
        point = LearningCurvePoint(checkpoint_id=f"{checkpoint['run_id']}-step-{checkpoint['step']}",
            training_run_id=checkpoint["run_id"], base_model_id=checkpoint["base_model_id"],
            capability_id=suite.capability_id, evaluation_suite_id=suite.evaluation_suite_id,
            evaluation_suite_digest=suite.identity().evaluation_suite_digest,
            task_family_id="greeting-fixed-prompt-diagnostic" if group == "fixed-v2" else "greeting-heldout-prompt-diagnostic",
            evaluation_split="validation", cumulative_trajectory_count=stats["trajectory_count"],
            cumulative_token_count=stats["exact_token_count"], tokenizer_id=stats["tokenizer_id"],
            success_rate=count / len(results))
        append_learning_curve_point(directory / "learning-curve.jsonl", point)
        old_manifest = json.loads((ROOT / pin["baseline_run"] / "run-manifest.json").read_text("utf-8"))
        manifest = {**pin, "checkpoint": (TRAINING / "checkpoint.json").relative_to(ROOT).as_posix(),
            "checkpoint_sha256": prepared["checkpoint_sha256"], "executor_sha256": prepared["executor_sha256"],
            "tools_sha256": prepared["tools_sha256"], "system_prompt_sha256": prepared["system_prompt_sha256"],
            "max_new_tokens": 256, "max_tool_rounds": 4, "do_sample": False,
            "case_count": len(results), "success_count": count,
            "time_bounded_success_count": sum(r.time_bounded_success for r in results),
            "baseline_success_count": old_manifest["success_count"],
            "scope": prepared["scope"], "prepared_at": prepared["prepared_at"], "completed_at": datetime.now(UTC).isoformat()}
        save(directory / "run-manifest.json", manifest)
        summaries[group] = manifest
    after = protected_hashes()
    changed = sorted(p for p in set(before) | set(after) if before.get(p) != after.get(p))
    save(RUN / "input-preservation.json", {"unchanged": not changed, "protected_file_count": len(before), "changed_paths": changed})
    if changed:
        raise ValueError("Protected inputs changed")
    save(RUN / "evaluation-summary.json", summaries)


if __name__ == "__main__":
    if sys.argv[1:] == ["--prepare-only"]:
        prepare()
    elif sys.argv[1:] == ["--evaluate-prepared"]:
        evaluate()
    elif not sys.argv[1:]:
        prepare()
        evaluate()
    else:
        raise ValueError("Use --prepare-only or --evaluate-prepared")
