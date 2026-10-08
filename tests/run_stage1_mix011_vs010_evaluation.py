"""One-shot 011 fixed regression and preregistered 011/010 frozen-six comparison."""

import json
import shutil
import sys
from datetime import UTC, datetime
from hashlib import sha256

import train_stage1_mix011_28step as training
from run_stage1_007_evaluation import ProgressGenerator
from capability_capsule.eval.coding_checkpoint import (
    CodingEvaluationCase, ConstrainedPowerShellExecutor, TransformersPeftTurnGenerator,
    copy_verified_evaluation_fixture, evaluate_coding_case, load_coding_evaluation_suite,
)
from capability_capsule.eval.fixture_provenance import inspect_fixture
from capability_capsule.eval.jsonl import append_jsonl, load_jsonl
from capability_capsule.eval.records import CaseResult
from capability_capsule.eval.learning_curve_ledger import append_learning_curve_point
from capability_capsule.eval.learning_rate import LearningCurvePoint

ROOT = training.ROOT
RUN = ROOT / "runs/evaluation/stage1-codex-qwen35-2b-011-mix006007-28step-vs010-frozen-v1"
FROZEN = training.frozen.OUTPUT
MODELS = {"011": training.RUN, "010": training.frozen.BASELINE.parent}
GROUPS = {"011": ("fixed-v2", "frozen-six"), "010": ("frozen-six",)}
digest, save = training.digest, training.save


def read(path):
    return json.loads(path.read_text("utf-8"))


def protected_hashes():
    hashes = training.protected_hashes()
    for directory in MODELS.values():
        for path in directory.rglob("*"):
            if path.is_file() and not any(p in {"__pycache__", ".pytest_cache"} for p in path.parts):
                hashes[path.relative_to(ROOT).as_posix()] = digest(path)
    prefix = RUN.relative_to(ROOT).as_posix() + "/"
    return {path:value for path,value in hashes.items() if not path.startswith(prefix)}


def verify_inputs():
    training.prepare()
    training.verify_preservation()
    for directory in MODELS.values():
        reload = read(directory / "adapter-reload-validation.json")
        if not all(reload.get(k) for k in ("independent_process", "base_model_revision_verified", "adapter_hashes_verified")):
            raise ValueError("Both independently reloaded checkpoints are required")
        checkpoint = read(directory / "checkpoint.json")
        adapter = directory / checkpoint["adapter_directory"]
        for name, key in (("adapter_model.safetensors", "adapter_model_sha256"), ("adapter_config.json", "adapter_config_sha256")):
            if digest(adapter / name) != checkpoint[key]:
                raise ValueError("Checkpoint adapter changed")


def prepare():
    if RUN.exists():
        raise FileExistsError("Refusing to overwrite evaluation preparation")
    verify_inputs()
    six = load_coding_evaluation_suite(FROZEN / "evaluation-suite.json")
    # First use means no previous model result for these six, not merely a new directory name.
    wanted = {case.case_id for case in six.cases}
    for path in (ROOT / "runs/evaluation").rglob("case-results.jsonl"):
        if wanted & {r.case_id for r in load_jsonl(path, CaseResult)}:
            raise ValueError("Frozen cases already consumed by model evaluation")
    before = protected_hashes()
    RUN.mkdir(parents=True)
    save(RUN / "protected-input-hashes.json", before)
    policies = read(FROZEN / "experiment-plan.json")
    sources, copies, suites = {}, {}, {}
    for side, groups in GROUPS.items():
        for group in groups:
            directory = RUN / side / group
            directory.mkdir(parents=True)
            suite_path = training.frozen.FIXED if group == "fixed-v2" else FROZEN / "evaluation-suite.json"
            shutil.copyfile(suite_path, directory / "evaluation-suite.json")
            suite = load_coding_evaluation_suite(suite_path)
            for case in suite.cases:
                source = ROOT / case.fixture_root
                if inspect_fixture(case.task.fixture_id, source).revision != case.task.fixture_revision:
                    if group != "fixed-v2":
                        raise ValueError("Frozen-six source changed")
                    source = ROOT / "runs/evaluation/stage1-codex-qwen35-2b-006-turns-v2-v2/workspaces" / case.case_id
                effective = CodingEvaluationCase.model_validate({**case.model_dump(mode="json"),
                    "fixture_root": source.relative_to(ROOT).as_posix()})
                destination = directory / "workspaces" / case.case_id
                copy_verified_evaluation_fixture(effective, artifact_root=ROOT, destination=destination)
                sources[f"{side}/{group}/{case.case_id}"] = source.relative_to(ROOT).as_posix()
                for path in destination.iterdir():
                    if path.is_file():
                        copies[path.relative_to(RUN).as_posix()] = digest(path)
            suites[f"{side}/{group}"] = {"suite_sha256": digest(directory / "evaluation-suite.json"),
                "case_digest": suite.identity().evaluation_suite_digest, "case_count": len(suite.cases)}
    if before != protected_hashes():
        raise ValueError("Protected input changed while preparing")
    save(RUN / "prepared.json", {"prepared_at": datetime.now(UTC).isoformat(),
        "training_run_ids": {side:read(path / "checkpoint.json")["run_id"] for side,path in MODELS.items()},
        "checkpoint_hashes": {side:digest(path / "checkpoint.json") for side,path in MODELS.items()},
        "frozen_validation_manifest_sha256": training.FREEZE_DIGEST,
        "executor_sha256": policies["executor_sha256"], "chat_contract_sha256": policies["chat_contract_sha256"],
        "max_new_tokens": 256, "max_tool_rounds": 4, "do_sample": False, "enable_thinking": False,
        "system_prompt_sha256": sha256(training.frozen.mix.baseline.contract().system_prompt.encode()).hexdigest(),
        "tools_sha256": sha256(json.dumps(policies["tools"], sort_keys=True).encode()).hexdigest(),
        "groups": suites, "fixture_sources": sources, "initial_workspace_hashes": copies,
        "model_evaluation_authorized": True, "root_planning_edits_authorized": False,
        "first_use_before_model_inference": True,
        "scope": "011 fixed-v2 regression plus same-six 011/010 once each; final checkpoints preselected; no training or existing ledger mutation"})
    print("Prepared 3 fixed-v2 + 6 frozen cases for 011, and identical 6 for 010; no inference yet", flush=True)


def verify_prepared():
    prepared = read(RUN / "prepared.json")
    before = read(RUN / "protected-input-hashes.json")
    current = protected_hashes()
    if any(current.get(path) != pinned for path,pinned in before.items()):
        raise ValueError("Protected inputs changed")
    if digest(FROZEN / "freeze-manifest.json") != prepared["frozen_validation_manifest_sha256"]:
        raise ValueError("Frozen validation manifest changed")
    for side, directory in MODELS.items():
        if digest(directory / "checkpoint.json") != prepared["checkpoint_hashes"][side]:
            raise ValueError("Preselected checkpoint changed")
    for group, pins in prepared["groups"].items():
        if digest(RUN / group / "evaluation-suite.json") != pins["suite_sha256"]:
            raise ValueError("Suite copy changed")
    return prepared


def report_metrics(results, reports):
    behavior = sum(all(v["passed"] for v in r["validators"] if v["validator_id"].startswith(("pytest:", "python-assert:"))) for r in reports)
    dynamic_reports = [r for r in reports if any(v["validator_id"].startswith("python-assert:") for v in r["validators"])]
    multi_reports = [r for r in dynamic_reports if sum(v["validator_id"].startswith("python-assert:") for v in r["validators"]) >= 2]
    return {"case_count":len(results), "success_count":sum(r.success for r in results), "behavior_pass_count":behavior,
        "time_bounded_success_count":sum(r.time_bounded_success for r in results),
        "duration_ms_by_case":{r.case_id:r.duration_ms for r in results},
        "invalid_tool_call_count":sum(r.invalid_tool_call_count for r in results),
        "denied_command_count":sum(not t["authorized"] for r in reports for t in r["tool_results"]),
        "python_call_case_count":len(dynamic_reports),
        "multi_input_case_count":len(multi_reports),
        "multi_input_pass_count":sum(all(v["passed"] for v in r["validators"] if v["validator_id"].startswith("python-assert:")) for r in multi_reports),
        "strict_source_pass_count":sum(any(v["validator_id"] == "exact-content:greeting.py" and v["passed"] for v in r["validators"]) for r in reports)}


def evaluate(side):
    prepared = verify_prepared()
    attempt = RUN / side / "inference-attempt.json"
    if attempt.exists():
        raise FileExistsError("Refusing to rerun a checkpoint after any inference attempt")
    for group in GROUPS[side]:
        directory = RUN / side / group
        if (directory / "case-results.jsonl").exists():
            raise FileExistsError("Existing results cannot be overwritten")
        suite = load_coding_evaluation_suite(directory / "evaluation-suite.json")
        for case in suite.cases:
            if inspect_fixture(case.task.fixture_id, directory / "workspaces" / case.case_id).revision != case.task.fixture_revision:
                raise ValueError("Inference copy is no longer the original")
    save(attempt, {"side":side, "started_at":datetime.now(UTC).isoformat(),
        "checkpoint_sha256": prepared["checkpoint_hashes"][side], "one_shot":True})
    print(f"Loading preselected checkpoint {side}", flush=True)
    generator = ProgressGenerator(TransformersPeftTurnGenerator(MODELS[side] / "checkpoint.json", max_new_tokens=256))
    manifest, checkpoint = read(MODELS[side] / "training-run.json"), read(MODELS[side] / "checkpoint.json")
    contract = training.frozen.mix.baseline.contract()
    summaries = {}
    for group in GROUPS[side]:
        directory = RUN / side / group
        suite = load_coding_evaluation_suite(directory / "evaluation-suite.json")
        results, reports = [], []
        for position, case in enumerate(suite.cases, 1):
            workspace = directory / "workspaces" / case.case_id
            tests_before = (workspace / "test_greeting.py").read_bytes()
            print(f"Starting {side}/{group}: {case.case_id}", flush=True)
            outcome = evaluate_coding_case(case, workspace=workspace, generator=generator,
                executor=ConstrainedPowerShellExecutor(), system_prompt=suite.system_prompt, tools=list(contract.tools),
                experiment_id=RUN.name, run_id=f"{RUN.name}:{side}:{group}:{case.case_id}",
                model_id=checkpoint["output_capsule_id"], hardware_id=manifest["hardware_id"],
                cycle_position=position, max_tool_rounds=4)
            report = {"result":outcome.result.model_dump(mode="json"), "completions":list(outcome.completions),
                "tool_results":[v.model_dump(mode="json") for v in outcome.tool_results],
                "validators":[v.model_dump(mode="json") for v in outcome.validators],
                "tests_byte_unchanged": (workspace / "test_greeting.py").read_bytes() == tests_before}
            append_jsonl(directory / "case-results.jsonl", outcome.result)
            save(directory / "case-reports" / (case.case_id + ".json"), report)
            results.append(outcome.result)
            reports.append(report)
            print(f"success={outcome.result.success}, invalid={outcome.result.invalid_tool_call_count}, error={outcome.result.error_type}", flush=True)
        summary = {**prepared["groups"][f"{side}/{group}"], **report_metrics(results,reports),
            "checkpoint_sha256":prepared["checkpoint_hashes"][side], "training_run_id":checkpoint["run_id"],
            "prepared_at":prepared["prepared_at"], "completed_at":datetime.now(UTC).isoformat(),
            "max_new_tokens":256, "max_tool_rounds":4, "do_sample":False, "enable_thinking":False,
            "test_files_unchanged":all(r["tests_byte_unchanged"] for r in reports),
            "scope":"Repeated development regression" if group == "fixed-v2" else "Predeclared paired first-use comparison; independence consumed, not checkpoint selection or broad generalization"}
        if group == "frozen-six":
            summary["subgroups"] = {g:report_metrics(results[i:i+2],reports[i:i+2]) for g,i in
                (("fstring",0),("structure",2),("business-transfer",4))}
        stats = manifest["dataset_stats"]["train"]
        append_learning_curve_point(directory / "learning-curve.jsonl", LearningCurvePoint(
            checkpoint_id=f"{checkpoint['run_id']}-step-{checkpoint['step']}", training_run_id=checkpoint["run_id"],
            base_model_id=checkpoint["base_model_id"], capability_id=suite.capability_id,
            evaluation_suite_id=suite.evaluation_suite_id, evaluation_suite_digest=suite.identity().evaluation_suite_digest,
            task_family_id="fixed-v2-regression" if group == "fixed-v2" else "mix011-frozen-six-paired-transfer",
            evaluation_split="validation", cumulative_trajectory_count=stats["trajectory_count"],
            cumulative_token_count=stats["exact_token_count"], tokenizer_id=stats["tokenizer_id"],
            success_rate=summary["success_count"]/len(results)))
        save(directory / "run-manifest.json", summary)
        summaries[group] = summary
    verify_prepared()
    save(RUN / side / "evaluation-summary.json", summaries)
    print(json.dumps({g:{k:v for k,v in s.items() if k in {"case_count","success_count","behavior_pass_count","multi_input_pass_count"}} for g,s in summaries.items()},indent=2),flush=True)


def finalize():
    prepared = verify_prepared()
    summaries = {side:read(RUN / side / "evaluation-summary.json") for side in MODELS}
    # Derive descriptive metrics from persisted reports. A fixed case with one
    # python_call is not a multi-input test. Never alter grades or raw traces.
    for side, groups in summaries.items():
        for group, summary in groups.items():
            directory = RUN / side / group
            results = load_jsonl(directory / "case-results.jsonl", CaseResult)
            reports = [read(directory / "case-reports" / (result.case_id + ".json")) for result in results]
            metrics = report_metrics(results, reports)
            for key in ("case_count", "success_count", "behavior_pass_count", "invalid_tool_call_count"):
                if metrics[key] != summary[key]:
                    raise ValueError("Persisted scores differ from actual case reports")
            summary.update(metrics)
            summary["metric_definitions"] = "Multi-input requires at least two independent python_call gates; single-call fixed case excluded"
            with (directory / "run-manifest.json").open("w", encoding="utf-8") as stream:
                stream.write(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
        with (RUN / side / "evaluation-summary.json").open("w", encoding="utf-8") as stream:
            stream.write(json.dumps(groups, ensure_ascii=False, indent=2) + "\n")
    a, b = summaries["011"]["frozen-six"], summaries["010"]["frozen-six"]
    cases = load_coding_evaluation_suite(FROZEN / "evaluation-suite.json").cases
    paired = []
    for case, variant in zip(cases, training.frozen.VARIANTS, strict=True):
        row = {"case_id":case.case_id, "group":variant.group}
        for side in MODELS:
            record = read(RUN / side / "frozen-six/case-reports" / (case.case_id + ".json"))
            row[side] = {"success":record["result"]["success"],
                "behavior_pass":all(v["passed"] for v in record["validators"] if v["validator_id"].startswith(("pytest:","python-assert:"))),
                "multi_input_pass":all(v["passed"] for v in record["validators"] if v["validator_id"].startswith("python-assert:")),
                "error_type":record["result"]["error_type"]}
        paired.append(row)
    accepted = summaries["011"]["fixed-v2"]["success_count"] == 3 and a["success_count"] == 6 and all(
        summaries["011"][g]["invalid_tool_call_count"] == 0 and summaries["011"][g]["test_files_unchanged"] for g in GROUPS["011"])
    save(RUN / "comparison-summary.json", {"summaries":summaries, "paired_cases":paired,
        "wins_011":sum(r["011"]["success"] and not r["010"]["success"] for r in paired),
        "wins_010":sum(r["010"]["success"] and not r["011"]["success"] for r in paired),
        "strict_success_delta":a["success_count"]-b["success_count"], "predeclared_targets_met":accepted,
        "checkpoint_promoted":False, "interpretation":"Data, exposure and linear-schedule horizon differ; exploratory paired evidence, not single-variable causality or proof against overfitting",
        "heldout_independence_consumed":True, "old_010_fixed_score_is_historical_not_rerun":True})
    save(RUN / "input-preservation.json", {"unchanged":True,
        "protected_file_count":len(read(RUN / "protected-input-hashes.json")), "changed_paths":[],
        "root_planning_files_unchanged":True, "frozen_suite_unchanged":True})
    save(RUN / "results-summary.json", {"groups":{side:{g:f"{s['success_count']}/{s['case_count']}" for g,s in groups.items()} for side,groups in summaries.items()},
        "targets_met":accepted, "task_execution_count":15,
        "recorded_generation_turn_count":sum(len(read(path)["completions"]) for path in RUN.glob("*/*/case-reports/*.json")),
        "reruns":0, "prepared_at":prepared["prepared_at"]})
    print(json.dumps({side:{g:f"{s['success_count']}/{s['case_count']}" for g,s in groups.items()} for side,groups in summaries.items()},indent=2),flush=True)


if __name__ == "__main__":
    commands = {"--prepare-only":prepare, "--evaluate-011":lambda:evaluate("011"),
        "--evaluate-010":lambda:evaluate("010"), "--finalize":finalize}
    if len(sys.argv) != 2 or sys.argv[1] not in commands:
        raise ValueError("Use --prepare-only, --evaluate-011, --evaluate-010 or --finalize")
    commands[sys.argv[1]]()
