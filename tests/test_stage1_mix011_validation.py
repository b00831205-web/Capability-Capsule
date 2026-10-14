"""Frozen first-use validation plan, multi-input gates and real preflight evidence."""

import ast
import json
import shutil

import pytest
import freeze_stage1_mix011_validation as freeze
from capability_capsule.eval.coding_checkpoint import CodingEvaluationCase, load_coding_evaluation_suite, run_coding_validators
from capability_capsule.eval.fixture_provenance import inspect_fixture


def read(name):
    return json.loads((freeze.OUTPUT / name).read_text("utf-8"))


def test_six_first_use_validation_cases_are_partitioned_and_source_pinned():
    suite = load_coding_evaluation_suite(freeze.OUTPUT / "evaluation-suite.json")
    assert len(suite.cases) == len(freeze.VARIANTS) == 6
    assert len({c.case_id for c in suite.cases}) == len({c.task.fixture_revision for c in suite.cases}) == 6
    assert read("preflight-summary.json")["groups"] == {"fstring":2, "structure":2, "business-transfer":2}
    for case, variant in zip(suite.cases, freeze.VARIANTS, strict=True):
        source = freeze.ROOT / case.fixture_root
        assert case.task.split.value == "validation"
        assert case.task.allowed_tools == ("exec_command",)
        assert inspect_fixture(case.task.fixture_id, source).revision == case.task.fixture_revision
        assert {p.name for p in source.iterdir()} == {"greeting.py", "test_greeting.py"}
        assert (source / "greeting.py").read_text("utf-8") == variant.source
        assert (source / "test_greeting.py").read_text("utf-8") == freeze.tests_for(variant)
        assert len(case.validators) == len(variant.checks) + 3
        assert [v.validator_id for v in case.validators] == list(case.task.validation_ids)
        assert all(text not in case.task.task for text in (".Replace(", "Set-Content", "$source"))
    assert [v.function for v in freeze.VARIANTS[-2:]] == ["status_line", "audit_line"]
    assert all(len(args) == 2 for variant in freeze.VARIANTS[-2:] for args, _ in variant.checks)


def test_all_originals_fail_and_references_pass_with_actual_authorized_calls():
    summary = read("preflight-summary.json")
    assert summary["original_failures"] == summary["reference_passes"] == 6
    assert summary["actual_reference_tool_calls"] == 18
    suite = load_coding_evaluation_suite(freeze.OUTPUT / "evaluation-suite.json")
    for case, variant in zip(suite.cases, freeze.VARIANTS, strict=True):
        report = read(f"preflight/{variant.slug}/report.json")
        assert report["case_id"] == case.case_id and report["source_snapshot_unchanged"]
        assert not report["original"][0]["passed"] and not report["original"][1]["passed"]
        assert report["original"][2]["passed"]
        assert all(v["passed"] for v in report["reference"])
        calls = report["actual_reference_calls"]
        assert len(calls) == 3
        assert all(c["result"]["authorized"] and c["result"]["exit_code"] == 0 for c in calls)
        assert calls[0]["result"]["output"] == variant.source
        assert "passed" in calls[-1]["result"]["output"]
        original = freeze.OUTPUT / "preflight" / variant.slug / "original"
        reference = freeze.OUTPUT / "preflight" / variant.slug / "reference"
        assert inspect_fixture(case.task.fixture_id, original).revision == case.task.fixture_revision
        assert (reference / "greeting.py").read_text("utf-8") == variant.source.replace(variant.old, variant.new)
        assert (reference / "test_greeting.py").read_bytes() == (original / "test_greeting.py").read_bytes()


def test_frozen_training_budget_and_matched_evaluation_policy_do_not_authorize_execution():
    plan = read("experiment-plan.json")
    assert plan["max_steps"] == 28 and plan["epochs"] == 2
    assert plan["train_count"] * plan["epochs"] == plan["max_steps"]
    assert plan["fresh_base"] and plan["seed"] == 42
    assert plan["learning_rate"] == 2e-4
    assert (plan["lora_rank"], plan["lora_alpha"], plan["lora_dropout"]) == (8,16,0.05)
    assert plan["target_modules"] == ["q_proj", "v_proj"]
    assert plan["train_batch_size"] == plan["gradient_accumulation_steps"] == 1
    assert plan["max_new_tokens"] == 256 and plan["max_tool_rounds"] == 4
    assert not plan["training_validation"] and not plan["do_sample"] and not plan["enable_thinking"]
    assert all(not plan[key] for key in ("training_authorized", "model_evaluation_authorized", "training_started", "model_inference_started"))
    assert plan["no_automatic_extension"]
    assert plan["hardware_purposes"] == ["inference", "evaluation", "benchmarking"]
    assert plan["sft_manifest_sha256"] == freeze.digest(freeze.mix.OUTPUT / "manifest.json")
    assert plan["old_checkpoint_sha256"] == freeze.digest(freeze.BASELINE)
    assert plan["executor_sha256"] == freeze.digest(freeze.EXECUTOR)
    assert plan["fixed_suite_sha256"] == freeze.digest(freeze.FIXED)
    contract = freeze.mix.baseline.contract()
    assert plan["chat_contract_sha256"] == contract.sha256()
    assert plan["tools"] == list(contract.tools)
    suite = load_coding_evaluation_suite(freeze.OUTPUT / "evaluation-suite.json")
    assert suite.system_prompt == contract.system_prompt


def test_exact_isolation_freeze_integrity_and_preserved_existing_artifacts():
    manifest = freeze.verify_freeze()
    assert manifest["suite_sha256"] == freeze.digest(freeze.OUTPUT / "evaluation-suite.json")
    suite = load_coding_evaluation_suite(freeze.OUTPUT / "evaluation-suite.json")
    assert manifest["case_digest"] == suite.identity().evaluation_suite_digest
    assert freeze.audit_overlap(suite) == read("overlap-audit.json")
    assert not read("overlap-audit.json")["known_exact_overlap"]
    assert read("overlap-audit.json")["legacy_plaintext_tool_results_reviewed"] > 0
    interruption = read("preparation-interruption.json")
    assert not interruption["model_inference_started"] and not interruption["preflights_had_started"]
    old = read("protected-input-hashes.json")
    current = freeze.protected_hashes()
    assert all(current.get(path) == digest for path, digest in old.items())
    summary = read("preflight-summary.json")
    assert summary["unchanged"] and summary["protected_file_count"] == len(old)
    assert not summary["training_started"] and not summary["model_inference_started"]


def test_all_cases_detect_constant_output_overfitting_without_model_inference(tmp_path):
    suite = load_coding_evaluation_suite(freeze.OUTPUT / "evaluation-suite.json")
    for case, variant in zip(suite.cases, freeze.VARIANTS, strict=True):
        workspace = tmp_path / variant.slug
        shutil.copytree(freeze.ROOT / case.fixture_root, workspace)
        tree = ast.parse(variant.source.replace(variant.old, variant.new))
        function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == variant.function)
        function.body = [ast.Return(value=ast.Constant(value=variant.checks[0][1]))]
        (workspace / "greeting.py").write_text(ast.unparse(ast.fix_missing_locations(tree)) + "\n", encoding="utf-8")
        # Pure function gates must detect memorized first-input output even without exact-source gates.
        behavior_case = CodingEvaluationCase.model_validate({**case.model_dump(mode="json"),
            "validators": [v.model_dump(mode="json") for v in case.validators if v.kind == "python_call"]})
        outcomes = run_coding_validators(behavior_case, workspace)
        assert outcomes[0].passed and any(not v.passed for v in outcomes[1:])


def test_frozen_artifact_tampering_and_overwrite_are_rejected(tmp_path, monkeypatch):
    monkeypatch.setattr(freeze.sys, "argv", ["freeze.py"])
    with pytest.raises(FileExistsError):
        freeze.main()
    copy = tmp_path / "frozen-copy"
    shutil.copytree(freeze.OUTPUT, copy)
    monkeypatch.setattr(freeze, "OUTPUT", copy)
    target = copy / "experiment-plan.json"
    target.write_bytes(target.read_bytes() + b" ")
    with pytest.raises(ValueError, match="Frozen validation artifact changed"):
        freeze.verify_freeze()
