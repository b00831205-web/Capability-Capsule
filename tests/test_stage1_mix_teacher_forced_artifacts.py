"""Read-only real-run checks for the authorized forward-only diagnostic."""

import ast
import math

import pytest

import run_stage1_mix_teacher_forced_diagnostic as p
from authorized_planning_history import input_matches


def test_runner_contains_no_generation_execution_or_training_calls():
    tree = ast.parse((p.ROOT / "tests/run_stage1_mix_teacher_forced_diagnostic.py").read_text("utf-8"))
    forbidden = {"generate","execute","backward","step","train","run_lora_training"}
    names = {n.func.attr if isinstance(n.func,ast.Attribute) else n.func.id if isinstance(n.func,ast.Name) else ""
        for n in ast.walk(tree) if isinstance(n,ast.Call)}
    assert not (names & forbidden)


def test_real_metrics_align_identical_gold_targets_and_are_recomputable():
    if (p.RUN / "interruption.json").exists():
        interruption = p.read(p.RUN / "interruption.json")
        assert interruption["recorded_complete_case_count"] == interruption["recorded_token_score_count"] == 0
        assert interruption["case_success_count"] is None and not interruption["diagnostic_comparison_completed"]
        assert not (p.RUN / "comparison.json").exists() and not (p.RUN / "completion.json").exists()
        assert not (p.RUN / "base/summary.json").exists()
        assert not list((p.RUN / "base/cases").glob("*.json"))
        assert not (p.RUN / "011").exists() and not (p.RUN / "012").exists()
        return
    examples = p.inventory.preparation.records(p.SFT / "train.jsonl")
    comparison = p.read(p.RUN / "comparison.json")
    per_variant = {}
    for variant in p.VARIANTS:
        records = []
        for example in examples:
            report = p.read(p.RUN / variant / "cases" / (example["source_trajectory_id"] + ".json"))
            tokens = report["tokens"]
            assert report["split"] == "train"
            assert [r["position"] for r in tokens] == p.prediction_positions(example["labels"])
            assert all(r["target_id"] == example["labels"][r["position"]] for r in tokens)
            assert all(math.isfinite(r["nll"]) and r["nll"] >= -1e-5 for r in tokens)
            assert all(r["target_probability"] == pytest.approx(math.exp(-r["nll"])) for r in tokens)
            records.extend(tokens)
        assert len(records) == 3093
        summary = p.read(p.RUN / variant / "summary.json")
        assert summary["case_count"] == 14 and summary["forward_only"]
        assert summary["generation_calls"] == summary["executed_commands"] == 0
        for category in p.CATEGORIES:
            selected = [r for r in records if category in r["categories"]]
            metrics = summary["metrics"][category]
            assert metrics["token_count"] == len(selected)
            if selected:
                assert metrics["mean_nll"] == pytest.approx(sum(r["nll"] for r in selected)/len(selected))
                assert metrics["top1_accuracy"] == pytest.approx(sum(r["predicted_id"] == r["target_id"] for r in selected)/len(selected))
        assert comparison["variants"][variant] == summary
        per_variant[variant] = [(r["position"],r["target_id"],r["categories"]) for r in records]
    assert per_variant["base"] == per_variant["011"] == per_variant["012"]


def test_resources_model_identity_and_old_inputs_are_preserved():
    plan = p.read(p.RUN / "diagnostic-plan.json")
    assert p.digest(p.ROOT / "tests/run_stage1_mix_teacher_forced_diagnostic.py") == plan["runner_sha256"]
    assert not plan["generation"] and not plan["training"] and not plan["command_execution"]
    for variant in p.VARIANTS:
        if (p.RUN / "interruption.json").exists() and variant != "base":
            assert not (p.RUN / variant).exists()
            continue
        end = p.read(p.RUN / variant / "supervisor-result.json")
        interrupted = (p.RUN / "interruption.json").exists()
        if interrupted:
            assert end["exit_code"] == -15 and end["resource_abort"]["low_fields"] == ["physical_free_bytes"]
            assert end["resource_abort"]["resources"]["physical_free_bytes"] < p.safety.STOP_LIMITS["physical_free_bytes"]
        else:
            assert end["exit_code"] == 0 and end["resource_abort"] is None
        assert not end["restarted"]
        loaded = p.read(p.RUN / variant / "model-load.json")
        assert loaded["devices"] == ["cpu"] and loaded["floating_dtypes"] == ["torch.float32"]
        assert not loaded["training_mode"]
        assert loaded["active_adapters"] == ([] if variant == "base" else ["default"])
        samples = p.inventory.preparation.records(p.RUN / variant / "resources.jsonl")
        assert samples
        if interrupted:
            assert samples[-1] == end["resource_abort"]["resources"]
            assert p.safety.below_limits(samples[-1],p.safety.STOP_LIMITS) == ["physical_free_bytes"]
        else:
            assert all(not p.safety.below_limits(s,p.safety.STOP_LIMITS) for s in samples)
    assert p.verify_preservation() == p.read(p.RUN / "input-preservation.json")["protected_file_count"]
    update = p.read(p.RUN / "planning-update.json")
    for name,before in update["before"].items():
        assert input_matches(name,before,p.digest(p.ROOT / name))
