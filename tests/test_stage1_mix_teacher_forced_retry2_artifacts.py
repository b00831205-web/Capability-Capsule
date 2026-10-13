"""Read-only checks of the renewed attempt, including honest partial-run handling."""

import pytest
import math

import retry2_stage1_mix_teacher_forced_diagnostic as current
from authorized_planning_history import input_matches

p = current.forward


def test_plan_pins_unchanged_protocol_and_both_prior_implementations():
    plan = p.read(current.RUN / "diagnostic-plan.json")
    assert plan["dtype"] == "float32" and plan["device"] == "cpu" and plan["torch_threads"] == 4
    assert not plan["generation"] and not plan["training"] and not plan["command_execution"]
    assert plan["case_count"] == 14 and plan["supervised_token_count"] == 3093
    assert plan["start_limits"]["physical_free_bytes"] == 14 * 1024**3
    assert plan["stop_limits"]["physical_free_bytes"] == int(1.5 * 1024**3)
    assert p.digest(current.ROOT / "tests/retry2_stage1_mix_teacher_forced_diagnostic.py") == plan["runner_sha256"]
    for path,expected in ((current.DEPENDENCY,current.DEPENDENCY_SHA256),
            (current.previous.DEPENDENCY,current.previous.DEPENDENCY_SHA256)):
        assert p.digest(path) == expected
        assert plan["input_pins"][path.relative_to(current.ROOT).as_posix()] == expected
    renewed = p.read(current.RUN / "renewed-request.json")
    assert renewed["protocol_change"] is None and not renewed["automatic_retry"]
    assert p.digest(current.PRIOR / "base-startup-blocked.json") == renewed["prior_block_sha256"]


def test_metrics_are_complete_or_explicitly_partial(monkeypatch):
    if (current.RUN / "comparison.json").exists():
        from test_stage1_mix_teacher_forced_artifacts import test_real_metrics_align_identical_gold_targets_and_are_recomputable
        with monkeypatch.context() as scoped:
            scoped.setattr(p,"RUN",current.RUN)
            test_real_metrics_align_identical_gold_targets_and_are_recomputable()
        completion = p.read(current.RUN / "completion.json")
        assert completion["variant_count"] == 3 and completion["case_count_per_variant"] == 14
        assert completion["generation_calls"] == completion["executed_commands"] == 0
        assert not completion["training_started"]
    else:
        interruption = p.read(current.RUN / "interruption.json")
        examples = {r["source_trajectory_id"]:r for r in p.inventory.preparation.records(p.SFT / "train.jsonl")}
        assert not interruption["diagnostic_comparison_completed"] and interruption["case_success_count"] is None
        assert not (current.RUN / "completion.json").exists()
        for variant in p.VARIANTS:
            cases = list((current.RUN / variant / "cases").glob("*.json"))
            all_tokens = []
            assert len(cases) == interruption["recorded_case_counts"][variant]
            assert len(cases) <= 14
            if (current.RUN / variant / "summary.json").exists():
                assert len(cases) == 14
            for path in cases:
                report = p.read(path)
                assert report["split"] == "train"
                assert report["metrics"]["all_supervised"]["token_count"] == len(report["tokens"])
                example = examples[report["trajectory_id"]]
                tokens = report["tokens"]
                assert [r["position"] for r in tokens] == p.prediction_positions(example["labels"])
                assert all(r["target_id"] == example["labels"][r["position"]] for r in tokens)
                assert all(math.isfinite(r["nll"]) and r["nll"] >= -1e-5 for r in tokens)
                all_tokens.extend(tokens)
            if (current.RUN / variant / "summary.json").exists():
                summary = p.read(current.RUN / variant / "summary.json")
                assert len(all_tokens) == 3093 and summary["case_count"] == 14
                for category in p.CATEGORIES:
                    selected = [r for r in all_tokens if category in r["categories"]]
                    metrics = summary["metrics"][category]
                    assert metrics["token_count"] == len(selected)
                    if selected:
                        assert metrics["mean_nll"] == pytest.approx(sum(r["nll"] for r in selected)/len(selected))
                        assert metrics["top1_accuracy"] == pytest.approx(sum(r["predicted_id"] == r["target_id"] for r in selected)/len(selected))
        assert interruption["generation_calls"] == interruption["executed_commands"] == 0


def test_guards_old_bytes_and_authorized_planning_receipt():
    plan = p.read(current.RUN / "diagnostic-plan.json")
    complete = (current.RUN / "comparison.json").exists()
    for variant in p.VARIANTS:
        directory = current.RUN / variant
        if not directory.exists():
            assert not complete
            continue
        launch = p.read(directory / "launch.json")
        assert not p.safety.below_limits(launch["resources"],plan["start_limits"])
        end = p.read(directory / "supervisor-result.json")
        assert not end["restarted"]
        if complete:
            assert end["exit_code"] == 0 and end["resource_abort"] is None
        samples = p.inventory.preparation.records(directory / "resources.jsonl")
        assert samples
        if end["resource_abort"] is None and end["exit_code"] == 0:
            assert all(not p.safety.below_limits(s,plan["stop_limits"]) for s in samples)
        elif end["resource_abort"] is not None and "resources" in end["resource_abort"]:
            abort = end["resource_abort"]
            assert samples[-1] == abort["resources"]
            assert p.safety.below_limits(samples[-1],plan["stop_limits"]) == abort["low_fields"]
        if (directory / "model-load.json").exists():
            loaded = p.read(directory / "model-load.json")
            assert loaded["devices"] == ["cpu"] and loaded["floating_dtypes"] == ["torch.float32"]
            assert not loaded["training_mode"]
            assert loaded["active_adapters"] == ([] if variant == "base" else ["default"])
    hashes = p.read(current.RUN / "protected-input-hashes.json")
    assert all(p.digest(current.ROOT / name) == value for name,value in hashes.items())
    assert len(hashes) == p.read(current.RUN / "input-preservation.json")["protected_file_count"]
    update = p.read(current.RUN / "planning-update.json")
    assert update["explicitly_authorized"]
    for name,before in update["before"].items():
        assert input_matches(name,before,p.digest(current.ROOT / name))
