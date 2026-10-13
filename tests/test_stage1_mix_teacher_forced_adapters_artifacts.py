"""Read-only reused-base and adapter-output acceptance, with explicit missing results."""

import pytest

import continue_stage1_mix_teacher_forced_adapters as current
from authorized_planning_history import input_matches

p = current.p


def test_base_reuse_and_adapter_only_frozen_identity():
    plan = p.read(current.RUN / "diagnostic-plan.json")
    reuse = p.read(current.RUN / "base-reuse.json")
    prepared = p.read(current.RUN / "prepared.json")
    assert p.digest(current.RUN / "diagnostic-plan.json") == prepared["plan_sha256"]
    assert p.digest(current.RUN / "base-reuse.json") == prepared["base_reuse_sha256"]
    assert plan["variant_order"] == ["011","012"] and not (current.RUN / "base").exists()
    assert plan["dtype"] == "float32" and plan["device"] == "cpu" and plan["torch_threads"] == 4
    assert not plan["generation"] and not plan["training"] and not plan["command_execution"]
    assert not reuse["model_reexecuted"] and reuse["case_count"] == 14 and len(reuse["hashes"]) == 18
    assert p.digest(current.ROOT / "tests/continue_stage1_mix_teacher_forced_adapters.py") == plan["runner_sha256"]
    for name,value in reuse["hashes"].items():
        assert p.digest(current.ROOT / name) == value == plan["input_pins"][name]
    assert {path.relative_to(current.ROOT).as_posix():value for path,value in current.verify_base().items()} == reuse["hashes"]


def test_saved_adapter_scores_align_to_reused_base_and_recompute():
    examples = p.inventory.preparation.records(p.SFT / "train.jsonl")
    counts = {}
    for variant in current.ADAPTERS:
        directory = current.RUN / variant
        reports = list((directory / "cases").glob("*.json"))
        counts[variant] = len(reports)
        tokens = []
        for example in examples:
            name = example["source_trajectory_id"]+".json"
            if not (directory / "cases" / name).exists():
                continue
            report = p.read(directory / "cases" / name)
            base = p.read(current.BASE / "base/cases" / name)
            def targets(record):
                return [(t["position"],t["target_id"],t["categories"]) for t in record["tokens"]]
            assert targets(report) == targets(base)
            assert report["split"] == "train" and report["seen_during_training"]
            assert report["metrics"] == p.aggregate(report["tokens"])
            tokens.extend(report["tokens"])
        if (directory / "summary.json").exists():
            summary = p.read(directory / "summary.json")
            assert counts[variant] == 14 and len(tokens) == 3093
            assert summary["metrics"] == p.aggregate(tokens)
            assert summary["generation_calls"] == summary["executed_commands"] == 0
    if (current.RUN / "comparison.json").exists():
        comparison = p.read(current.RUN / "comparison.json")
        assert counts == {"011":14,"012":14} and not comparison["base_reexecuted"]
        assert comparison["variants"]["base"] == p.read(current.BASE / "base/summary.json")
        for variant in current.ADAPTERS:
            assert comparison["variants"][variant] == p.read(current.RUN / variant / "summary.json")
        completion = p.read(current.RUN / "completion.json")
        assert completion["new_variants"] == ["011","012"] and completion["reused_variant"] == "base"
        assert not completion["base_reexecuted"] and completion["comparison_variant_count"] == 3
    else:
        interruption = p.read(current.RUN / "interruption.json")
        assert interruption["recorded_adapter_case_counts"] == counts
        assert interruption["reused_base_case_count"] == 14 and not interruption["diagnostic_comparison_completed"]
        assert not interruption["base_reexecuted"] and not (current.RUN / "completion.json").exists()


def test_guards_protected_files_and_document_receipts():
    plan = p.read(current.RUN / "diagnostic-plan.json")
    assert plan["start_limits"]["physical_free_bytes"] == 14 * 1024**3
    assert plan["stop_limits"]["physical_free_bytes"] == int(1.5 * 1024**3)
    for variant in current.ADAPTERS:
        directory = current.RUN / variant
        blocked = current.RUN / (variant+"-startup-blocked.json")
        if blocked.exists():
            receipt = p.read(blocked)
            assert p.safety.below_limits(receipt["resources"],plan["start_limits"])
            assert not directory.exists()
            continue
        assert directory.exists()
        launch = p.read(directory / "launch.json")
        assert not p.safety.below_limits(launch["resources"],plan["start_limits"])
        end = p.read(directory / "supervisor-result.json")
        assert not end["restarted"]
        samples = p.inventory.preparation.records(directory / "resources.jsonl")
        assert samples
        if end["exit_code"] == 0:
            assert end["resource_abort"] is None
            assert all(not p.safety.below_limits(s,plan["stop_limits"]) for s in samples)
        elif end["exit_code"] == 1 and end["resource_abort"] is None and not (directory / "model-load.json").exists():
            worker = p.read(directory / "worker-startup-blocked.json")
            assert not worker["model_loaded"] and worker["recorded_case_count"] == 0
            assert p.safety.below_limits(worker["resources"],plan["start_limits"]) == worker["failed_fields"]
            assert not (directory / "forward-attempt.json").exists()
        elif end["resource_abort"] is not None and "resources" in end["resource_abort"]:
            assert samples[-1] == end["resource_abort"]["resources"]
        if (directory / "model-load.json").exists():
            loaded = p.read(directory / "model-load.json")
            assert loaded["active_adapters"] == ["default"] and not loaded["training_mode"]
            assert loaded["devices"] == ["cpu"] and loaded["floating_dtypes"] == ["torch.float32"]
    hashes = p.read(current.RUN / "protected-input-hashes.json")
    assert all(p.digest(current.ROOT / name) == value for name,value in hashes.items())
    assert p.read(current.RUN / "input-preservation.json")["protected_file_count"] == len(hashes)
    update = p.read(current.RUN / "planning-update.json")
    assert update["explicitly_authorized"]
    for name,before in update["before"].items():
        assert input_matches(name,before,p.digest(current.ROOT / name))
