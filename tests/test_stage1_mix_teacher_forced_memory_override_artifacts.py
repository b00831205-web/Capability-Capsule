"""Read-only real scores under the explicit, per-run memory waiver."""

import run_stage1_mix_teacher_forced_adapters_memory_override as current
from authorized_planning_history import input_matches

p = current.p


def test_scoped_authorization_preserves_disk_and_gold_base(monkeypatch):
    plan = p.read(current.RUN / "diagnostic-plan.json")
    auth = p.read(current.RUN / "execution-authorization.json")
    prepared = p.read(current.RUN / "prepared.json")
    assert plan["memory_guard_override"] and auth["memory_guard_override"] and auth["disk_guard_retained"]
    assert auth["authorized_variants"] == ["011","012"]
    assert not auth["system_configuration_changed"] and not auth["automatic_model_retry_authorized"]
    assert not auth["training_authorized"] and not auth["generation_authorized"] and not auth["command_execution_authorized"]
    assert plan["start_limits"] == {"c_free_bytes":8 * 1024**3}
    assert plan["stop_limits"] == {"c_free_bytes":4 * 1024**3}
    assert plan["dtype"] == "float32" and plan["torch_threads"] == 4 and plan["variant_order"] == ["011","012"]
    assert p.digest(current.RUN / "diagnostic-plan.json") == prepared["plan_sha256"]
    assert p.digest(current.RUN / "execution-authorization.json") == prepared["authorization_sha256"]
    assert p.digest(current.RUN / "base-reuse.json") == prepared["base_reuse_sha256"]
    assert p.digest(current.ROOT / "tests/run_stage1_mix_teacher_forced_adapters_memory_override.py") == plan["runner_sha256"]
    assert p.digest(current.DEPENDENCY) == current.DEPENDENCY_SHA256
    reuse = p.read(current.RUN / "base-reuse.json")
    assert len(reuse["hashes"]) == 18 and not reuse["model_reexecuted"]
    assert not (current.RUN / "base").exists()
    assert all(p.digest(current.ROOT / name) == value == plan["input_pins"][name] for name,value in reuse["hashes"].items())


def test_complete_or_honest_partial_scores_match_reused_base(monkeypatch):
    from test_stage1_mix_teacher_forced_adapters_artifacts import test_saved_adapter_scores_align_to_reused_base_and_recompute
    with monkeypatch.context() as scoped:
        scoped.setattr(current.previous,"RUN",current.RUN)
        test_saved_adapter_scores_align_to_reused_base_and_recompute()


def test_actual_disk_monitoring_and_preservation():
    plan = p.read(current.RUN / "diagnostic-plan.json")
    complete = (current.RUN / "comparison.json").exists()
    for variant in current.previous.ADAPTERS:
        directory = current.RUN / variant
        if not directory.exists():
            assert not complete
            continue
        launch = p.read(directory / "launch.json")
        assert not p.safety.below_limits(launch["resources"],plan["start_limits"])
        end = p.read(directory / "supervisor-result.json")
        assert not end["restarted"]
        samples = p.inventory.preparation.records(directory / "resources.jsonl")
        assert samples and all(all(k in s for k in ("physical_free_bytes","commit_free_bytes","wsl_available_bytes","c_free_bytes")) for s in samples)
        if complete:
            assert end["exit_code"] == 0 and end["resource_abort"] is None
            assert all(not p.safety.below_limits(s,plan["stop_limits"]) for s in samples)
        if (directory / "model-load.json").exists():
            loaded = p.read(directory / "model-load.json")
            assert loaded["active_adapters"] == ["default"] and not loaded["training_mode"]
            assert loaded["floating_dtypes"] == ["torch.float32"] and loaded["devices"] == ["cpu"]
    hashes = p.read(current.RUN / "protected-input-hashes.json")
    assert all(p.digest(current.ROOT / name) == value for name,value in hashes.items())
    assert len(hashes) == p.read(current.RUN / "input-preservation.json")["protected_file_count"]
    if complete:
        completion = p.read(current.RUN / "completion.json")
        assert completion["memory_guard_override"] and completion["disk_guard_retained"]
        assert not completion["system_configuration_changed"] and not completion["base_reexecuted"]
    update = p.read(current.RUN / "planning-update.json")
    assert update["explicitly_authorized"]
    for name,before in update["before"].items():
        assert input_matches(name,before,p.digest(current.ROOT / name))


def test_critical_analysis_comes_only_from_saved_scores():
    analysis = p.read(current.RUN / "critical-token-analysis.json")
    examples = p.inventory.preparation.records(p.SFT / "train.jsonl")
    for variant,directory in (("base",current.previous.BASE / "base"),("011",current.RUN / "011"),("012",current.RUN / "012")):
        counts = []
        for example in examples:
            report = p.read(directory / "cases" / (example["source_trajectory_id"]+".json"))
            counts.append(sum(bool({"source_literal","replacement_literal"}.intersection(r["categories"])) and not r["top1_correct"] for r in report["tokens"]))
        profile = analysis["profiles"][variant]
        assert profile["literal_argmax_mismatches"] == sum(counts)
        assert profile["cases_with_literal_argmax_mismatch"] == sum(c>0 for c in counts)
        assert len(profile["literal_mismatch_tokens"]) == sum(counts)
    transitions = analysis["011_to_012_literal_transitions"]
    assert (analysis["profiles"]["011"]["literal_argmax_mismatches"]-analysis["profiles"]["012"]["literal_argmax_mismatches"]
        == transitions["011_wrong_012_correct"]-transitions["011_correct_012_wrong"])
