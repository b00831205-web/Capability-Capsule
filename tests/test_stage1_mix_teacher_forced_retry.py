"""New attempt identity and stronger startup guard; never load a model."""

import pytest

import retry_stage1_mix_teacher_forced_diagnostic as retry


def test_retry_preserves_implementation_and_strengthens_only_startup_ram(monkeypatch):
    p = retry.forward
    original_stop = dict(p.safety.STOP_LIMITS)
    with monkeypatch.context() as scoped:
        scoped.setattr(p,"RUN",p.RUN)
        scoped.setattr(p,"__file__",p.__file__)
        scoped.setattr(p,"PINS",p.PINS)
        scoped.setattr(p.safety,"START_LIMITS",p.safety.START_LIMITS)
        retry.configure()
        assert p.RUN == retry.RUN and p.RUN != retry.ORIGINAL
        assert p.PINS[retry.DEPENDENCY] == retry.DEPENDENCY_SHA256
        assert p.safety.START_LIMITS["physical_free_bytes"] == 14 * 1024**3
        assert p.safety.STOP_LIMITS == original_stop
        assert p.safety.START_LIMITS["c_free_bytes"] == 8 * 1024**3
        assert p.safety.START_LIMITS["commit_free_bytes"] == 14 * 1024**3
        assert p.safety.START_LIMITS["wsl_available_bytes"] == 10 * 1024**3


def test_changed_dependency_is_rejected(monkeypatch):
    monkeypatch.setattr(retry.forward,"digest",lambda path:"changed")
    with pytest.raises(ValueError,match="Frozen forward"):
        retry.configure()


def test_real_startup_block_preserves_old_inputs_and_records_no_model_scores():
    p = retry.forward
    plan = p.read(retry.RUN / "diagnostic-plan.json")
    blocked = p.read(retry.RUN / "base-startup-blocked.json")
    assert plan["start_limits"]["physical_free_bytes"] == 14 * 1024**3
    assert plan["stop_limits"]["physical_free_bytes"] == int(1.5 * 1024**3)
    assert p.safety.below_limits(blocked["resources"],plan["start_limits"]) == ["physical_free_bytes"]
    assert p.digest(retry.DEPENDENCY) == retry.DEPENDENCY_SHA256
    assert p.digest(retry.ROOT / "tests/retry_stage1_mix_teacher_forced_diagnostic.py") == plan["runner_sha256"]
    for variant in p.VARIANTS:
        assert not (retry.RUN / variant).exists()
    assert not (retry.RUN / "comparison.json").exists() and not (retry.RUN / "completion.json").exists()
    interruption = p.read(retry.RUN / "interruption.json")
    assert interruption["recorded_case_counts"] == {"base":0,"011":0,"012":0}
    assert interruption["case_success_count"] is None and not interruption["automatic_retry_performed"]
    protected = p.read(retry.RUN / "protected-input-hashes.json")
    assert all(p.digest(retry.ROOT / path) == value for path,value in protected.items())
    assert len(protected) == p.read(retry.RUN / "input-preservation.json")["protected_file_count"]
    from authorized_planning_history import input_matches
    update = p.read(retry.RUN / "planning-update.json")
    for name,before in update["before"].items():
        assert input_matches(name,before,p.digest(retry.ROOT / name))
