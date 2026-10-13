"""The explicit memory waiver changes only process-local gates, never scoring or disk safety."""

import pytest

import run_stage1_mix_teacher_forced_adapters_memory_override as current


def test_only_memory_gates_are_disabled_and_base_cannot_run(monkeypatch,tmp_path):
    p,previous = current.p,current.previous
    with monkeypatch.context() as scoped:
        for module,names in ((p,("RUN","__file__","PINS","VARIANTS")),
                (previous,("RUN","__file__","configure")),(p.safety,("START_LIMITS","STOP_LIMITS"))):
            for name in names:
                scoped.setattr(module,name,getattr(module,name))
        scoped.setattr(current,"RUN",tmp_path / "not-prepared")
        current.configure()
        assert p.safety.START_LIMITS == {"c_free_bytes":8 * 1024**3}
        assert p.safety.STOP_LIMITS == {"c_free_bytes":4 * 1024**3}
        resources = {"c_free_bytes":9 * 1024**3,"physical_free_bytes":0,"commit_free_bytes":0,"wsl_available_bytes":0}
        assert not p.safety.below_limits(resources,p.safety.START_LIMITS)
        assert p.safety.below_limits({**resources,"c_free_bytes":3 * 1024**3},p.safety.STOP_LIMITS) == ["c_free_bytes"]
        assert tuple(p.VARIANTS) == ("011","012")
        assert p.PINS[current.DEPENDENCY] == current.DEPENDENCY_SHA256
        assert p.PINS[previous.DEPENDENCY] == previous.DEPENDENCY_SHA256


def test_modified_dependency_cannot_gain_memory_waiver(monkeypatch):
    monkeypatch.setattr(current.p,"digest",lambda path:"modified")
    with pytest.raises(ValueError,match="Frozen adapter-only"):
        current.configure()
