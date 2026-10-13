"""Configuration checks for a new identity; no model loading."""

import pytest

import retry2_stage1_mix_teacher_forced_diagnostic as current


def test_new_attempt_retains_protocol_and_pins_dependencies(monkeypatch):
    p,previous = current.forward,current.previous
    stops = dict(p.safety.STOP_LIMITS)
    with monkeypatch.context() as scoped:
        for module,names in ((p,("RUN","__file__","PINS")),(previous,("RUN","__file__","configure")),(p.safety,("START_LIMITS",))):
            for name in names:
                scoped.setattr(module,name,getattr(module,name))
        current.configure()
        assert p.RUN == current.RUN and p.RUN != current.PRIOR
        assert p.safety.START_LIMITS["physical_free_bytes"] == 14 * 1024**3
        assert p.safety.STOP_LIMITS == stops
        assert p.PINS[current.DEPENDENCY] == current.DEPENDENCY_SHA256
        assert p.PINS[previous.DEPENDENCY] == previous.DEPENDENCY_SHA256
        assert p.__file__ == current.__file__


def test_changed_dependency_cannot_start_attempt(monkeypatch):
    monkeypatch.setattr(current.forward,"digest",lambda path:"changed")
    with pytest.raises(ValueError,match="Frozen retry"):
        current.configure()
