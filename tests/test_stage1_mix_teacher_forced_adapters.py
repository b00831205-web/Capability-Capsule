"""Adapter-only setup and reused-base validation; no torch/model loading."""

import pytest

import continue_stage1_mix_teacher_forced_adapters as current


def test_only_two_adapters_can_launch_and_original_driver_stays_pinned(monkeypatch,tmp_path):
    p = current.p
    with monkeypatch.context() as scoped:
        for module,names in ((p,("RUN","__file__","PINS","VARIANTS")),(p.safety,("START_LIMITS",))):
            for name in names:
                scoped.setattr(module,name,getattr(module,name))
        scoped.setattr(current,"RUN",tmp_path / "not-prepared")
        current.configure()
        assert tuple(p.VARIANTS) == ("011","012") and "base" not in p.VARIANTS
        assert p.PINS[current.DEPENDENCY] == current.DEPENDENCY_SHA256
        assert p.safety.START_LIMITS["physical_free_bytes"] == 14 * 1024**3
        assert p.safety.STOP_LIMITS["physical_free_bytes"] == int(1.5 * 1024**3)


def test_reused_base_is_complete_and_recomputable_without_reexecution():
    hashes = current.verify_base()
    assert len(hashes) == 18  # fourteen cases plus plan/summary/model/exit receipts
    assert hashes[current.BASE / "base/summary.json"] == current.BASE_PINS[current.BASE / "base/summary.json"]


def test_wrong_base_digest_is_rejected_without_loading_model(monkeypatch):
    monkeypatch.setattr(current.p,"digest",lambda path:"wrong")
    with pytest.raises(ValueError,match="Reused base identity"):
        current.verify_base()


def test_base_is_rejected_by_evaluation_entrypoint(monkeypatch):
    monkeypatch.setattr(current,"configure",lambda:None)
    with pytest.raises(ValueError,match="Only 011/012"):
        current.evaluate("base")


def test_worker_rejection_parser_is_literal_only():
    from record_stage1_mix_teacher_forced_adapter_status import parse_headroom
    text = "RuntimeError: Insufficient headroom before load: {'physical_free_bytes':1,'commit_free_bytes':2,'c_free_bytes':3,'wsl_available_bytes':4}"
    assert parse_headroom(text)["physical_free_bytes"] == 1
    with pytest.raises(ValueError):
        parse_headroom("no structured rejection")
    with pytest.raises((ValueError,SyntaxError)):
        parse_headroom("RuntimeError: Insufficient headroom before load: __import__('os').system('nope')")
