"""Recovery guards and immutable protocol, without loading a model."""

import pytest
import recover_stage1_mix012_train_fit_diagnostic as recovery


@pytest.fixture(autouse=True)
def restore_globals(monkeypatch):
    candidate = recovery.candidate
    monkeypatch.setattr(candidate, "RUN", candidate.RUN)
    monkeypatch.setattr(candidate.diagnostic, "RUN", candidate.diagnostic.RUN)
    for name in ("RUN_ID", "CHECKPOINT", "PROPOSED_OUTPUT", "PINS", "build_plan"):
        monkeypatch.setattr(candidate.preparation, name, getattr(candidate.preparation, name))


def test_recovery_matches_original_and_011_protocol():
    recovery.verify_original()
    candidate = recovery.candidate
    plan = candidate.build_plan()
    old = recovery.diagnostic.read(recovery.ORIGINAL / "diagnostic-plan.json")
    assert all(plan[key] == old[key] for key in candidate.SAME_FIELDS)
    assert plan["checkpoint_sha256"] == candidate.CHECKPOINT_SHA256
    assert plan["checkpoint_step"] == 56
    assert recovery.RUN.parent == recovery.ORIGINAL
    assert all(case["split"] == "train" for case in plan["cases"])


def test_recovery_refuses_existing_directory(tmp_path, monkeypatch):
    monkeypatch.setattr(recovery, "RUN", tmp_path)
    with pytest.raises(FileExistsError):
        recovery.prepare()


def test_recovery_restores_memory_guards_and_retains_disk_guards():
    assert recovery.START_LIMITS == recovery.safety.START_LIMITS
    assert recovery.STOP_LIMITS == recovery.safety.STOP_LIMITS
    assert recovery.START_LIMITS["c_free_bytes"] == 8 * recovery.safety.GIB
    assert recovery.STOP_LIMITS["c_free_bytes"] == 4 * recovery.safety.GIB
    assert "physical_free_bytes" in recovery.START_LIMITS
    assert "wsl_available_bytes" in recovery.STOP_LIMITS


def test_paging_check_requires_actual_f_allocation(monkeypatch):
    from types import SimpleNamespace
    import json
    monkeypatch.setattr(recovery.subprocess, "run", lambda *a, **k: SimpleNamespace(
        stdout=json.dumps({"Name": "C:\\pagefile.sys", "AllocatedBaseSize": 2048})))
    with pytest.raises(ValueError, match="not actually active"):
        recovery.paging_state()
    monkeypatch.setattr(recovery.subprocess, "run", lambda *a, **k: SimpleNamespace(
        stdout=json.dumps({"Name": "F:\\pagefile.sys", "AllocatedBaseSize": 16384})))
    assert recovery.paging_state()[0]["AllocatedBaseSize"] == 16384


def test_recovery_refuses_repeated_inference(tmp_path, monkeypatch):
    monkeypatch.setattr(recovery, "RUN", tmp_path)
    recovery.diagnostic.save(tmp_path / "recovery-authorization.json", {})
    recovery.diagnostic.save(tmp_path / "diagnostic-launch.json", {})
    with pytest.raises(FileExistsError):
        recovery.supervise()


def test_startup_gate_blocks_model_on_low_memory_or_disk(tmp_path, monkeypatch):
    monkeypatch.setattr(recovery, "RUN", tmp_path)
    recovery.diagnostic.save(tmp_path / "recovery-authorization.json", {})
    monkeypatch.setattr(recovery, "paging_state", lambda: [])
    monkeypatch.setattr(recovery.safety, "resources", lambda: {})
    monkeypatch.setattr(recovery.subprocess, "Popen", lambda *a, **k: pytest.fail("Must not start model"))
    with pytest.raises(RuntimeError, match="resource gate"):
        recovery.supervise()
    assert not (tmp_path / "diagnostic-launch.json").exists()


@pytest.mark.parametrize("probe_failure", [False, True])
def test_runtime_guard_stops_only_owned_worker(tmp_path, monkeypatch, probe_failure):
    monkeypatch.setattr(recovery, "RUN", tmp_path)
    recovery.diagnostic.save(tmp_path / "recovery-authorization.json", {})
    monkeypatch.setattr(recovery, "paging_state", lambda: [])
    monkeypatch.setattr(recovery.diagnostic, "verify_preservation", lambda: 1)
    calls = 0
    good = {key: value + recovery.safety.GIB for key, value in recovery.START_LIMITS.items()}

    def resources():
        nonlocal calls
        calls += 1
        if calls == 1:
            return good
        if probe_failure:
            raise RuntimeError("Probe unavailable")
        return {**good, "c_free_bytes": 0}

    class Worker:
        pid = 43210
        stopped = False
        stdout = ()

        def poll(self):
            return -15 if self.stopped else None

        def wait(self, timeout=None):
            return -15

    worker, signals = Worker(), []

    def start(command, **options):
        assert command[-1] == "--worker"
        assert "--train-worker" not in command
        assert options["start_new_session"]
        return worker

    def stop(pid, sig):
        signals.append((pid, sig))
        worker.stopped = True

    monkeypatch.setattr(recovery.safety, "resources", resources)
    monkeypatch.setattr(recovery.subprocess, "Popen", start)
    monkeypatch.setattr(recovery.os, "killpg", stop)
    with pytest.raises(RuntimeError, match="interrupted"):
        recovery.supervise()
    assert signals == [(43210, recovery.signal.SIGTERM)]
    end = recovery.diagnostic.read(tmp_path / "supervisor-result.json")
    assert end["exit_code"] == -15 and not end["model_restarted"]
    assert end["resource_abort"]["reason"] == ("Resource monitor unavailable" if probe_failure else "Resource threshold crossed")


def test_completed_recovery_reports_and_comparison_agree(monkeypatch):
    from test_stage1_mix011_train_fit_results import test_all_14_real_reports_match_prespecified_train_cases_and_limits
    recovery.configure()
    test_all_14_real_reports_match_prespecified_train_cases_and_limits()
    read = recovery.diagnostic.read
    summary = read(recovery.RUN / "summary.json")
    comparison = read(recovery.RUN / "comparison.json")
    assert summary["checkpoint_id"] == recovery.candidate.TRAIN_ID
    assert comparison["split"] == "train" and len(comparison["pairs"]) == 14
    assert comparison["baseline"]["strict_success"] == comparison["baseline"]["behavior_pass"] == 0
    assert comparison["strict_delta"] == summary["strict_success"]
    assert comparison["behavior_delta"] == summary["behavior_pass"]
    assert comparison["candidate"] == recovery.diagnostic.summarize([
        read(recovery.RUN / "case-reports" / (case["task_id"] + ".json"))
        for case in read(recovery.RUN / "diagnostic-plan.json")["cases"]])
    assert sum(row["strict_012"] for row in comparison["pairs"]) == summary["strict_success"]
    assert sum(row["behavior_012"] for row in comparison["pairs"]) == summary["behavior_pass"]
    end = read(recovery.RUN / "supervisor-result.json")
    assert end["exit_code"] == 0 and end["resource_abort"] is None and not end["model_restarted"]
    authorization = read(recovery.RUN / "recovery-authorization.json")
    assert not authorization["memory_guard_override"] and authorization["disk_guard_retained"]
    samples = recovery.diagnostic.preparation.records(recovery.RUN / "resource-monitor.jsonl")
    assert samples and not any(recovery.safety.below_limits(row, recovery.STOP_LIMITS) for row in samples)
    completion = read(recovery.RUN / "completion.json")
    assert completion["case_count"] == 14 and completion["original_interrupted_attempt_preserved"]
    assert not completion["new_training_started"] and not completion["checkpoint_promoted"]
    assert recovery.diagnostic.verify_preservation() == completion["protected_file_count"]


def test_planning_receipt_extends_only_exact_authorized_chain(tmp_path, monkeypatch):
    import json
    from hashlib import sha256
    import authorized_planning_history as history
    monkeypatch.setattr(history, "ROOT", tmp_path)
    name = "MVP_ROADMAP.md"
    (tmp_path / name).write_text("recovered", encoding="utf-8")
    first, middle, final = [sha256(value).hexdigest() for value in (b"initial", b"interrupted", b"recovered")]
    initial = tmp_path / "initial.json"
    initial.write_text(json.dumps({"explicitly_authorized": True,
        "before": {name: first}, "after": {name: middle}}), encoding="utf-8")
    receipt = tmp_path / recovery.RUN.relative_to(recovery.ROOT) / "planning-update.json"
    receipt.parent.mkdir(parents=True)
    value = {"explicitly_authorized": True, "before": {name: middle}, "after": {name: final}}
    receipt.write_text(json.dumps(value), encoding="utf-8")
    assert history.input_matches(name, first, final, update_path=initial)
    receipt.write_text(json.dumps({**value, "explicitly_authorized": False}), encoding="utf-8")
    assert not history.input_matches(name, first, final, update_path=initial)
