"""Host limits and no-overwrite recovery controls."""

import pytest

import recover_stage1_mix011_train_fit_diagnostic as recovery


def test_resource_gate_checks_every_host_and_wsl_limit():
    enough = {key: limit + recovery.GIB for key, limit in recovery.START_LIMITS.items()}
    assert recovery.below_limits(enough, recovery.START_LIMITS) == []
    for key in enough:
        snapshot = {**enough, key: recovery.START_LIMITS[key] - 1}
        assert recovery.below_limits(snapshot, recovery.START_LIMITS) == [key]
    assert len(recovery.below_limits({}, recovery.START_LIMITS)) == 4


def test_existing_recovery_cannot_be_overwritten(tmp_path, monkeypatch):
    monkeypatch.setattr(recovery, "RUN", tmp_path)
    with pytest.raises(FileExistsError):
        recovery.prepare()
    recovery.diagnostic.save(tmp_path / "supervisor-attempt.json", {"existing": True})
    with pytest.raises(FileExistsError):
        recovery.supervise()


def test_completed_recovery_reports_agree_with_actual_train_outputs(monkeypatch):
    from test_stage1_mix011_train_fit_results import test_all_14_real_reports_match_prespecified_train_cases_and_limits
    assert (recovery.RUN / "summary.json").exists()
    monkeypatch.setattr(recovery.diagnostic, "RUN", recovery.RUN)
    test_all_14_real_reports_match_prespecified_train_cases_and_limits()
    summary = recovery.diagnostic.read(recovery.RUN / "summary.json")
    assert summary["case_count"] == 14 and summary["split"] == "train"
    assert summary["reruns"] == 0 and not summary["checkpoint_promoted"]


def test_recovery_keeps_original_attempt_and_all_old_inputs():
    prepared = recovery.diagnostic.read(recovery.RUN / "prepared.json")
    assert not prepared["preflights_reexecuted"]
    assert prepared["originals_failed"] == prepared["references_passed"] == 14
    protected = recovery.diagnostic.read(recovery.RUN / "protected-input-hashes.json")
    original_prefix = recovery.ORIGINAL.relative_to(recovery.ROOT).as_posix() + "/"
    old_receipts = {p: digest for p, digest in protected.items() if p.startswith(original_prefix)}
    assert old_receipts
    assert all(recovery.diagnostic.digest(recovery.ROOT / path) == pinned for path, pinned in old_receipts.items())
    evidence = recovery.diagnostic.read(recovery.RUN / "input-preservation.json")
    assert evidence["unchanged"] and evidence["changed_paths"] == []
    assert evidence["protected_file_count"] == len(protected)
    assert not (recovery.ORIGINAL / "case-results.jsonl").exists()


def test_runtime_monitor_retains_samples_and_clean_exit():
    result = recovery.diagnostic.read(recovery.RUN / "supervisor-result.json")
    assert result["exit_code"] == 0 and result["resource_abort"] is None and not result["model_restarted"]
    attempt = recovery.diagnostic.read(recovery.RUN / "supervisor-attempt.json")
    assert recovery.below_limits(attempt["resources"], recovery.START_LIMITS) == []
    samples = recovery.diagnostic.preparation.records(recovery.RUN / "resource-monitor.jsonl")
    assert samples and all(not recovery.below_limits(s, recovery.STOP_LIMITS) for s in samples)


@pytest.mark.parametrize("probe_error", [False, True])
def test_supervisor_stops_only_owned_process_on_resource_failure(tmp_path, monkeypatch, probe_error):
    monkeypatch.setattr(recovery, "RUN", tmp_path)
    good = {key: limit + recovery.GIB for key, limit in recovery.START_LIMITS.items()}
    calls = 0

    def snapshot():
        nonlocal calls
        calls += 1
        if calls == 1:
            return good
        if probe_error:
            raise RuntimeError("probe unavailable")
        return {**good, "commit_free_bytes": recovery.STOP_LIMITS["commit_free_bytes"] - 1}

    class OwnedProcess:
        pid = 43210
        stdout = iter(())
        stopped = False

        def poll(self):
            return -15 if self.stopped else None

        def wait(self, timeout=None):
            return -15

    owned = OwnedProcess()
    terminated = []

    def terminate(group, signal):
        terminated.append((group, signal))
        owned.stopped = True

    monkeypatch.setattr(recovery, "resources", snapshot)
    monkeypatch.setattr(recovery.subprocess, "Popen", lambda *a, **kw: owned)
    monkeypatch.setattr(recovery.os, "killpg", terminate)
    with pytest.raises(RuntimeError, match="Recovery did not complete"):
        recovery.supervise()
    assert terminated == [(owned.pid, recovery.signal.SIGTERM)]
    result = recovery.diagnostic.read(tmp_path / "supervisor-result.json")
    assert result["resource_abort"] is not None and not result["model_restarted"]
