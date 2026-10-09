"""Budget isolation, immutable inputs and non-destructive execution controls."""

import json
import pytest
import train_stage1_mix012_56step as training


def test_same_sft_fresh_base_changes_only_budget_in_config():
    old, _ = training.baseline.prepare()
    new, manifest = training.prepare()
    assert {k for k, v in old.model_dump().items() if new.model_dump()[k] != v} == {"epochs", "max_steps"}
    assert new.max_steps == 56 and new.epochs == 4
    assert manifest.dataset_stats.train.trajectory_count == 14
    assert manifest.dataset_stats.validation.trajectory_count == 0
    assert manifest.hyperparameters["supervised_token_exposures"] == 12372
    assert not manifest.hyperparameters["model_evaluation_authorized"]


def test_existing_run_cannot_be_overwritten(tmp_path, monkeypatch):
    monkeypatch.setattr(training, "RUN", tmp_path)
    with pytest.raises(FileExistsError):
        training.train_worker()
    with pytest.raises(FileExistsError):
        training.supervise("train")


def test_resource_gate_prevents_worker_start(tmp_path, monkeypatch):
    monkeypatch.setattr(training, "RUN", tmp_path / "new-run")
    monkeypatch.setattr(training.safety, "resources", lambda: {})
    monkeypatch.setattr(training.subprocess, "Popen", lambda *a, **k: pytest.fail("Must not start worker"))
    with pytest.raises(RuntimeError, match="resource gate"):
        training.supervise("train")
    assert not training.RUN.exists()


def test_explicit_override_ignores_memory_but_retains_disk_limits(monkeypatch):
    assert training.resource_limits(True) == training.safety.START_LIMITS
    monkeypatch.setattr(training, "IGNORE_MEMORY_GUARD", True)
    assert training.resource_limits(True) == {"c_free_bytes": 8 * training.safety.GIB}
    assert training.resource_limits(False) == {"c_free_bytes": 4 * training.safety.GIB}
    low_memory = {"c_free_bytes": 9 * training.safety.GIB, "physical_free_bytes": 0,
        "commit_free_bytes": 0, "wsl_available_bytes": 0}
    assert training.safety.below_limits(low_memory, training.resource_limits(True)) == []
    assert training.safety.below_limits({**low_memory, "c_free_bytes": 0}, training.resource_limits(True)) == ["c_free_bytes"]


def test_override_reaches_worker_and_low_memory_does_not_stop_it(tmp_path, monkeypatch):
    monkeypatch.setattr(training, "RUN", tmp_path / "run")
    monkeypatch.setattr(training, "IGNORE_MEMORY_GUARD", True)
    monkeypatch.setattr(training.safety, "resources", lambda: {
        "c_free_bytes": 9 * training.safety.GIB, "physical_free_bytes": 0,
        "commit_free_bytes": 0, "wsl_available_bytes": 0})
    commands = []

    class Worker:
        polls = 0

        def poll(self):
            self.polls += 1
            return None if self.polls == 1 else 0

        def wait(self):
            return 0

    def start(command, **options):
        commands.append(command)
        assert options["start_new_session"]
        return Worker()

    monkeypatch.setattr(training.subprocess, "Popen", start)
    monkeypatch.setattr(training.time, "sleep", lambda seconds: None)
    monkeypatch.setattr(training.os, "killpg", lambda *args: pytest.fail("Memory override must not stop worker"))
    training.supervise("train")
    assert commands[0][-2:] == ["--train-worker", "--ignore-memory-guard"]
    result = json.loads((training.RUN / "train-supervisor-result.json").read_text("utf-8"))
    assert result["exit_code"] == 0 and result["resource_abort"] is None


def test_completed_run_has_exact_budget_and_intact_adapter():
    if not (training.RUN / "training-summary.json").exists():
        pytest.skip("Authorized training has not completed")
    read = lambda name: json.loads((training.RUN / name).read_text("utf-8"))
    state, summary = read("trainer-state.json"), read("training-summary.json")
    assert state["global_step"] == summary["step"] == 56
    assert state["epoch"] == summary["epoch"] == 4
    logs = [json.loads(line) for line in (training.RUN / "optimizer-log.jsonl").read_text("utf-8").splitlines()]
    losses = [entry for entry in logs if "loss" in entry]
    assert [entry["step"] for entry in losses] == list(range(1, 57))
    assert not any("eval_loss" in entry for entry in logs)
    assert losses[0]["learning_rate"] == 2e-4
    assert abs(losses[1]["learning_rate"] - 2e-4 * 55 / 56) < 1e-12
    checkpoint = read("checkpoint.json")
    assert checkpoint["step"] == 56
    adapter = training.RUN / checkpoint["adapter_directory"]
    assert training.digest(adapter / "adapter_model.safetensors") == checkpoint["adapter_model_sha256"]
    assert training.digest(adapter / "adapter_config.json") == checkpoint["adapter_config_sha256"]
    authorization = read("execution-authorization.json")
    assert authorization["memory_guard_override"] and authorization["disk_guard_retained"]
    assert authorization["memory_telemetry_retained"] and not authorization["model_evaluation_authorized"]
    assert training.verify_preservation() == summary["protected_file_count"]
    for mode in ("train",):
        result = read(mode + "-supervisor-result.json")
        assert result["exit_code"] == 0 and result["resource_abort"] is None
        assert not result["model_restarted"]
        attempt = read(mode + "-supervisor-attempt.json")
        assert attempt["memory_guard_override"]
        assert attempt["start_limits"] == {"c_free_bytes": 8 * training.safety.GIB}
        samples = read(mode + "-resource-samples.json")
        assert samples and all(sample["c_free_bytes"] >= 4 * training.safety.GIB for sample in samples)


def test_independent_reload_is_verified_or_explicitly_blocked_before_start():
    read = lambda name: json.loads((training.RUN / name).read_text("utf-8"))
    if not (training.RUN / "adapter-reload-validation.json").exists():
        blocked = read("reload-startup-blocked.json")
        assert blocked["status"] == "blocked_before_worker_or_model_start"
        assert blocked["failed_fields"] == ["c_free_bytes"]
        assert blocked["resources"]["c_free_bytes"] < blocked["required_c_free_bytes"] == 8 * training.safety.GIB
        assert blocked["training_completed"] and not blocked["independent_adapter_reload_completed"]
        assert not (training.RUN / "reload-supervisor-attempt.json").exists()
        assert not (training.RUN / "reload-supervisor-result.json").exists()
        return
    reload = read("adapter-reload-validation.json")
    assert reload["independent_process"] and reload["generated_tokens"] == 0
    assert reload["active_adapters"] == ["default"]
    result = read("reload-supervisor-result.json")
    assert result["exit_code"] == 0 and result["resource_abort"] is None and not result["model_restarted"]


@pytest.mark.parametrize("probe_failure", [False, True])
def test_monitor_stops_only_owned_worker_on_resource_failure(tmp_path, monkeypatch, probe_failure):
    monkeypatch.setattr(training, "RUN", tmp_path / "run")
    good = {key: value + training.safety.GIB for key, value in training.safety.START_LIMITS.items()}
    calls = 0

    def resources():
        nonlocal calls
        calls += 1
        if calls == 1:
            return good
        if probe_failure:
            raise RuntimeError("Probe unavailable")
        return {**good, "physical_free_bytes": 0}

    class Worker:
        pid = 43210
        stopped = False

        def poll(self):
            return -15 if self.stopped else None

        def wait(self, timeout=None):
            return -15

    worker, signals = Worker(), []

    def stop(pid, sig):
        signals.append((pid, sig))
        worker.stopped = True

    monkeypatch.setattr(training.safety, "resources", resources)
    monkeypatch.setattr(training.subprocess, "Popen", lambda *a, **k: worker)
    monkeypatch.setattr(training.os, "killpg", stop)
    with pytest.raises(RuntimeError, match="Owned worker stopped"):
        training.supervise("train")
    assert signals == [(worker.pid, training.signal.SIGTERM)]
    result = json.loads((training.RUN / "train-supervisor-result.json").read_text("utf-8"))
    assert result["resource_abort"] and not result["model_restarted"]
