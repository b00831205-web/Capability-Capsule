"""The waiver changes only resource gates, never contexts, generation or old artifacts."""
import pytest
import run_stage1_mix012_first_edit_precision_memory_override as m


def test_only_disk_thresholds_remain():
    assert m.START == {"c_free_bytes":8 * 1024**3}
    assert m.STOP == {"c_free_bytes":4 * 1024**3}
    assert m.RUN != m.ORIGINAL


def test_child_worker_enters_scoped_wrapper(monkeypatch):
    for key in ("RUN","START","STOP","verify","__file__"):
        monkeypatch.setattr(m.p,key,getattr(m.p,key))
    m.configure()
    assert m.p.RUN == m.RUN and m.p.__file__ == m.__file__
    assert m.p.verify is m.verify and m.p.START == m.START and m.p.STOP == m.STOP


def test_frozen_cases_and_full_generation_policy_unchanged():
    plan = m.verify()
    original = m.p.read(m.ORIGINAL/"diagnostic-plan.json")
    for key in ("cases","contract","variant_order","max_new_tokens","use_cache","do_sample",
                "seed","torch_threads","device","adapter_autocast_dtype"):
        assert plan[key] == original[key]
    auth = m.p.read(m.RUN/"execution-authorization.json")
    assert auth["memory_guard_override"] and auth["disk_guard_retained"]
    assert auth["memory_telemetry_retained"]
    assert not auth["training"] and not auth["command_execution"] and not auth["system_configuration_changed"]


@pytest.mark.parametrize("key,value",[("memory_guard_override",False),("disk_guard_retained",False),
                                      ("max_new_tokens",512),("start_limits",{})])
def test_tampered_scope_or_budget_rejected(monkeypatch,key,value):
    original_read = m.p.read
    def read(path):
        data = original_read(path)
        if path == m.RUN/"diagnostic-plan.json":
            data[key] = value
        return data
    monkeypatch.setattr(m.p,"read",read)
    with pytest.raises(ValueError):
        m.verify()


def test_old_inputs_unchanged_and_no_missing_score_counted_as_failure():
    protected = m.p.read(m.RUN/"protected-input-hashes.json")
    assert all(m.p.digest(m.ROOT/name) == expected for name,expected in protected.items())
    result = m.p.read(m.RUN/"comparison.json")
    assert result["executed_generated_commands"] == 0 and not result["training_started"]
    assert result["not_task_success_rate"]
    for variant in m.p.VARIANTS:
        report = result["variants"][variant]
        assert len(report["cases"]) == report["recorded_cases"] <= 4
        assert report["exact_reference_count"] <= report["recorded_cases"]


def test_paired_prompts_identical_and_finish_reasons_based_on_ids():
    for task in m.p.CASES:
        paths = [m.RUN/v/"prompts"/(task+".json") for v in m.p.VARIANTS]
        if all(path.exists() for path in paths):
            assert m.p.read(paths[0]) == m.p.read(paths[1])
    for variant in m.p.VARIANTS:
        for path in (m.RUN/variant/"raw").glob("*.json"):
            raw = m.p.read(path)
            assert raw["finish_reason"] in ("eos","length","unknown")
            assert 0 < len(raw["token_ids"]) <= 256


def test_completed_comparison_and_measured_dtype_receipts():
    report = m.p.read(m.RUN/"comparison.json")
    assert report["full_comparison_completed"]
    for variant in m.p.VARIANTS:
        assert report["variants"][variant]["supervisor"]["exit_code"] == 0
        assert report["variants"][variant]["recorded_cases"] == 4
        load = m.p.read(m.RUN/variant/"model-load.json")
        assert load["requested_base_dtype"] == variant and not load["training"]
        assert load["active_adapters"] == ["default"] and load["torch_threads"] == 4
        assert f"cpu/torch.{variant}" in load["parameter_numel"]
        assert load["use_cache"]
