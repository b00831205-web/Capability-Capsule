"""Train provenance and answer exclusion for the proposed diagnostic."""

import copy

import pytest

import prepare_stage1_mix011_train_fit_diagnostic as preparation


def test_readonly_preparation_matches_all_14_seen_cases_without_relabeling():
    plan = preparation.build_plan()
    assert plan["checkpoint_step"] == 28 and plan["case_count"] == 14
    assert all(case["split"] == "train" and case["seen_during_training"] for case in plan["cases"])
    assert not plan["inference_started"] and not plan["artifact_writes_started"]
    assert plan["max_tool_calls"] == 4 and plan["max_generated_turns"] == 5
    assert all(not case["task_id"].startswith("validation-") for case in plan["cases"])


def test_model_context_contains_only_original_prompt_not_teacher_answer():
    contract = {"system_prompt": "Original system", "answer": "Hidden answer"}
    task = {"task": "Original request", "reference": "Hidden reference", "messages": ["Teacher reply"]}
    assert preparation.model_messages(contract, task) == [
        {"role": "system", "content": "Original system"},
        {"role": "user", "content": "Original request"},
    ]


def test_training_case_rejects_relabeling_and_sft_provenance_mismatch():
    assignment = preparation.read(preparation.ROOT / "plans/teacher" / preparation.PLAN_IDS[0] / "collection-plan.json")["assignments"][0]
    record = preparation.records(preparation.PUBLICATION / "train.jsonl")[0]
    example = preparation.records(preparation.SFT / "train.jsonl")[0]
    relabeled = copy.deepcopy(assignment)
    relabeled["task"]["split"] = "validation"
    with pytest.raises(ValueError, match="original train labels"):
        preparation.check_case(record, relabeled, example)
    mismatched = {**example, "source_revision": "0" * 64}
    with pytest.raises(ValueError, match="correspondence mismatch"):
        preparation.check_case(record, assignment, mismatched)
