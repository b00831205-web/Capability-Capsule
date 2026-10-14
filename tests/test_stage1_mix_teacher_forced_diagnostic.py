"""Synthetic causal scoring tests: no model download, generation or tool execution."""

import math
from types import SimpleNamespace

import pytest
import torch

import run_stage1_mix_teacher_forced_diagnostic as probe


def test_causal_prediction_positions_exclude_masked_and_shift_in_forward():
    assert probe.prediction_positions([-100,-100,2,-100,4]) == [2,4]
    with pytest.raises(ValueError,match="Position zero"):
        probe.prediction_positions([2,-100])
    seen = {}
    def model(**kwargs):
        assert torch.is_inference_mode_enabled()
        assert "labels" not in kwargs and not kwargs["use_cache"]
        seen.update(kwargs)
        return SimpleNamespace(logits=torch.tensor([[[0.,0.,8.,0.,0.],[0.,0.,0.,0.,8.]]]))
    example = {"input_ids":[0,1,2,3,4],"attention_mask":[1]*5}
    meta = [{"position":2,"target_id":2,"categories":["all_supervised"]},
            {"position":4,"target_id":4,"categories":["all_supervised"]}]
    scored = probe.forward_scores(model,example,meta)
    assert seen["logits_to_keep"].tolist() == [1,3]
    assert all(r["top1_correct"] for r in scored)
    assert seen["input_ids"].tolist() == [example["input_ids"]]


def test_probabilities_ranks_and_topk_match_known_distribution():
    logits = torch.tensor([0.,1.,2.,3.,4.,5.])
    row = probe.score_row(logits,3)
    assert row["rank"] == 3 and not row["top1_correct"] and row["top5_contains_target"]
    assert row["target_probability"] == pytest.approx(float(logits.softmax(0)[3]),rel=1e-5)
    assert row["nll"] == pytest.approx(-math.log(row["target_probability"]))
    assert not probe.score_row(logits,0)["top5_contains_target"]
    with pytest.raises(ValueError,match="Non-finite"):
        probe.score_row(torch.tensor([float("nan"),0.]),1)


def test_ties_and_nonunique_categories_are_explicit():
    row = probe.score_row(torch.tensor([1.,1.,0.]),1)
    assert row["rank"] == 1 and not row["top1_correct"]
    records = [{**row,"categories":["all_supervised","replacement_literal","literal_punctuation"]}]
    metrics = probe.aggregate(records)
    assert metrics["all_supervised"]["token_count"] == 1
    assert metrics["replacement_literal"]["token_count"] == 1
    assert metrics["literal_punctuation"]["token_count"] == 1
    assert metrics["eos"] == {"token_count":0}


def test_literal_spans_keep_dynamic_names_and_punctuation_separate():
    text = "<parameter=cmd>\n$x.Replace('Hey: {name}', 'Build / {name}!')\n</parameter></function></tool_call>"
    spans = probe.mark_intervals(text)
    assert [text[a:b] for a,b in spans["source_literal"]] == ["Hey: {name}"]
    assert [text[a:b] for a,b in spans["replacement_literal"]] == ["Build / {name}!"]
    assert [text[a:b] for a,b in spans["literal_punctuation"]] == [":","/","!"]
    assert [text[a:b] for a,b in spans["dynamic_placeholder"]] == ["{name}","{name}"]
    assert len(spans["tool_closing"]) == 3
    assert probe.overlaps((2,5),(4,7)) and not probe.overlaps((2,4),(4,7))


def test_aggregation_is_token_weighted_not_case_mean():
    def row(nll,correct):
        return {"nll":nll,"target_probability":math.exp(-nll),"top1_correct":correct,
            "top5_contains_target":True,"categories":["all_supervised"]}
    metrics = probe.aggregate([row(1,True),row(3,False),row(5,False)])["all_supervised"]
    assert metrics["token_count"] == 3 and metrics["mean_nll"] == 3
    assert metrics["top1_accuracy"] == pytest.approx(1/3)


def test_bad_sparse_logits_shape_fails_closed():
    def model(**kwargs):
        return SimpleNamespace(logits=torch.zeros((1,3,6)))
    with pytest.raises(ValueError,match="sparse"):
        probe.forward_scores(model,{"input_ids":[0,1],"attention_mask":[1,1]},
            [{"position":1,"target_id":1,"categories":[]}])
