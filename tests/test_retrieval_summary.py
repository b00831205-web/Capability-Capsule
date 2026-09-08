import importlib

import pytest

from capability_capsule.eval.retrieval import RetrievalMetrics


@pytest.mark.parametrize(
    ("rows", "expected"),
    [
        ([(True, 1.0, 1.0)], (1.0, 1.0, 1.0)),
        ([(False, 0.0, 0.0), (False, 0.0, 0.0)], (0.0, 0.0, 0.0)),
        (
            [(True, 1.0, 1.0), (True, 0.5, 0.5), (False, 0.0, 0.0)],
            (2 / 3, 0.5, 0.5),
        ),
        ([(True, 0.25, 0.5), (True, 1.0, 1.0)], (1.0, 0.625, 0.75)),
    ],
)
def test_summarize_retrieval(
    rows: list[tuple[bool, float, float]],
    expected: tuple[float, float, float],
) -> None:
    module = importlib.import_module("capability_capsule.eval.retrieval")
    metrics = [
        RetrievalMetrics(hit=hit, recall=recall, reciprocal_rank=rank) for hit, recall, rank in rows
    ]
    before = [metric.model_dump() for metric in metrics]
    summary = module.summarize_retrieval(metrics)
    assert summary.query_count == len(rows)
    assert summary.hit_rate == pytest.approx(expected[0])
    assert summary.mean_recall == pytest.approx(expected[1])
    assert summary.mrr == pytest.approx(expected[2])
    assert [metric.model_dump() for metric in metrics] == before
    reversed_summary = module.summarize_retrieval(tuple(reversed(metrics)))
    assert reversed_summary.query_count == summary.query_count
    assert reversed_summary.hit_rate == pytest.approx(summary.hit_rate)
    assert reversed_summary.mean_recall == pytest.approx(summary.mean_recall)
    assert reversed_summary.mrr == pytest.approx(summary.mrr)


def test_summarize_retrieval_rejects_empty_input() -> None:
    module = importlib.import_module("capability_capsule.eval.retrieval")
    with pytest.raises(ValueError):
        module.summarize_retrieval(())
