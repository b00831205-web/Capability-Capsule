"""Run the fixed v3 command-policy comparison with the existing step-32 adapter."""

import json
import shutil
import sys
from hashlib import sha256
from pathlib import Path

from capability_capsule.eval.coding_checkpoint import (
    CodingEvaluationCase,
    ConstrainedPowerShellExecutor,
    TransformersPeftTurnGenerator,
    copy_verified_evaluation_fixture,
    evaluate_coding_case,
    load_coding_evaluation_suite,
)
from capability_capsule.eval.fixture_provenance import inspect_fixture
from capability_capsule.eval.jsonl import append_jsonl, load_jsonl
from capability_capsule.eval.learning_rate import LearningCurvePoint


ROOT = Path(__file__).resolve().parents[1]
SUITE_PATH = ROOT / "plans/evaluation/stage1-codex-validation-v3-command-policy/evaluation-suite.json"
PREVIOUS_RUN = ROOT / "runs/evaluation/stage1-codex-qwen35-2b-006-turns-v2-v2"
RUN = ROOT / "runs/evaluation/stage1-codex-qwen35-2b-006-turns-v2-v3-command-policy"
CHECKPOINT = ROOT / "runs/training/stage1-codex-qwen35-2b-006-turns-v2/checkpoint.json"
TOOLS = (
    {
        "type": "function",
        "function": {
            "name": "exec_command",
            "description": "Run one PowerShell command in the current workspace.",
            "parameters": {
                "type": "object",
                "properties": {"cmd": {"type": "string"}},
                "required": ["cmd"],
                "additionalProperties": False,
            },
        },
    },
)


def main() -> None:
    global RUN
    if sys.argv[1:] == ["--raw-order-rerun"]:
        RUN = ROOT / "runs/evaluation/stage1-codex-qwen35-2b-006-turns-v2-v3-command-policy-raw-order"
    elif sys.argv[1:]:
        raise ValueError("Only --raw-order-rerun is supported")
    v2 = load_coding_evaluation_suite(PREVIOUS_RUN / "evaluation-suite.json")
    v3 = load_coding_evaluation_suite(SUITE_PATH)
    if v3.cases != v2.cases or v3.identity().evaluation_suite_digest != v2.identity().evaluation_suite_digest:
        raise ValueError("The v3 cases or validators differ from the fixed v2 suite")

    if RUN.exists():
        existing = tuple(path for path in RUN.rglob("*") if path.is_file())
        if existing != (RUN / "evaluation-suite.json",):
            raise FileExistsError(RUN)
        if (RUN / "evaluation-suite.json").read_bytes() != SUITE_PATH.read_bytes():
            raise ValueError("Partial run contains a different suite")
    else:
        RUN.mkdir(parents=True)
        shutil.copyfile(SUITE_PATH, RUN / "evaluation-suite.json")
    (RUN / "case-reports").mkdir(exist_ok=True)
    generator = TransformersPeftTurnGenerator(CHECKPOINT, max_new_tokens=256)
    executor = ConstrainedPowerShellExecutor()
    old_report = json.loads(
        (PREVIOUS_RUN / "case-reports" / f"{v2.cases[0].case_id}.json").read_text("utf-8")
    )
    hardware_id = old_report["result"]["hardware_id"]
    results = []
    fixture_sources = {}
    for position, case in enumerate(v3.cases, start=1):
        workspace = RUN / "workspaces" / case.case_id
        source = ROOT / case.fixture_root
        if inspect_fixture(case.task.fixture_id, source).revision != case.task.fixture_revision:
            if case.case_id != v3.cases[0].case_id:
                raise ValueError(f"No approved snapshot fallback for {case.case_id}")
            source = PREVIOUS_RUN / "workspaces" / case.case_id
            if inspect_fixture(case.task.fixture_id, source).revision != case.task.fixture_revision:
                raise ValueError(f"Neither source matches the pinned revision for {case.case_id}")
        relative_source = source.relative_to(ROOT).as_posix()
        fixture_sources[case.case_id] = relative_source
        copy_case = CodingEvaluationCase.model_validate(
            {**case.model_dump(mode="json"), "fixture_root": relative_source}
        )
        copy_verified_evaluation_fixture(copy_case, artifact_root=ROOT, destination=workspace)
        outcome = evaluate_coding_case(
            case,
            workspace=workspace,
            generator=generator,
            executor=executor,
            system_prompt=v3.system_prompt,
            tools=TOOLS,
            experiment_id="stage1-codex-qwen35-2b-006-turns-v2-validation-v3-command-policy",
            run_id=f"stage1-codex-qwen35-2b-006-turns-v2-v3-command-policy:{case.case_id}",
            model_id="capsule-stage1-codex-qwen35-2b-006-turns-v2",
            hardware_id=hardware_id,
            cycle_position=position,
            max_tool_rounds=4,
        )
        results.append(outcome.result)
        append_jsonl(RUN / "case-results.jsonl", outcome.result)
        report = {
            "result": outcome.result.model_dump(mode="json"),
            "completions": list(outcome.completions),
            "tool_results": [item.model_dump(mode="json") for item in outcome.tool_results],
            "validators": [item.model_dump(mode="json") for item in outcome.validators],
        }
        (RUN / "case-reports" / f"{case.case_id}.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(f"{case.case_id}: success={outcome.result.success}, invalid={outcome.result.invalid_tool_call_count}", flush=True)

    old_point = load_jsonl(PREVIOUS_RUN / "learning-curve.jsonl", LearningCurvePoint)[0]
    point = LearningCurvePoint.model_validate(
        {
            **old_point.model_dump(),
            "evaluation_suite_id": v3.evaluation_suite_id,
            "evaluation_suite_digest": v3.identity().evaluation_suite_digest,
            "success_rate": sum(item.success for item in results) / len(results),
        }
    )
    append_jsonl(RUN / "learning-curve.jsonl", point)
    manifest = {
        "checkpoint": str(CHECKPOINT.relative_to(ROOT)).replace("\\", "/"),
        "checkpoint_sha256": sha256(CHECKPOINT.read_bytes()).hexdigest(),
        "suite": str(SUITE_PATH.relative_to(ROOT)).replace("\\", "/"),
        "suite_sha256": sha256(SUITE_PATH.read_bytes()).hexdigest(),
        "system_prompt_sha256": sha256(v3.system_prompt.encode("utf-8")).hexdigest(),
        "case_digest": v3.identity().evaluation_suite_digest,
        "baseline_run": str(PREVIOUS_RUN.relative_to(ROOT)).replace("\\", "/"),
        "max_new_tokens": 256,
        "max_tool_rounds": 4,
        "case_count": len(results),
        "success_count": sum(item.success for item in results),
        "fixture_sources": fixture_sources,
    }
    (RUN / "run-manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
