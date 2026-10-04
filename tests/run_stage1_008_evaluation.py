"""Matched v3 regression plus separately reported teacher-forced edit probes."""

import json
import shutil
import sys
from datetime import UTC, datetime
from hashlib import sha256

import run_stage1_007_evaluation as previous
import train_stage1_edit005_64step as control
from capability_capsule.eval.coding_checkpoint import (
    CodingEvaluationCase, TransformersPeftTurnGenerator, copy_verified_evaluation_fixture,
    load_coding_evaluation_suite, parse_qwen_tool_completion, _GUARDED_REPLACE, _PIPE_REPLACE,
)
from capability_capsule.eval.fixture_provenance import inspect_fixture
from capability_capsule.eval.records import TeacherTrajectory
from capability_capsule.training.sft import SFTExportManifest, _chat_messages

ROOT = control.ROOT
RUN_ID = control.RUN_ID
RUN = ROOT / "runs/evaluation" / RUN_ID
OLD_EVAL = previous.RUN
TRAINING = control.RUN


def protected_hashes():
    hashes = control.protected_hashes()
    for path in TRAINING.rglob("*"):
        if path.is_file():
            hashes[path.relative_to(ROOT).as_posix()] = sha256(path.read_bytes()).hexdigest()
    return hashes


def prepare():
    if RUN.exists():
        raise FileExistsError(RUN)
    checkpoint = TRAINING / "checkpoint.json"
    before = protected_hashes()
    reload_report = json.loads((TRAINING / "adapter-reload-validation.json").read_text("utf-8"))
    if not reload_report["independent_process"] or not reload_report["adapter_hashes_verified"]:
        raise ValueError("Independent adapter reload must pass first")
    RUN.mkdir(parents=True)
    previous.save(RUN / "protected-input-hashes.json", before)
    sources, groups = {}, {}
    old_prepared = json.loads((OLD_EVAL / "prepared.json").read_text("utf-8"))
    for group in ("fixed-v3", "heldout-v1"):
        directory = RUN / group
        directory.mkdir()
        old_suite = OLD_EVAL / group / "evaluation-suite.json"
        shutil.copyfile(old_suite, directory / "evaluation-suite.json")
        suite = load_coding_evaluation_suite(old_suite)
        for case in suite.cases:
            relative = (old_prepared["fixture_sources"][case.case_id]
                        if group == "fixed-v3" else case.fixture_root)
            copy_case = CodingEvaluationCase.model_validate({**case.model_dump(mode="json"), "fixture_root": relative})
            copy_verified_evaluation_fixture(copy_case, artifact_root=ROOT,
                destination=directory / "workspaces" / case.case_id)
            sources[case.case_id] = relative
        groups[group] = {"suite_sha256": previous.digest(directory / "evaluation-suite.json"),
            "case_digest": suite.identity().evaluation_suite_digest}
    previous.save(RUN / "prepared.json", {"prepared_at": datetime.now(UTC).isoformat(),
        "checkpoint_sha256": previous.digest(checkpoint), "fixture_sources": sources,
        "executor_sha256": previous.digest(previous.EXECUTOR),
        "tools_sha256": sha256(json.dumps(previous.TOOLS, sort_keys=True).encode()).hexdigest(),
        "groups": groups, "validation_scope": "Identical run-007 cases, including repeated heldout fixtures; not newly unseen."})
    print("Pinned identical 3 fixed and 2 heldout-regression cases for 008.", flush=True)


def probe_edits(generator):
    output_path = RUN / "edit-turn-probes.json"
    if output_path.exists():
        raise FileExistsError(output_path)
    trajectories = [TeacherTrajectory.model_validate_json(line) for line in
        (control.PUBLICATION / "train.jsonl").read_text("utf-8").splitlines()]
    manifest = SFTExportManifest.model_validate_json((control.SFT / "manifest.json").read_bytes())
    v3 = load_coding_evaluation_suite(RUN / "fixed-v3/evaluation-suite.json")
    results = []
    for position in (0, 7):
        trajectory = trajectories[position]
        messages = _chat_messages(trajectory, chat_contract=manifest.chat_contract)
        expected = messages[4]["tool_calls"][0]["function"]["arguments"]["cmd"]
        source = json.loads(messages[3]["content"])["output"]
        oracle = _GUARDED_REPLACE.fullmatch(expected) or _PIPE_REPLACE.fullmatch(expected)
        target = source.replace(oracle.group("old"), oracle.group("new"))
        for prompt, system in (("v2-training", manifest.chat_contract.system_prompt), ("v3-evaluation", v3.system_prompt)):
            context = [dict(message) for message in messages[:4]]
            context[0] = {"role": "system", "content": system}
            print(f"Edit probe {trajectory.trajectory_id}: {prompt}", flush=True)
            generated = generator.generate(context, manifest.chat_contract.tools)
            result = {"trajectory_id": trajectory.trajectory_id, "prompt": prompt,
                "completion": generated.completion, "input_tokens": generated.input_tokens,
                "output_tokens": generated.output_tokens, "expected_command": expected,
                "supported_edit": False, "target_file_matches_teacher": False}
            try:
                parsed = parse_qwen_tool_completion(generated.completion)
                command = (parsed.tool_calls[0].arguments.get("cmd")
                    if len(parsed.tool_calls) == 1 and parsed.tool_calls[0].name == "exec_command" else None)
                match = (_GUARDED_REPLACE.fullmatch(command.strip()) or _PIPE_REPLACE.fullmatch(command.strip())) if isinstance(command, str) else None
                accepted = bool(match) and not any(token in command for token in ("..", ":", "/", "\\"))
                result.update(command=command, supported_edit=accepted,
                    target_file_matches_teacher=accepted and source.replace(match.group("old"), match.group("new")) == target)
            except Exception as error:
                result["parse_error"] = f"{type(error).__name__}: {error}"
            results.append(result)
            print(f"supported={result['supported_edit']}, target={result['target_file_matches_teacher']}", flush=True)
    previous.save(output_path, {"scope": "Teacher-forced single edit turn on two training examples; no commands executed, not heldout or closed-loop scores.",
        "max_new_tokens": 256, "do_sample": False, "results": results})


def evaluate():
    if (RUN / "evaluation-summary.json").exists():
        raise FileExistsError("Evaluation already complete")
    previous.RUN_ID, previous.RUN, previous.TRAINING = RUN_ID, RUN, TRAINING
    previous.BASELINE = OLD_EVAL / "fixed-v3"
    previous.protected_hashes = protected_hashes
    cached = []

    def shared_generator(checkpoint, *, max_new_tokens):
        if not cached:
            cached.append(TransformersPeftTurnGenerator(checkpoint, max_new_tokens=max_new_tokens))
        return cached[0]

    previous.TransformersPeftTurnGenerator = shared_generator
    previous.evaluate()
    probe_edits(cached[0])
    summaries = {}
    for group in ("fixed-v3", "heldout-v1"):
        path = RUN / group / "run-manifest.json"
        manifest = json.loads(path.read_text("utf-8"))
        manifest.update(baseline_run=(OLD_EVAL / group).relative_to(ROOT).as_posix(),
            validation_scope="Repeated immutable run-007 cases; heldout-v1 is a regression set, not newly unseen.",
            ledger_scope="Fresh-base edit005-only run: 12 unique trajectories / 8169 tokens, not cumulative across 007/008.")
        previous.save(path, manifest)
        summaries[group] = manifest
    previous.save(RUN / "evaluation-summary.json", summaries)
    before = json.loads((RUN / "protected-input-hashes.json").read_text("utf-8"))
    after = protected_hashes()
    changed = sorted(p for p in set(before) | set(after) if before.get(p) != after.get(p))
    previous.save(RUN / "input-preservation.json", {"unchanged": not changed,
        "protected_file_count": len(before), "changed_paths": changed})
    if changed:
        raise ValueError("Protected inputs changed during probes")


if __name__ == "__main__":
    if sys.argv[1:] == ["--prepare-only"]:
        prepare()
    elif sys.argv[1:] == ["--evaluate-prepared"]:
        evaluate()
    elif not sys.argv[1:]:
        prepare()
        evaluate()
    else:
        raise ValueError("Use --prepare-only or --evaluate-prepared")
