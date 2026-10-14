"""Freeze six validation-only cases and real reference preflights; no inference."""

import ast
import json
import os
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from hashlib import sha256

import prepare_stage1_literal006_fstring007_sft as mix
from capability_capsule.eval.coding_checkpoint import (
    CodingEvaluationCase, CodingEvaluationSuite, ConstrainedPowerShellExecutor,
    copy_verified_evaluation_fixture, load_coding_evaluation_suite, run_coding_validators,
)
from capability_capsule.eval.fixture_provenance import inspect_fixture
from capability_capsule.eval.jsonl import load_jsonl
from capability_capsule.eval.records import TeacherTrajectory
from capability_capsule.eval.student_target import verify_student_target
from capability_capsule.eval.teacher_collection_plan_publication import load_teacher_collection_plan

ROOT = mix.ROOT
ID = "stage1-codex-mix011-validation-v1"
OUTPUT = ROOT / "artifacts/evaluation-fixtures" / ID
FIXED = ROOT / "plans/evaluation/stage1-codex-validation-v2/evaluation-suite.json"
BASELINE = ROOT / "runs/training/stage1-codex-qwen35-2b-010-literal006-64step-retry1/checkpoint.json"
EXECUTOR = ROOT / "src/capability_capsule/eval/coding_checkpoint.py"
SFT_DIGEST = "5d8747cbf4ea5d1f023ecfd12694ec7b839af49701b7b59b6c9c06e27ef823cc"
FIXED_DIGEST = "53687e1d7c56a854f4a07ecf38f1c4446188694a74f2d1fae76ebc6028b1d91e"
EXECUTOR_DIGEST = "c1b3ec738da80c59da83411adc0884223b36b372041b45390ffa0f9834381356"


@dataclass(frozen=True)
class Variant:
    slug: str
    group: str
    source: str
    old: str
    new: str
    function: str
    checks: tuple
    task: str


VARIANTS = (
    Variant("conditional-fstring", "fstring", '# Preserve the anonymous fallback expression.\n'
        'def greeting(name: str) -> str:\n    return f"Hello voyager, {name if name else \'guest\'}!"\n',
        "Hello voyager", "Bright horizons", "greeting",
        ((("Iris",), "Bright horizons, Iris!"), (("Tess",), "Bright horizons, Tess!"), (("",), "Bright horizons, guest!")),
        'Change greeting.py only so named callers receive "Bright horizons, <name>!". '
        'An empty name must still use guest. Preserve the conditional expression and comment. Run the supplied tests without editing them.'),
    Variant("width-fstring", "fstring", '# Retain right alignment and its minimum field width.\n'
        'def greeting(name: str) -> str:\n    return f"Hi traveller {name:>8}"\n',
        "Hi traveller", "Steady stride", "greeting",
        ((("Uma",), "Steady stride      Uma"), (("Li",), "Steady stride       Li"),
         (("",), "Steady stride         "), (("LongSurname",), "Steady stride LongSurname")),
        'Change the leading words in greeting.py to "Steady stride". Keep the separating space and '
        'the existing right-aligned eight-character minimum name field, including empty and long names. Validate without editing test_greeting.py.'),
    Variant("branch-local-mapping", "structure", '# Keep the inactive template and missing-name path.\n'
        'def greeting(name: str) -> str:\n'
        '    templates = {"active": "Hello guest {name}", "spare": "Keep spare notice"}\n'
        '    cleaned = name.strip()\n    if not cleaned:\n        return "No caller"\n'
        '    display = cleaned.casefold()\n    return templates["active"].format(name=display)\n',
        "Hello guest", "Clear passage", "greeting",
        (((" IVY ",), "Clear passage ivy"), (("NOEL",), "Clear passage noel"), (("",), "No caller"), (("  ",), "No caller")),
        'Make greeting.py use "Clear passage <normalized name>" for nonblank callers. '
        'Preserve trimming, case folding, both local templates, comment and the No caller fallback. Run the supplied tests unchanged.'),
    Variant("branch-list-assembly", "structure", '# Leave the helper and absent-participant branch intact.\n'
        'def greeting(name: str) -> str:\n    if not name:\n        return "Vacant participant"\n'
        '    segments = ["Welcome", " - ", name.swapcase(), "."]\n    return "".join(segments)\n\n'
        'def unrelated_value() -> int:\n    return 41\n',
        "Welcome", "New chapter", "greeting",
        ((("Lea",), "New chapter - lEA."), (("ROAN",), "New chapter - roan."), (("",), "Vacant participant")),
        'Use "New chapter" as the greeting prefix in greeting.py. Preserve list assembly, joining, '
        'case swapping, punctuation, the empty-name response, comment and unrelated_value. Validate without changing tests.'),
    Variant("job-status", "business-transfer", '# Keep count formatting and the invalid-count response.\n'
        'def status_line(job: str, count: int) -> str:\n    if count < 0:\n        return "Invalid count"\n'
        '    return f"Queued {job} ({count:03d})"\n',
        "Queued", "Dispatch set", "status_line",
        ((("mesh", 7), "Dispatch set mesh (007)"), (("index", 24), "Dispatch set index (024)"),
         (("", 0), "Dispatch set  (000)"), (("mesh", -1), "Invalid count")),
        'In greeting.py, status_line should describe nonnegative jobs as "Dispatch set <job> (<count>)". '
        'Keep three-digit zero padding, both parameters, the invalid-count branch and comment. Run test_greeting.py without editing it.'),
    Variant("audit-message", "business-transfer", '# Preserve normalization of both event fields.\n'
        'def audit_line(actor: str, action: str) -> str:\n    if not actor.strip():\n        return "Anonymous actor"\n'
        '    entry = {"actor": actor.strip().upper(), "action": action.casefold()}\n'
        '    return "Logged {action} by {actor}".format(**entry)\n',
        "Logged", "Recorded event", "audit_line",
        (((" neri ", "SAVE"), "Recorded event save by NERI"), (("oma", "LOAD"), "Recorded event load by OMA"),
         (("  ", "SAVE"), "Anonymous actor"), (("neri", ""), "Recorded event  by NERI")),
        'Update audit_line in greeting.py so identified actors produce "Recorded event <action> by <actor>". '
        'Keep actor trimming/uppercasing, action case folding, dictionary formatting, both dynamic fields, '
        'the anonymous-actor response and comment. Validate without changing test_greeting.py.'),
)


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def protected_hashes():
    hashes = mix.protected_hashes()
    for directory in (mix.PUBLICATION, mix.OUTPUT):
        for path in directory.iterdir():
            hashes[path.relative_to(ROOT).as_posix()] = digest(path)
    hashes[mix.EVIDENCE.relative_to(ROOT).as_posix()] = digest(mix.EVIDENCE)
    prefix = OUTPUT.relative_to(ROOT).as_posix() + "/"
    return {path:value for path,value in hashes.items() if not path.startswith(prefix)}


def tests_for(variant):
    text = f"from greeting import {variant.function}\n\ndef test_multiple_dynamic_inputs():\n"
    for arguments, expected in variant.checks:
        text += f"    assert {variant.function}({', '.join(repr(a) for a in arguments)}) == {expected!r}\n"
    if variant.slug == "branch-list-assembly":
        text += '\n    from greeting import unrelated_value\n    assert unrelated_value() == 41\n'
    return text


def make_case(variant):
    case_id = "validation-mix011-" + variant.slug + "-001"
    source = OUTPUT / "sources" / variant.slug
    source.mkdir(parents=True, exist_ok=True)
    tests = tests_for(variant)
    for filename, payload in (("greeting.py", variant.source.encode()), ("test_greeting.py", tests.encode())):
        path = source / filename
        if path.exists():
            if path.read_bytes() != payload:
                raise ValueError("Unexplained changes in an unfinished snapshot")
        else:
            with path.open("xb") as stream:
                stream.write(payload)
    if variant.source.count(variant.old) != 1:
        raise ValueError("Reference must have exactly one replacement")
    validators = [
        {"validator_id": "pytest:test_greeting.py", "kind": "pytest", "target": "test_greeting.py"},
        {"validator_id": "exact-content:greeting.py", "kind": "exact_text", "target": "greeting.py",
         "expected": variant.source.replace(variant.old, variant.new)},
        {"validator_id": "unchanged:test_greeting.py", "kind": "exact_text", "target": "test_greeting.py", "expected": tests},
    ]
    for index, (arguments, expected) in enumerate(variant.checks):
        validators.append({"validator_id": f"python-assert:input-{index}", "kind": "python_call",
            "target": "greeting.py", "function": variant.function, "arguments": arguments, "expected": expected})
    task = {"task_id": case_id, "fixture_id": case_id, "fixture_family_id": "mix011-" + variant.slug,
        "fixture_revision": inspect_fixture(case_id, source).revision, "defect_family": "text-preservation-" + variant.slug,
        "split": "validation", "category": "single_file_change", "difficulty": "easy", "task": variant.task,
        "knowledge_distance": 0.4 if variant.group != "business-transfer" else 0.7,
        "logical_arrival": "PT0S", "time_limit": "PT3M", "allowed_tools": ["exec_command"],
        "expected_changes": [{"path": "greeting.py", "operation": "modify"}],
        "validation_ids": [v["validator_id"] for v in validators]}
    return CodingEvaluationCase.model_validate({"case_id": case_id, "fixture_root": source.relative_to(ROOT).as_posix(),
        "task": task, "validators": validators})


def audit_overlap(suite):
    training_paths = list((ROOT / "datasets/teacher").glob("*/raw.jsonl")) + list(
        (ROOT / "datasets/teacher/published").glob("*/train.jsonl"))
    revisions, ids, texts, sources, messages = set(), set(), set(), set(), []
    legacy_plaintext_results = 0
    for path in training_paths:
        for record in load_jsonl(path, TeacherTrajectory):
            if record.split.value != "train":
                continue
            revisions.add(record.source_revision)
            ids.add(record.task_id)
            texts.add(record.task)
            messages.append(record.model_dump_json())
            for message in record.messages:
                if message.role.value == "tool":
                    try:
                        payload = json.loads(message.content)
                    except json.JSONDecodeError:
                        sources.add(message.content)
                        legacy_plaintext_results += 1
                        continue
                    if isinstance(payload, dict) and isinstance(payload.get("output"), str):
                        sources.add(payload["output"])
    prior_ids, prior_texts, prior_revisions, prior_sources = set(), set(), set(), set()
    prior_paths = set((ROOT / "plans/evaluation").rglob("evaluation-suite.json")) | set(
        (ROOT / "runs/evaluation").rglob("evaluation-suite.json")) | set(
        (ROOT / "artifacts/evaluation-fixtures").rglob("evaluation-suite.json"))
    for path in prior_paths:
        if OUTPUT in path.parents:
            continue
        for case in load_coding_evaluation_suite(path).cases:
            if case.task.split.value != "validation":
                raise ValueError("Do not inspect locked-test cases")
            # Later evaluations may mirror this frozen suite byte-for-byte. They do not
            # retroactively become pre-freeze cases; the first-use statement is historical.
            origin = next((item for item in suite.cases if item.case_id == case.case_id), None)
            if origin is not None:
                if case != origin:
                    raise ValueError("Conflicting derived validation case")
                continue
            prior_ids.add(case.case_id)
            prior_texts.add(case.task.task)
            prior_revisions.add(case.task.fixture_revision)
            initial = ROOT / case.fixture_root / "greeting.py"
            if initial.exists():
                prior_sources.add(initial.read_text("utf-8"))
    train_text = "\n".join(messages)
    for case, variant in zip(suite.cases, VARIANTS, strict=True):
        if (case.task.fixture_revision in revisions | prior_revisions or case.task.task_id in ids | prior_ids
                or case.task.task in texts | prior_texts or variant.source in sources | prior_sources):
            raise ValueError("Known exact train/prior-validation overlap")
        if variant.new in train_text:
            raise ValueError("New target prefix already appears in train")
        # Verify all changed full outputs; unchanged fallback text is intentionally preserved.
        for arguments, expected in variant.checks:
            if expected.startswith(variant.new) and expected in train_text:
                raise ValueError("New validation answer occurs in training messages")
    return {"known_exact_overlap": False, "training_task_count": len(ids), "training_revision_count": len(revisions),
        "legacy_plaintext_tool_results_reviewed": legacy_plaintext_results,
        "prior_validation_case_count": len(prior_ids), "prior_validation_revision_count": len(prior_revisions),
        "scope": "Known exact sources/tasks/revisions/changed targets only; not semantic independence or broad generalization"}


def preflight(case, variant):
    original_dir = OUTPUT / "preflight" / variant.slug / "original"
    reference_dir = OUTPUT / "preflight" / variant.slug / "reference"
    for directory in (original_dir, reference_dir):
        copy_verified_evaluation_fixture(case, artifact_root=ROOT, destination=directory)
    original = run_coding_validators(case, original_dir)
    executor = ConstrainedPowerShellExecutor()
    commands = ("Get-Content -LiteralPath greeting.py -Raw",
        f"$source = Get-Content -LiteralPath greeting.py -Raw; $updated = $source.Replace('{variant.old}', '{variant.new}'); Set-Content -LiteralPath greeting.py -Value $updated",
        "python -m pytest -q test_greeting.py")
    executions = []
    for command in commands:
        result = executor.execute(command, reference_dir)
        executions.append({"cmd": command, "result": result.model_dump(mode="json")})
        if not result.authorized or result.exit_code != 0:
            raise ValueError(f"Reference command failed: {variant.slug}: {result.output}")
    corrected = run_coding_validators(case, reference_dir)
    if original[0].passed or original[1].passed or not original[2].passed or not all(v.passed for v in corrected):
        raise ValueError("Expected original failure/reference success")
    if not any(not v.passed for v in original if v.validator_id.startswith("python-assert:")):
        raise ValueError("Behavior validator does not detect original defect")
    report = {"case_id": case.case_id, "group": variant.group,
        "original": [v.model_dump(mode="json") for v in original],
        "reference": [v.model_dump(mode="json") for v in corrected], "actual_reference_calls": executions,
        "source_snapshot_unchanged": inspect_fixture(case.task.fixture_id, ROOT / case.fixture_root).revision == case.task.fixture_revision,
        "scope": "Deterministic original/reference preflight; not Teacher trajectory or Student inference"}
    save(OUTPUT / "preflight" / variant.slug / "report.json", report)
    print(f"{variant.slug}: original failed / reference passed, three authorized real calls", flush=True)
    return report


def verify_freeze():
    manifest = json.loads((OUTPUT / "freeze-manifest.json").read_text("utf-8"))
    for relative, pinned in manifest["artifact_hashes"].items():
        if digest(OUTPUT / relative) != pinned:
            raise ValueError("Frozen validation artifact changed")
    actual = {p.relative_to(OUTPUT).as_posix() for p in OUTPUT.rglob("*")
              if p.is_file() and not any(x in {"__pycache__", ".pytest_cache"} for x in p.parts)}
    if actual != set(manifest["artifact_hashes"]) | {"freeze-manifest.json"}:
        raise ValueError("Unexpected frozen artifact set")
    return manifest


def main():
    resume = sys.argv[1:] == ["--resume-preparation"]
    if sys.argv[1:] and not resume:
        raise ValueError("Only --resume-preparation is supported")
    if OUTPUT.exists():
        if not resume:
            raise FileExistsError("Refusing to overwrite frozen validation")
        expected = {f"sources/{variant.slug}/{name}" for variant in VARIANTS for name in ("greeting.py", "test_greeting.py")}
        actual = {p.relative_to(OUTPUT).as_posix() for p in OUTPUT.rglob("*") if p.is_file()}
        if actual != expected:
            raise FileExistsError("Resume is restricted to the twelve untouched, unfrozen source files")
    elif resume:
        raise FileNotFoundError("No interrupted source-only preparation exists")
    for path, pinned in ((mix.OUTPUT / "manifest.json", SFT_DIGEST), (FIXED, FIXED_DIGEST), (EXECUTOR, EXECUTOR_DIGEST)):
        if digest(path) != pinned:
            raise ValueError("Experiment input digest changed")
    fixed, contract = load_coding_evaluation_suite(FIXED), mix.baseline.contract()
    if fixed.system_prompt != contract.system_prompt:
        raise ValueError("System contract mismatch")
    plan = load_teacher_collection_plan(mix.fstring.collection.PLAN_DIR).plan
    verify_student_target(plan.student_target, artifact_root=ROOT)
    before = protected_hashes()
    cases = tuple(make_case(variant) for variant in VARIANTS)
    suite = CodingEvaluationSuite.model_validate({**fixed.model_dump(mode="json"),
        "evaluation_suite_id": ID, "cases": cases})
    overlap = audit_overlap(suite)
    if resume:
        save(OUTPUT / "preparation-interruption.json", {"stage": "known-exact overlap audit before preflights/freeze",
            "cause": "Legacy training tool results include plaintext; initial JSON-only reader rejected one",
            "recovery": "Read both plaintext and JSON; verified all twelve existing source files byte-identical; no source rewrite",
            "model_inference_started": False, "preflights_had_started": False})
    save(OUTPUT / "evaluation-suite.json", suite.model_dump(mode="json"))
    save(OUTPUT / "overlap-audit.json", overlap)
    save(OUTPUT / "protected-input-hashes.json", before)
    # Prevent preflight imports from writing bytecode even when invoked without the documented env.
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    sys.dont_write_bytecode = True
    reports = [preflight(case, variant) for case, variant in zip(cases, VARIANTS, strict=True)]
    if protected_hashes() != before:
        raise ValueError("Existing inputs changed")
    experiment = {"experiment_id": "stage1-codex-qwen35-2b-011-mix006007-28step", "training_authorized": False,
        "model_evaluation_authorized": False, "training_started": False, "model_inference_started": False,
        "sft_export_id": mix.EXPORT_ID, "sft_manifest_sha256": SFT_DIGEST,
        "base_model_id": mix.baseline.MODEL_ID, "base_model_revision": mix.baseline.REVISION,
        "fresh_base": True, "seed": 42, "epochs": 2, "max_steps": 28,
        "train_batch_size": 1, "gradient_accumulation_steps": 1, "learning_rate": 2e-4,
        "lora_rank": 8, "lora_alpha": 16, "lora_dropout": 0.05, "target_modules": ["q_proj", "v_proj"],
        "max_length": 4096, "train_count": 14, "scheduler": "linear decay over 28 updates; no warmup",
        "training_validation": False, "checkpoint_selection": "final step-28 only; do not select using heldout scores",
        "old_comparison_checkpoint": BASELINE.relative_to(ROOT).as_posix(), "old_checkpoint_sha256": digest(BASELINE),
        "fixed_suite_sha256": FIXED_DIGEST, "executor_sha256": EXECUTOR_DIGEST,
        "chat_contract_sha256": contract.sha256(), "tools": list(contract.tools),
        "max_new_tokens": 256, "max_tool_rounds": 4, "do_sample": False, "enable_thinking": False,
        "future_evaluation": "Fixed v2 regression for new final checkpoint; same six frozen cases once each for new checkpoint and predeclared old 010, in separate fresh copies",
        "targets": {"fixed_regression": "3/3", "new_validation": "6/6", "invalid_tool_calls": 0},
        "interpretation": "Limited single-file text-edit transfer; changed data and update budget, not a single-variable comparison; cannot prove no overfitting",
        "no_automatic_extension": True, "hardware_profile_id": plan.student_target.hardware.profile_id,
        "hardware_purposes": list(plan.student_target.hardware.purposes),
        "recommendation_evidence_status": "provisional", "teacher_skill_version": "0.2.0"}
    save(OUTPUT / "experiment-plan.json", experiment)
    save(OUTPUT / "preflight-summary.json", {"case_count": 6, "groups": {g:2 for g in ("fstring", "structure", "business-transfer")},
        "original_failures": 6, "reference_passes": 6, "actual_reference_tool_calls": 18,
        "protected_file_count": len(before), "unchanged": True, "source_snapshot_count": 6,
        "training_started": False, "model_inference_started": False, "case_ids": [r["case_id"] for r in reports]})
    hashes = {p.relative_to(OUTPUT).as_posix():digest(p) for p in OUTPUT.rglob("*")
        if p.is_file() and not any(x in {"__pycache__", ".pytest_cache"} for x in p.parts)}
    save(OUTPUT / "freeze-manifest.json", {"frozen_at": datetime.now(UTC).isoformat(),
        "evaluation_suite_id": ID, "suite_sha256": digest(OUTPUT / "evaluation-suite.json"),
        "case_digest": suite.identity().evaluation_suite_digest, "artifact_hashes": hashes,
        "independence": "First-use cases not yet run with any model; never train on these sources, tasks, references or preflight outputs"})
    verify_freeze()
    print(f"Frozen six cases, all preflights passed; {len(before)} protected inputs unchanged; no training/inference", flush=True)


if __name__ == "__main__":
    main()
