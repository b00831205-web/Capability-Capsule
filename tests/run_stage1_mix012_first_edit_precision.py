"""Eight frozen single-turn generations; interpret literals in memory, never execute."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time
from collections import Counter
from hashlib import sha256

import run_stage1_mix_teacher_forced_diagnostic as prior
from capability_capsule.eval import coding_checkpoint_v2 as evaluator

ROOT = prior.ROOT
RUN = ROOT / "runs/evaluation/stage1-codex-qwen35-2b-012-first-edit-precision-v1"
OLD = ROOT / "runs/evaluation/stage1-codex-qwen35-2b-012-mix006007-56step-train-fit-diagnostic-evalfix-v2"
CASES = ("train-literal-edit-006-01-positional-format", "train-literal-edit-006-05-tuple-prefix",
         "train-fstring-edit-007-01-inline-docstring", "train-fstring-edit-007-05-trim-expression")
VARIANTS = ("float32", "bfloat16")
START = {**prior.safety.START_LIMITS, "physical_free_bytes":14 * 1024**3}
STOP = dict(prior.safety.STOP_LIMITS)
read, save, digest = prior.read, prior.save, prior.digest


def canonical_hash(value):
    return sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def inspect_command(command, original, reference):
    match = evaluator._GUARDED.fullmatch(command.strip()) or evaluator._PIPE.fullmatch(command.strip())
    if match is None:
        return {"grammar_supported":False, "exact_reference":False}
    old, new = (evaluator._literal(match.group(key)) for key in ("old", "new"))
    found = bool(old) and old in original
    result = original.replace(old,new) if found else original
    return {"grammar_supported":True, "old":old, "new":new, "old_found":found,
            "replacement_count":original.count(old) if old else 0,
            "result_source":result, "exact_reference":found and result == reference}


def inspect_completion(text, original, reference):
    try:
        parsed = evaluator.legacy.parse_qwen_tool_completion(text)
        if len(parsed.tool_calls) != 1 or parsed.tool_calls[0].name != "exec_command":
            raise ValueError("Expected one exec_command call")
        command = parsed.tool_calls[0].arguments["cmd"]
        return {"call_format_valid":True, "command":command,
                **inspect_command(command,original,reference)}
    except (ValueError,KeyError,TypeError) as error:
        return {"call_format_valid":False, "error":str(error), "exact_reference":False}


def verify():
    plan = read(RUN / "diagnostic-plan.json")
    for name, expected in plan["input_pins"].items():
        if digest(ROOT / name) != expected:
            raise ValueError("Frozen input changed: " + name)
    if plan["start_limits"] != START or plan["stop_limits"] != STOP or plan["memory_guard_override"]:
        raise ValueError("Resource policy changed")
    return plan


def prepare():
    from capability_capsule.eval.records import TeacherTrajectory
    from capability_capsule.training.sft import SFTChatContract, _chat_messages
    if RUN.exists():
        raise FileExistsError("Never overwrite/resume a diagnostic")
    prior.verify_inputs()
    contract = read(prior.SFT / "manifest.json")["chat_contract"]
    trajectories = [TeacherTrajectory.model_validate_json(line) for line in
                    (prior.PUBLICATION / "train.jsonl").read_text("utf-8").splitlines()]
    frozen = []
    paths = set(prior.PINS)
    checkpoint = prior.VARIANTS["012"]
    metadata = read(checkpoint)
    paths.update(checkpoint.parent / metadata["adapter_directory"] / name
                 for name in ("adapter_config.json", "adapter_model.safetensors"))
    paths.update((prior.PUBLICATION / "train.jsonl", prior.SFT / "train.jsonl", Path(__file__),
                  Path(prior.__file__), Path(prior.inventory.__file__), Path(prior.safety.__file__),
                  Path(evaluator.__file__), Path(evaluator.legacy.__file__)))
    for task in CASES:
        trajectory = next(t for t in trajectories if t.task_id == task)
        messages = _chat_messages(trajectory, chat_contract=SFTChatContract.model_validate(contract))
        context = messages[:4]
        original = json.loads(context[-1]["content"])["output"]
        report_path = OLD / "case-reports" / (task + ".json")
        report = read(report_path)
        first = report["turns"][0]
        # Require the exact assistant read and tool output, not just the same initial prompt.
        old_read = evaluator.legacy.parse_qwen_tool_completion(first["completion"])
        expected_call = context[2]["tool_calls"][0]["function"]
        expected_args = expected_call["arguments"]
        if isinstance(expected_args,str):
            expected_args = json.loads(expected_args)
        if (old_read.content != context[2]["content"] or len(old_read.tool_calls) != 1
                or old_read.tool_calls[0].name != expected_call["name"]
                or old_read.tool_calls[0].arguments != expected_args):
            raise ValueError("First read assistant history differs")
        if first["execution"]["output"] != original:
            raise ValueError("First read output differs")
        reference_path = OLD / "preflight/reference" / task / "greeting.py"
        reference = reference_path.read_text("utf-8")
        gold = messages[4]["tool_calls"][0]["function"]["arguments"]
        if isinstance(gold,str):
            gold = json.loads(gold)
        if not inspect_command(gold["cmd"],original,reference)["exact_reference"]:
            raise ValueError("Selected reference needs more than one edit")
        paths.update((report_path,reference_path))
        frozen.append({"task_id":task,"messages":context,"context_sha256":canonical_hash(context),
                       "original_source":original,"reference_source":reference})
    prior.RUN = RUN
    protected = prior.protected_hashes()
    RUN.mkdir(parents=True)
    save(RUN / "protected-input-hashes.json",protected)
    save(RUN / "planning-before.json",{name:digest(ROOT/name) for name in prior.inventory.DOCS})
    save(RUN / "diagnostic-plan.json",{"cases":frozen,"contract":contract,
        "input_pins":{p.relative_to(ROOT).as_posix():digest(p) for p in paths},
        "variant_order":list(VARIANTS),"max_new_tokens":256,"do_sample":False,"use_cache":True,
        "seed":42,"torch_threads":4,"device":"cpu","adapter_autocast_dtype":True,
        "start_limits":START,"stop_limits":STOP,"memory_guard_override":False,
        "command_execution":False,"training":False,"automatic_retry":False,
        "interpretation":"Failure-selected seen cases, in-memory single-edit match only, not task or validation success"})
    print("Prepared four cases, eight one-turn generations; guards restored",flush=True)


def worker(variant):
    plan = verify()
    snapshot = prior.safety.resources()
    save(RUN / variant / "worker-resource-check.json",snapshot)
    if prior.safety.below_limits(snapshot,START):
        raise RuntimeError("Worker startup resource gate blocked")
    import torch
    import transformers
    import peft
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    torch.set_num_threads(4)
    torch.manual_seed(42)
    tokenizer = AutoTokenizer.from_pretrained(prior.MODEL_ID,revision=prior.REVISION,local_files_only=True)
    prompts = []
    for case in plan["cases"]:
        encoded = tokenizer.apply_chat_template(case["messages"], tools=plan["contract"]["tools"],
            tokenize=True,add_generation_prompt=True,enable_thinking=False,return_dict=True,return_tensors="pt")
        prompts.append(encoded)
        save(RUN / variant / "prompts" / (case["task_id"]+".json"),{
            "input_ids":encoded["input_ids"][0].tolist(),"context_sha256":case["context_sha256"]})
    base = AutoModelForCausalLM.from_pretrained(prior.MODEL_ID,revision=prior.REVISION,
        dtype=getattr(torch,variant),local_files_only=True)
    checkpoint = prior.VARIANTS["012"]
    model = PeftModel.from_pretrained(base,checkpoint.parent / read(checkpoint)["adapter_directory"],
                                    autocast_adapter_dtype=True,is_trainable=False)
    model.eval()
    counts = Counter()
    for parameter in model.parameters():
        counts[f"{parameter.device}/{parameter.dtype}"] += parameter.numel()
    save(RUN / variant / "model-load.json",{"requested_base_dtype":variant,"parameter_numel":dict(counts),
        "active_adapters":list(model.active_adapters),"training":model.training,
        "torch":torch.__version__,"transformers":transformers.__version__,"peft":peft.__version__,
        "model_config":model.config.to_dict(),"use_cache":True,"torch_threads":torch.get_num_threads()})
    for case, encoded in zip(plan["cases"],prompts,strict=True):
        with torch.inference_mode():
            result = model.generate(**encoded,max_new_tokens=256,do_sample=False,use_cache=True,
                eos_token_id=tokenizer.eos_token_id,pad_token_id=tokenizer.pad_token_id)
        ids = result[0,encoded["input_ids"].shape[-1]:].tolist()
        text = tokenizer.decode(ids,skip_special_tokens=False)
        save(RUN / variant / "raw" / (case["task_id"]+".json"),{"completion":text,"token_ids":ids,
            "finish_reason":evaluator.generation_finish_reason(ids,tokenizer.eos_token_id,256)})
        save(RUN / variant / "cases" / (case["task_id"]+".json"),
             inspect_completion(text,case["original_source"],case["reference_source"]))
        print(variant + " " + case["task_id"] + " recorded",flush=True)


def evaluate(variant):
    verify()
    directory = RUN / variant
    directory.mkdir()  # A blocked or partial attempt is never overwritten or retried.
    try:
        snapshot = prior.safety.resources()
        blocked = prior.safety.below_limits(snapshot,START)
    except Exception as error:
        save(directory / "supervisor-result.json",{"status":"probe_failed","error":str(error)})
        return
    if blocked:
        save(directory / "supervisor-result.json",{"status":"startup_blocked","fields":blocked,"snapshot":snapshot})
        print(variant+" blocked: "+json.dumps(snapshot),flush=True)
        return
    save(directory / "supervisor-attempt.json",snapshot)
    with (directory / "process.log").open("x",encoding="utf-8") as logfile:
        process = subprocess.Popen([sys.executable,__file__,"--worker",variant],start_new_session=True,
            stdout=logfile,stderr=subprocess.STDOUT)
        stopped = None
        try:
            while process.poll() is None:
                try:
                    snapshot = prior.safety.resources()
                    prior.inventory.append(directory / "resource-monitor.jsonl",snapshot)
                    low = prior.safety.below_limits(snapshot,STOP)
                    if low:
                        stopped = {"fields":low,"snapshot":snapshot}
                except Exception as error:
                    stopped = {"probe_error":str(error)}
                if stopped:
                    break
                time.sleep(3)
        finally:
            if process.poll() is None:
                os.killpg(process.pid,signal.SIGTERM)
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid,signal.SIGKILL)
            code = process.wait()
    save(directory / "supervisor-result.json",{"status":"completed" if code == 0 and not stopped else "interrupted",
        "exit_code":code,"resource_abort":stopped,"automatic_retry":False})


def finish():
    plan = verify()
    before = read(RUN / "protected-input-hashes.json")
    changed = [name for name,value in before.items() if not (ROOT/name).is_file() or digest(ROOT/name) != value]
    if changed:
        raise ValueError(changed)
    results = {}
    for variant in VARIANTS:
        status = read(RUN / variant / "supervisor-result.json")
        rows = {case["task_id"]:read(path) for case in plan["cases"]
                if (path := RUN/variant/"cases"/(case["task_id"]+".json")).exists()}
        results[variant] = {"supervisor":status,"recorded_cases":len(rows),"cases":rows,
                            "exact_reference_count":sum(row["exact_reference"] for row in rows.values())}
    for task in CASES:
        a,b = (RUN/v/"prompts"/(task+".json") for v in VARIANTS)
        if a.exists() and b.exists() and read(a) != read(b):
            raise ValueError("Precision conditions received different tokenized prompts")
    save(RUN/"comparison.json",{"variants":results,"full_comparison_completed":all(
        r["recorded_cases"] == 4 and r["supervisor"]["status"] == "completed" for r in results.values()),
        "not_task_success_rate":True,"executed_generated_commands":0,"training_started":False})
    save(RUN/"input-preservation.json",{"unchanged":True,"protected_file_count":len(before),"changed_paths":[]})
    print(json.dumps({v:{k:r[k] for k in ("recorded_cases","exact_reference_count","supervisor")} for v,r in results.items()}),flush=True)


def seal():
    save(RUN/"planning-update.json",{"explicitly_authorized":True,
        "before":read(RUN/"planning-before.json"),"after":{name:digest(ROOT/name) for name in prior.inventory.DOCS},
        "reason":"Authorized precision-controlled single-turn diagnostic; default memory and disk guards retained"})


if __name__ == "__main__":
    args = sys.argv[1:]
    if args == ["--prepare"]:
        prepare()
    elif len(args) == 2 and args[0] in ("--worker","--evaluate") and args[1] in VARIANTS:
        (worker if args[0] == "--worker" else evaluate)(args[1])
    elif args == ["--finish"]:
        finish()
    elif args == ["--seal"]:
        seal()
    else:
        raise ValueError("Use --prepare, --evaluate float32/bfloat16, --finish, --seal")
