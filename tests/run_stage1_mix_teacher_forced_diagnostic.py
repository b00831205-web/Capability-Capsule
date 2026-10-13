"""Authorized forward-only comparison on immutable gold SFT; never generate/execute."""

import json
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import threading
import time

import run_stage1_mix011_train_fit_diagnostic as inventory
import recover_stage1_mix011_train_fit_diagnostic as safety

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "runs/evaluation/stage1-codex-mix006007-teacher-forced-base-011-012-v1"
SFT = inventory.preparation.SFT
PUBLICATION = inventory.preparation.PUBLICATION
MODEL_ID = "Qwen/Qwen3.5-2B"
REVISION = "15852e8c16360a2fea060d615a32b45270f8a8fc"
VARIANTS = {
    "base": None,
    "011": ROOT / "runs/training/stage1-codex-qwen35-2b-011-mix006007-28step/checkpoint.json",
    "012": ROOT / "runs/training/stage1-codex-qwen35-2b-012-mix006007-56step/checkpoint.json",
}
PINS = {
    SFT / "manifest.json": "5d8747cbf4ea5d1f023ecfd12694ec7b839af49701b7b59b6c9c06e27ef823cc",
    PUBLICATION / "manifest.json": "2739762194576d51c89900dc3f7aceade571e1b4ed4a7359f6134198432faa60",
    VARIANTS["011"]: "30b10ef7364c6b1ea7cb94a72d2d064b84318a5969d078a2b4e0598ed44bba52",
    VARIANTS["012"]: "60bf2bd64456b1264cc16dfe425cf7cd9d8ae781853eb6cff83587ebe72f2921",
}
CATEGORIES = ("all_supervised", "prompt_prefix", "post_eos", "assistant_body", "tool_command", "edit_command",
    "source_literal", "replacement_literal", "literal_punctuation", "dynamic_placeholder", "tool_closing", "eos", "final_response")
ARGS = re.compile(r"\.Replace\(\s*(?P<old>'(?:[^']|'')*'|\"[^\"$`]*\")\s*,\s*(?P<new>'(?:[^']|'')*'|\"[^\"$`]*\")\s*\)")
read, digest, save = inventory.read, inventory.digest, inventory.save


def verify_inputs():
    for path, value in PINS.items():
        if digest(path) != value:
            raise ValueError(f"Pinned input changed: {path.name}")
    for directory in (SFT, PUBLICATION):
        inventory.preparation.verify_bundle(directory)
    for path in VARIANTS.values():
        if path is None:
            continue
        checkpoint = read(path)
        if (checkpoint["base_model_id"], checkpoint["base_model_revision"]) != (MODEL_ID, REVISION):
            raise ValueError("Checkpoint base mismatch")
        for filename,key in (("adapter_config.json","adapter_config_sha256"), ("adapter_model.safetensors","adapter_model_sha256")):
            if digest(path.parent / checkpoint["adapter_directory"] / filename) != checkpoint[key]:
                raise ValueError("Adapter bytes changed")


def protected_hashes():
    previous = inventory.RUN
    inventory.RUN = RUN
    try:
        return inventory.protected_hashes()
    finally:
        inventory.RUN = previous


def verify_preservation():
    before = read(RUN / "protected-input-hashes.json")
    changed = [p for p,value in before.items() if not (ROOT / p).is_file() or digest(ROOT / p) != value]
    if changed:
        raise ValueError(f"Protected inputs changed: {changed}")
    return len(before)


def prediction_positions(labels):
    if labels[0] != -100:
        raise ValueError("Position zero cannot have a causal target")
    return [i for i,value in enumerate(labels) if value != -100]


def overlaps(offset, interval):
    return offset[1] > interval[0] and offset[0] < interval[1]


def mark_intervals(text):
    """Character ranges only; no tool parsing/execution or fixture writes."""
    intervals = {key:[] for key in CATEGORIES}
    for command in re.finditer(r"<parameter=cmd>\n?(.*?)\n?</parameter>", text, re.S):
        intervals["tool_command"].append(command.span(1))
        edit = ARGS.search(command.group(1))
        if edit is None:
            continue
        intervals["edit_command"].append(command.span(1))
        for group,category in (("old","source_literal"),("new","replacement_literal")):
            a,b = edit.span(group)
            a,b = a+command.start(1)+1,b+command.start(1)-1
            intervals[category].append((a,b))
            for match in re.finditer(r"[.,!:;/?]", text[a:b]):
                intervals["literal_punctuation"].append((a+match.start(),a+match.end()))
            for match in re.finditer(r"\{[^{}]*\}", text[a:b]):
                intervals["dynamic_placeholder"].append((a+match.start(),a+match.end()))
    intervals["tool_closing"] = [m.span() for m in re.finditer(r"</(?:parameter|function|tool_call)>",text)]
    return intervals


def metadata(example, trajectory, tokenizer, contract):
    from capability_capsule.eval.records import TeacherTrajectory
    from capability_capsule.training.sft import SFTChatContract, _chat_messages
    messages = _chat_messages(TeacherTrajectory.model_validate(trajectory), chat_contract=SFTChatContract.model_validate(contract))
    def encode(items, generation=False):
        return list(tokenizer.apply_chat_template(items, tools=contract["tools"], tokenize=True,
            add_generation_prompt=generation, enable_thinking=False, return_dict=True)["input_ids"])
    ids,labels = example["input_ids"],example["labels"]
    if encode(messages) != ids:
        raise ValueError("Stored SFT differs from full untruncated template")
    positions = prediction_positions(labels)
    headers, bodies, tails, finals = set(),set(),set(),set()
    prior = 0
    for end,message in enumerate(messages,1):
        if not any(m["role"] == "user" for m in messages[:end]):
            continue
        prefix = encode(messages[:end])
        if ids[:len(prefix)] != prefix:
            raise ValueError("Conversation prefix mismatch")
        if message["role"] == "assistant":
            prompt = encode(messages[:end-1],True)
            if ids[:len(prompt)] != prompt:
                raise ValueError("Inference prompt mismatch")
            last_eos = max(i for i in range(len(prompt),len(prefix)) if ids[i] == tokenizer.eos_token_id)
            headers.update(range(prior,len(prompt)))
            bodies.update(range(len(prompt),last_eos+1))
            tails.update(range(last_eos+1,len(prefix)))
            if not message.get("tool_calls"):
                finals.update(range(len(prompt),last_eos+1))
        elif any(v != -100 for v in labels[prior:len(prefix)]):
            raise ValueError("Non-assistant supervision")
        prior = len(prefix)
    if set(positions) != headers | bodies | tails or headers & bodies or bodies & tails:
        raise ValueError("Assistant supervision spans do not match stored labels")
    text = tokenizer.decode(ids,skip_special_tokens=False)
    encoded = tokenizer(text,add_special_tokens=False,return_offsets_mapping=True)
    if encoded["input_ids"] != ids:
        raise ValueError("Decoded text/offsets do not round trip")
    intervals = mark_intervals(text)
    result = []
    for position in positions:
        if labels[position] != ids[position]:
            raise ValueError("Label differs from input target")
        groups = ["all_supervised", "prompt_prefix" if position in headers else "post_eos" if position in tails else "assistant_body"]
        if position in finals:
            groups.append("final_response")
        if ids[position] == tokenizer.eos_token_id:
            groups.append("eos")
        if position in bodies:
            groups.extend(k for k,ranges in intervals.items() if any(overlaps(encoded["offset_mapping"][position],r) for r in ranges))
        result.append({"position":position,"target_id":ids[position],"target_text":tokenizer.decode([ids[position]],skip_special_tokens=False),"categories":groups})
    if not any("edit_command" in row["categories"] for row in result):
        raise ValueError("No edit command identified")
    return result


def score_row(logits, target):
    """One float32 vocabulary row; correct one-token causal shift is applied by caller."""
    import torch
    row = logits.float()
    if not torch.isfinite(row).all():
        raise ValueError("Non-finite logits")
    nll = float((torch.logsumexp(row,dim=-1)-row[target]).item())
    best = int(row.argmax().item())
    return {"nll":nll,"target_probability":math.exp(-nll),"rank":int((row > row[target]).sum().item())+1,
        "top1_correct":best == target,"top5_contains_target":bool((row.topk(min(5,row.numel())).indices == target).any().item()),
        "predicted_id":best}


def aggregate(records):
    output = {}
    for category in CATEGORIES:
        selected = [row for row in records if category in row["categories"]]
        if not selected:
            output[category] = {"token_count":0}
            continue
        nll = sum(row["nll"] for row in selected)
        output[category] = {"token_count":len(selected),"nll_sum":nll,"mean_nll":nll/len(selected),
            "mean_target_probability":sum(row["target_probability"] for row in selected)/len(selected),
            "top1_accuracy":sum(row["top1_correct"] for row in selected)/len(selected),
            "top5_accuracy":sum(row["top5_contains_target"] for row in selected)/len(selected)}
    return output


def forward_scores(model, example, meta):
    import torch
    ids = torch.tensor([example["input_ids"]],dtype=torch.long)
    mask = torch.tensor([example["attention_mask"]],dtype=torch.long)
    positions = torch.tensor([row["position"]-1 for row in meta],dtype=torch.long)
    with torch.inference_mode():
        output = model(input_ids=ids,attention_mask=mask,use_cache=False,logits_to_keep=positions,return_dict=True)
        if output.logits.shape[:2] != (1,len(meta)):
            raise ValueError("Model did not honor sparse prediction positions")
        return [{**row,**score_row(output.logits[0,n],row["target_id"])} for n,row in enumerate(meta)]


def prepare():
    if RUN.exists():
        raise FileExistsError("Refusing overwrite/resume")
    verify_inputs()
    protected = protected_hashes()
    RUN.mkdir(parents=True)
    save(RUN / "protected-input-hashes.json",protected)
    save(RUN / "planning-before.json",{name:digest(ROOT / name) for name in inventory.DOCS})
    save(RUN / "diagnostic-plan.json",{
        "scope":"Fourteen seen gold train trajectories; forward-only, not task success or unseen validation",
        "model_id":MODEL_ID,"revision":REVISION,"variant_order":list(VARIANTS),
        "case_count":14,"supervised_token_count":3093,"dtype":"float32","device":"cpu","torch_threads":4,
        "batch_size":1,"use_cache":False,"training":False,"generation":False,"command_execution":False,
        "automatic_retry":False,"input_pins":{p.relative_to(ROOT).as_posix():v for p,v in PINS.items()},
        "runner_sha256":digest(Path(__file__)),"start_limits":safety.START_LIMITS,"stop_limits":safety.STOP_LIMITS,
        "memory_guard_override":False,"disk_guard_retained":True,"categories":CATEGORIES,
        "expected_category_counts":{"all_supervised":3093,"prompt_prefix":413,"assistant_body":2621,
            "tool_command":894,"tool_closing":315,"eos":59,"post_eos":59,"edit_command":724,
            "source_literal":68,"literal_punctuation":34,"dynamic_placeholder":61,"replacement_literal":74,"final_response":159},
        "causal_alignment":"Logits at position p-1 predict gold label at p; all earlier tokens remain gold",
        "category_policy":"Character overlap; categories may overlap; prompt-prefix and post-EOS template targets excluded from assistant_body",
        "rank_policy":"1 plus count strictly greater than target; top1 is argmax; top5 uses actual topk indices",
        "aggregation":"Token-weighted category NLL/accuracy; also per-case metrics; no statistical generalization claim"})
    print(f"Prepared forward-only comparison; {len(protected)} old files protected",flush=True)


def worker(variant):
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    verify_inputs()
    verify_preservation()
    plan = read(RUN / "diagnostic-plan.json")
    if digest(Path(__file__)) != plan["runner_sha256"]:
        raise ValueError("Runner changed since freeze")
    target = RUN / variant
    if (target / "forward-attempt.json").exists():
        raise FileExistsError("No repeat forward pass")
    snapshot = safety.resources()
    if safety.below_limits(snapshot,safety.START_LIMITS):
        raise RuntimeError(f"Insufficient headroom before load: {snapshot}")
    save(target / "forward-attempt.json",{"started_at":safety.timestamp(),"resources":snapshot,"one_shot":True})
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    torch.set_num_threads(plan["torch_threads"])
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID,revision=REVISION,local_files_only=True)
    examples = inventory.preparation.records(SFT / "train.jsonl")
    trajectories = inventory.preparation.records(PUBLICATION / "train.jsonl")
    if len(examples) != 14 or len(trajectories) != 14:
        raise ValueError("Expected fourteen train examples")
    contract = read(SFT / "manifest.json")["chat_contract"]
    metas = []
    for example,trajectory in zip(examples,trajectories,strict=True):
        if trajectory["split"] != "train" or trajectory["trajectory_id"] != example["source_trajectory_id"]:
            raise ValueError("Original train identity mismatch")
        metas.append(metadata(example,trajectory,tokenizer,contract))
    if sum(map(len,metas)) != 3093:
        raise ValueError("Unexpected supervision count")
    from collections import Counter
    counts = Counter(c for meta in metas for token in meta for c in token["categories"])
    if dict(counts) != plan["expected_category_counts"]:
        raise ValueError("Category coverage changed since preflight")
    print(f"Loading {variant}; metadata and causal targets verified",flush=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID,revision=REVISION,local_files_only=True,dtype=torch.float32)
    if VARIANTS[variant] is not None:
        checkpoint = read(VARIANTS[variant])
        model = PeftModel.from_pretrained(model,VARIANTS[variant].parent / checkpoint["adapter_directory"],is_trainable=False)
    model.eval()
    devices = sorted({str(p.device) for p in model.parameters()})
    dtypes = sorted({str(p.dtype) for p in model.parameters() if p.is_floating_point()})
    if devices != ["cpu"] or dtypes != ["torch.float32"] or model.training:
        raise ValueError("Unexpected model execution configuration")
    save(target / "model-load.json",{"devices":devices,"floating_dtypes":dtypes,"training_mode":model.training,
        "active_adapters":list(model.active_adapters) if variant != "base" else [],
        "checkpoint_sha256":digest(VARIANTS[variant]) if VARIANTS[variant] is not None else None})
    all_records = []
    for index,(example,meta) in enumerate(zip(examples,metas,strict=True),1):
        snapshot = safety.resources()
        if safety.below_limits(snapshot,safety.STOP_LIMITS):
            raise RuntimeError(f"Resource gate before forward: {snapshot}")
        scored = forward_scores(model,example,meta)
        for row in scored:
            row["predicted_text"] = tokenizer.decode([row["predicted_id"]],skip_special_tokens=False)
        save(target / "cases" / (example["source_trajectory_id"] + ".json"),{
            "trajectory_id":example["source_trajectory_id"],"split":"train","seen_during_training":variant != "base",
            "input_token_count":len(example["input_ids"]),"metrics":aggregate(scored),"tokens":scored})
        all_records.extend(scored)
        print(f"{variant} {index}/14: body NLL={aggregate(scored)['assistant_body']['mean_nll']:.6f}",flush=True)
    save(target / "summary.json",{"variant":variant,"case_count":14,"metrics":aggregate(all_records),
        "forward_only":True,"generation_calls":0,"executed_commands":0,"completed_at":safety.timestamp()})
    del model
    verify_preservation()


def run_variant(variant):
    target = RUN / variant
    if target.exists():
        raise FileExistsError("Refusing overwrite/retry of variant")
    snapshot = safety.resources()
    for _ in range(6):
        if not safety.below_limits(snapshot,safety.START_LIMITS):
            break
        print(f"Waiting for default resource headroom before {variant}: {snapshot}",flush=True)
        time.sleep(5)
        snapshot = safety.resources()
    if safety.below_limits(snapshot,safety.START_LIMITS):
        save(RUN / (variant + "-startup-blocked.json"),{"resources":snapshot,"limits":safety.START_LIMITS})
        raise RuntimeError(f"Startup gates not met for {variant}: {snapshot}")
    save(target / "launch.json",{"started_at":safety.timestamp(),"resources":snapshot,"one_shot":True})
    stopped = None
    with (target / "process.log").open("x",encoding="utf-8") as logfile:
        process = subprocess.Popen([sys.executable,__file__,"--worker",variant],start_new_session=True,
            stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
        def forward():
            for line in process.stdout:
                logfile.write(line)
                logfile.flush()
                print(line,end="",flush=True)
        reader = threading.Thread(target=forward,daemon=True)
        reader.start()
        while process.poll() is None:
            try:
                snapshot = safety.resources()
                inventory.append(target / "resources.jsonl",snapshot)
                low = safety.below_limits(snapshot,safety.STOP_LIMITS)
                if low:
                    stopped = {"low_fields":low,"resources":snapshot}
            except Exception as exc:
                stopped = {"probe_error":str(exc)}
            if stopped:
                try:
                    os.killpg(process.pid,signal.SIGTERM)
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid,signal.SIGKILL)
                except ProcessLookupError:
                    pass
                break
            time.sleep(3)
        code = process.wait()
        reader.join(timeout=10)
    save(target / "supervisor-result.json",{"exit_code":code,"resource_abort":stopped,"restarted":False})
    if code or stopped:
        raise RuntimeError(f"{variant} interrupted: exit={code}; {stopped}")


def run():
    for variant in VARIANTS:
        run_variant(variant)
    summaries = {variant:read(RUN / variant / "summary.json") for variant in VARIANTS}
    save(RUN / "comparison.json",{"variants":summaries,
        "interpretation":"Gold-history next-token prediction on seen data; neither free-running success nor unseen generalization"})
    save(RUN / "input-preservation.json",{"unchanged":True,"protected_file_count":verify_preservation(),"changed_paths":[]})


def finalize():
    comparison = read(RUN / "comparison.json")
    if len(comparison["variants"]) != 3:
        raise ValueError("Incomplete comparison")
    for variant in VARIANTS:
        end = read(RUN / variant / "supervisor-result.json")
        if end["exit_code"] or end["resource_abort"] is not None:
            raise ValueError("Incomplete variant")
    save(RUN / "planning-update.json",{"explicitly_authorized":True,
        "before":read(RUN / "planning-before.json"),"after":{name:digest(ROOT / name) for name in inventory.DOCS},
        "reason":"Authorized forward-only base/011/012 gold-history comparison; no generation or training"})
    save(RUN / "completion.json",{"case_count_per_variant":14,"variant_count":3,
        "protected_file_count":verify_preservation(),"training_started":False,"generation_calls":0,"executed_commands":0})


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--worker" and sys.argv[2] in VARIANTS:
        worker(sys.argv[2])
    elif sys.argv[1:] == ["--prepare-only"]:
        prepare()
    elif sys.argv[1:] == ["--run"]:
        run()
    elif sys.argv[1:] == ["--finalize"]:
        finalize()
    else:
        raise ValueError("Use --prepare-only, --run, --finalize or internal --worker VARIANT")
