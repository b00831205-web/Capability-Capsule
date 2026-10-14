"""Derive error detail from saved logits scores only; never load or generate with a model."""

import run_stage1_mix_teacher_forced_adapters_memory_override as current

p = current.p
CRITICAL = {"source_literal","replacement_literal"}


def summarize():
    current.configure()
    comparison = p.read(current.RUN / "comparison.json")
    examples = p.inventory.preparation.records(p.SFT / "train.jsonl")
    profiles,aligned = {},{}
    for variant,directory in (("base",current.previous.BASE / "base"),("011",current.RUN / "011"),("012",current.RUN / "012")):
        cases,errors,all_rows = [],[],[]
        for example in examples:
            report = p.read(directory / "cases" / (example["source_trajectory_id"]+".json"))
            rows = report["tokens"]
            missed = [r for r in rows if CRITICAL.intersection(r["categories"]) and not r["top1_correct"]]
            cases.append({"trajectory_id":report["trajectory_id"],"body_mean_nll":report["metrics"]["assistant_body"]["mean_nll"],
                "literal_argmax_mismatches":len(missed)})
            errors.extend({"trajectory_id":report["trajectory_id"],**row} for row in missed)
            all_rows.extend({"trajectory_id":report["trajectory_id"],**row} for row in rows)
        if p.aggregate(all_rows) != comparison["variants"][variant]["metrics"]:
            raise ValueError("Analysis inputs differ from verified comparison")
        profiles[variant] = {"case_count":14,"literal_argmax_mismatches":len(errors),
            "cases_with_literal_argmax_mismatch":sum(c["literal_argmax_mismatches"]>0 for c in cases),
            "cases":cases,"literal_mismatch_tokens":errors}
        aligned[variant] = all_rows
    transitions = {"011_wrong_012_correct":0,"011_correct_012_wrong":0,"both_wrong":0,"both_correct":0}
    for a,b in zip(aligned["011"],aligned["012"],strict=True):
        if (a["trajectory_id"],a["position"],a["target_id"],a["categories"]) != (b["trajectory_id"],b["position"],b["target_id"],b["categories"]):
            raise ValueError("Unaligned pair")
        if not CRITICAL.intersection(a["categories"]):
            continue
        key = ("both_correct" if a["top1_correct"] and b["top1_correct"] else "both_wrong" if not a["top1_correct"] and not b["top1_correct"]
            else "011_wrong_012_correct" if b["top1_correct"] else "011_correct_012_wrong")
        transitions[key] += 1
    resources = {}
    for variant in ("011","012"):
        samples = p.inventory.preparation.records(current.RUN / variant / "resources.jsonl")
        resources[variant] = {"sample_count":len(samples),"minimum_observed_bytes":{
            key:min(s[key] for s in samples) for key in ("physical_free_bytes","commit_free_bytes","wsl_available_bytes","c_free_bytes")},
            "limit":"Periodic samples may miss transient peaks; observed minima are not absolute peaks"}
    p.save(current.RUN / "critical-token-analysis.json",{
        "source":"Verified saved per-token reports; no new inference",
        "profiles":profiles,"011_to_012_literal_transitions":transitions,"resource_observations":resources,
        "limitations":["Gold-token argmax mismatch is not task failure or proof that every alternative command is invalid",
            "Categories overlap; punctuation tokens can include placeholders and expression dots",
            "Seen train data only; neither autonomous success nor unseen generalization",
            "Budget comparison also changes the LR schedule horizon; not an isolated causal test"]})


if __name__ == "__main__":
    summarize()
