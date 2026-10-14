"""Seal an unscored resource interruption; never load a model or repeat a forward pass."""

import run_stage1_mix_teacher_forced_diagnostic as p


def record():
    end = p.read(p.RUN / "base/supervisor-result.json")
    abort = end["resource_abort"]
    if end["exit_code"] != -15 or abort is None or abort["low_fields"] != ["physical_free_bytes"]:
        raise ValueError("Expected recorded memory-guard termination")
    if end["restarted"] or any((p.RUN / name).exists() for name in ("011","012","comparison.json","completion.json")):
        raise ValueError("Attempt scope differs from unscored base interruption")
    if (p.RUN / "base/summary.json").exists() or list((p.RUN / "base/cases").glob("*.json")):
        raise ValueError("Scored cases exist; do not record zero complete cases")
    count = p.verify_preservation()
    p.save(p.RUN / "interruption.json",{
        "reason":"Windows physical-memory runtime protection during float32 base load/first-forward phase",
        "model_load_completed":True,"recorded_complete_case_count":0,"recorded_token_score_count":0,
        "011_started":False,"012_started":False,"automatic_retry_performed":False,
        "case_success_count":None,"diagnostic_comparison_completed":False,
        "generation_calls":0,"executed_commands":0,"new_training_started":False,
        "worker_exit_code":end["exit_code"],"resource_abort":abort,
        "protected_file_count":count,
        "interpretation":"Unscored infrastructure interruption, not 0/14 or evidence about learning"})
    p.save(p.RUN / "input-preservation.json",{"unchanged":True,"protected_file_count":count,"changed_paths":[]})
    p.save(p.RUN / "planning-update.json",{"explicitly_authorized":True,
        "before":p.read(p.RUN / "planning-before.json"),
        "after":{name:p.digest(p.ROOT / name) for name in p.inventory.DOCS},
        "reason":"Record authorized forward-only protocol and actual memory-guard interruption; no scores or retry"})
    print(f"Sealed unscored interruption; {count} protected files unchanged",flush=True)


if __name__ == "__main__":
    record()
