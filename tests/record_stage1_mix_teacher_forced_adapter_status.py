"""Preserve worker-side startup rejection without retrying or inventing scores."""

import ast

import continue_stage1_mix_teacher_forced_adapters as current


def parse_headroom(log):
    prefix = "RuntimeError: Insufficient headroom before load: "
    lines = [line[len(prefix):] for line in log.splitlines() if line.startswith(prefix)]
    if len(lines) != 1:
        raise ValueError("Expected one exact startup rejection")
    snapshot = ast.literal_eval(lines[0])
    if not isinstance(snapshot,dict) or not all(isinstance(snapshot.get(k),int) for k in
            ("physical_free_bytes","commit_free_bytes","c_free_bytes","wsl_available_bytes")):
        raise ValueError("Malformed resource snapshot")
    return snapshot


def record():
    current.configure()
    p = current.p
    for variant in current.ADAPTERS:
        directory = current.RUN / variant
        if not directory.exists():
            continue
        end = p.read(directory / "supervisor-result.json")
        if end["exit_code"] == 1 and end["resource_abort"] is None:
            if any((directory / name).exists() for name in ("forward-attempt.json","model-load.json","summary.json","cases")):
                raise ValueError("Do not classify a model/scored attempt as worker pre-load rejection")
            snapshot = parse_headroom((directory / "process.log").read_text("utf-8"))
            limits = p.read(current.RUN / "diagnostic-plan.json")["start_limits"]
            low = p.safety.below_limits(snapshot,limits)
            if not low:
                raise ValueError("Startup rejection snapshot passes its limits")
            p.save(directory / "worker-startup-blocked.json",{"resources":snapshot,"limits":limits,
                "failed_fields":low,"model_loaded":False,"recorded_case_count":0,
                "evidence":"Exact recorded RuntimeError plus absence of forward-attempt/model-load/score files"})
    current.seal_status()


if __name__ == "__main__":
    record()
