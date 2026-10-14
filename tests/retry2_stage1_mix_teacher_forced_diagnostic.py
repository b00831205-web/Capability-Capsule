"""Second separately requested attempt; unchanged 14 GiB/default runtime protection."""

from pathlib import Path
import sys

import retry_stage1_mix_teacher_forced_diagnostic as previous

forward = previous.forward
ROOT = forward.ROOT
PRIOR = previous.RUN
RUN = ROOT / "runs/evaluation/stage1-codex-mix006007-teacher-forced-base-011-012-v1-retry2-20261013"
DEPENDENCY = Path(previous.__file__)
DEPENDENCY_SHA256 = "a893f47705f6c6a5fed8f7c850353314db1e33e92334925aa227fb0fd83715dc"
PREVIOUS_CONFIGURE = previous.configure


def configure():
    if forward.digest(DEPENDENCY) != DEPENDENCY_SHA256:
        raise ValueError("Frozen retry implementation changed")
    previous.RUN = RUN
    previous.__file__ = __file__
    previous.configure = configure
    PREVIOUS_CONFIGURE()
    forward.PINS = {**forward.PINS,DEPENDENCY:DEPENDENCY_SHA256}


def prepare():
    configure()
    previous.prepare()
    forward.save(RUN / "renewed-request.json",{
        "basis":"User explicitly requested another attempt",
        "prior_attempt":PRIOR.relative_to(ROOT).as_posix(),
        "prior_block_sha256":forward.digest(PRIOR / "base-startup-blocked.json"),
        "protocol_change":None,"precision":"float32","startup_ram_gib":14,
        "automatic_retry":False,"generation":False,"command_execution":False,"training":False})


if __name__ == "__main__":
    if sys.argv[1:] == ["--prepare-only"]:
        prepare()
    elif sys.argv[1:] == ["--run"]:
        configure()
        forward.run()
    elif sys.argv[1:] == ["--record-status"]:
        configure()
        previous.record_status()
    elif len(sys.argv) == 3 and sys.argv[1] == "--worker" and sys.argv[2] in forward.VARIANTS:
        configure()
        forward.worker(sys.argv[2])
    else:
        raise ValueError("Use --prepare-only, --run, --record-status or internal --worker VARIANT")
