"""Verify immutable run inputs while accounting for explicitly recorded later doc updates."""

import json
from hashlib import sha256
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
UPDATE = ROOT / "runs/evaluation/stage1-codex-qwen35-2b-011-mix006007-28step-train-fit-diagnostic-v1/planning-update.json"
DOCS = frozenset({"MVP_ROADMAP.md", "PENDING_CONFIGURATION_OPTIMIZATIONS.md"})


def input_matches(path, pinned, current, *, update_path=UPDATE):
    if current == pinned:
        return True
    if path not in DOCS or not update_path.exists():
        return False
    cursor = pinned
    # Recovery is an additive receipt; never rewrite the original authorization.
    updates = (update_path, update_path.parent / "recovery-20261009/planning-update.json",
               ROOT / "tests/mix012-budget-planning-update.json",
               ROOT / "runs/training/stage1-codex-qwen35-2b-012-mix006007-56step/planning-update.json",
               ROOT / "runs/training/stage1-codex-qwen35-2b-012-mix006007-56step/reload-planning-update-20261010.json",
               ROOT / "runs/evaluation/stage1-codex-qwen35-2b-012-mix006007-56step-train-fit-diagnostic-v1/planning-update.json",
               ROOT / "runs/evaluation/stage1-codex-qwen35-2b-012-mix006007-56step-train-fit-diagnostic-v1/recovery-20261011/planning-update.json",
               ROOT / "tests/eval-v2-planning-update.json",
               ROOT / "runs/evaluation/stage1-codex-qwen35-2b-012-mix006007-56step-train-fit-diagnostic-evalfix-v2/planning-update.json")
    for candidate in updates:
        if not candidate.exists():
            continue
        update = json.loads(candidate.read_text("utf-8"))
        if update.get("explicitly_authorized") is True and update.get("before", {}).get(path) == cursor:
            cursor = update.get("after", {}).get(path)
    return cursor == current and sha256((ROOT / path).read_bytes()).hexdigest() == current
