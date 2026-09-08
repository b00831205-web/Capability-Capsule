"""Load and validate retrieval evaluation cases from a JSON file."""

from pathlib import Path

from pydantic import TypeAdapter

from capability_capsule.eval.runner import RetrievalCase

_CASES_ADAPTER = TypeAdapter(tuple[RetrievalCase, ...])


def load_cases(path: Path) -> tuple[RetrievalCase, ...]:
    """Read a nonempty JSON array of evaluation cases."""

    content = path.read_text(encoding="utf-8")
    cases = _CASES_ADAPTER.validate_json(content)

    if not cases:
        raise ValueError("At least one evaluation case is required")

    return cases
