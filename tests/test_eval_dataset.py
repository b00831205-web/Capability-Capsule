import importlib
import json
from pathlib import Path

import pytest

from capability_capsule.eval.runner import RetrievalCase


@pytest.mark.parametrize("indent", [None, 2, 4])
def test_load_cases_preserves_content_and_order(tmp_path: Path, indent: int | None) -> None:
    module = importlib.import_module("capability_capsule.eval.dataset")
    path = tmp_path / "cases.json"
    data = [
        {"question": "恢复码是什么？", "expected_paths": ["docs/recovery.md"]},  # noqa: RUF001
        {"question": "如何配置？", "expected_paths": ["README.md", "docs/config.md"]},  # noqa: RUF001
    ]
    path.write_text(json.dumps(data, ensure_ascii=False, indent=indent), encoding="utf-8")
    before = path.read_bytes()
    cases = module.load_cases(path)
    assert isinstance(cases, tuple)
    assert all(isinstance(case, RetrievalCase) for case in cases)
    assert [case.model_dump(mode="json") for case in cases] == data
    assert path.read_bytes() == before


@pytest.mark.parametrize(
    "content",
    [
        "", "[", "[]", "null", "{}",
        '[{"question": "q", "expected_paths": []}]',
        '[{"question": "  ", "expected_paths": ["a.md"]}]',
        '[{"question": "q", "expected_paths": [" "]}]',
        '[{"question": "q", "expected_paths": ["a.md"], "unknown": true}]',
        '[{"question": "q"}]',
        '[{"question": "q", "expected_paths": ["a.md"]}, null]',
    ],
)
def test_load_cases_rejects_invalid_dataset(tmp_path: Path, content: str) -> None:
    module = importlib.import_module("capability_capsule.eval.dataset")
    path = tmp_path / "cases.json"
    path.write_text(content, encoding="utf-8")
    with pytest.raises(ValueError):
        module.load_cases(path)


def test_load_cases_reports_missing_file(tmp_path: Path) -> None:
    module = importlib.import_module("capability_capsule.eval.dataset")
    with pytest.raises(FileNotFoundError):
        module.load_cases(tmp_path / "missing.json")


def test_load_cases_rejects_invalid_utf8(tmp_path: Path) -> None:
    module = importlib.import_module("capability_capsule.eval.dataset")
    path = tmp_path / "cases.json"
    path.write_bytes(b"\xff\xfe")
    with pytest.raises(UnicodeError):
        module.load_cases(path)
