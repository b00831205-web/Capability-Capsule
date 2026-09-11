"""Append-only JSONL persistence for versioned experiment records."""

from pathlib import Path

from pydantic import BaseModel


def append_jsonl(path: Path, record: BaseModel) -> None:
    """Append one validated model as single JSONL record."""

    path.parent.mkdir(parents = True, exist_ok = True)

    with path.open("a", encoding = "utf-8", newline = "\n") as stream:
        stream.write(record.model_dump_json())
        stream.write("\n")

def load_jsonl[RecordT: BaseModel](
        path: Path,
        model: type[RecordT]
) -> tuple[RecordT, ...]:
    """Load and validate every JSONL record using the requested model"""

    records: list[RecordT] =[]

    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                raise ValueError(
                    f"Invalid JSONL record in {path} at line {line_number}: "
                    f"blank lines are not allowed"
                )

            try:
                record = model.model_validate_json(line)

            except ValueError as error:
                raise ValueError(
                    f"Invalid JSONL record in {path} at line {line_number}: "
                    f"{error}"
                ) from error

            records.append(record)

    return tuple(records)