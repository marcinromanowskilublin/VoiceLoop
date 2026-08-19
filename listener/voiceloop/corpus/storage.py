from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, TypeVar

from pydantic import BaseModel

ModelT = TypeVar("ModelT", bound=BaseModel)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def word_count(text: str) -> int:
    return len(text.split())


def read_jsonl(path: Path, model: type[ModelT]) -> list[ModelT]:
    if not path.exists():
        return []
    records: list[ModelT] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
                records.append(model.model_validate(payload))
            except (json.JSONDecodeError, ValueError) as exc:
                raise ValueError(
                    f"Nieprawidłowy JSONL w {path.name}, linia {line_number}."
                ) from exc
    return records


def write_json(path: Path, value: BaseModel | dict[str, Any]) -> None:
    payload = value.model_dump(mode="json") if isinstance(value, BaseModel) else value
    _atomic_write(path, json.dumps(payload, ensure_ascii=False, indent=2) + "\n")


def write_jsonl(
    path: Path,
    records: list[BaseModel | dict[str, Any]],
) -> None:
    content = "".join(
        json.dumps(
            record.model_dump(mode="json") if isinstance(record, BaseModel) else record,
            ensure_ascii=False,
            separators=(",", ":"),
        )
        + "\n"
        for record in records
    )
    _atomic_write(path, content)


def write_text(path: Path, content: str) -> None:
    _atomic_write(path, content if content.endswith("\n") else content + "\n")


def _atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temp_path.write_text(content, encoding="utf-8", newline="\n")
        temp_path.replace(path)
    finally:
        temp_path.unlink(missing_ok=True)
