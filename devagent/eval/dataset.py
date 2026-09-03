"""Loading and validating the hand-written golden set.

The whole value of the set is that a person wrote it against the real
repository, which also means it carries hand-made mistakes. Every rule below
exists because its mistake is otherwise silent: a typo'd `kind` would quietly
create a third population and report it as its own row, and a duplicated
question would collide in `baseline.json`, where the question text is the key.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

KINDS = ("identifier", "conceptual")


class GoldenSetError(ValueError):
    """The golden set is missing, unparseable, or does not satisfy the schema."""


@dataclass(frozen=True)
class GoldenQuestion:
    """One labelled question.

    `expect_files` is the scored label: a hit is a retrieved chunk whose
    `file_path` is in this tuple. `expect_symbols` is recorded and deliberately
    *not* scored -- chunk boundaries move whenever the chunker changes, so a
    symbol-level metric would measure the chunker rather than retrieval.
    """

    question: str
    expect_files: tuple[str, ...]
    expect_symbols: tuple[str, ...]
    kind: str


def load_golden(path: Path) -> list[GoldenQuestion]:
    """Load and validate the golden set, or raise GoldenSetError."""
    try:
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    except OSError as exc:
        raise GoldenSetError(f"could not read {path}: {exc}") from exc
    except yaml.YAMLError as exc:
        raise GoldenSetError(f"{path} is not valid YAML: {exc}") from exc

    if not isinstance(raw, list):
        raise GoldenSetError(f"{path} must hold a list of entries, not {type(raw).__name__}")

    questions = [_entry(index, entry, path) for index, entry in enumerate(raw, start=1)]

    seen: set[str] = set()
    for question in questions:
        if question.question in seen:
            raise GoldenSetError(
                f"duplicate question in {path}: {question.question!r}. "
                "The question text is the key in baseline.json, so it must be unique."
            )
        seen.add(question.question)
    return questions


def _entry(index: int, entry: Any, path: Path) -> GoldenQuestion:
    where = f"{path} entry {index}"
    if not isinstance(entry, dict):
        raise GoldenSetError(f"{where} must be a mapping, not {type(entry).__name__}")

    question = entry.get("question")
    if not isinstance(question, str) or not question.strip():
        raise GoldenSetError(f"{where} has no question")

    kind = entry.get("kind")
    if kind not in KINDS:
        raise GoldenSetError(f"{where} has kind {kind!r}; expected one of {list(KINDS)}")

    files = entry.get("expect_files") or []
    if not isinstance(files, list) or not files:
        raise GoldenSetError(f"{where} has no expect_files; it could only ever score zero")
    for file_path in files:
        if not isinstance(file_path, str) or not file_path.strip():
            raise GoldenSetError(f"{where} has a blank entry in expect_files")
        if file_path.startswith("/") or ".." in Path(file_path).parts:
            raise GoldenSetError(
                f"{where}: {file_path!r} must be repo-relative. It is compared "
                "against RetrievedChunk.file_path, which is always a repo-relative "
                "posix path, so anything else can never match."
            )

    symbols = entry.get("expect_symbols") or []
    if not isinstance(symbols, list):
        raise GoldenSetError(f"{where} has a non-list expect_symbols")

    return GoldenQuestion(
        question=question.strip(),
        expect_files=tuple(files),
        expect_symbols=tuple(str(symbol) for symbol in symbols),
        kind=kind,
    )
