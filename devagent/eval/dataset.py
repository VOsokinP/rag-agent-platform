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
KEYS = frozenset({"question", "expect_files", "expect_docs", "expect_symbols", "kind"})


class GoldenSetError(ValueError):
    """The golden set is missing, unparseable, or does not satisfy the schema."""


@dataclass(frozen=True)
class GoldenQuestion:
    """One labelled question.

    `expect_files` is the scored label: a hit is a retrieved chunk whose
    `file_path` is in this tuple. `expect_symbols` is recorded and deliberately
    *not* scored -- chunk boundaries move whenever the chunker changes, so a
    symbol-level metric would measure the chunker rather than retrieval.
    `expect_docs` is scored exactly like `expect_files` -- the corpus is two
    thirds documentation, and a conceptual question answered from a docs page is
    answered correctly. It is a separate key rather than more entries in
    `expect_files` so the report can say which side of the corpus a hit came from.
    """

    question: str
    expect_files: tuple[str, ...]
    expect_symbols: tuple[str, ...]
    expect_docs: tuple[str, ...]
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


def _paths(entry: Any, key: str, where: str, required: bool) -> tuple[str, ...]:
    """Validate one list-of-repo-relative-paths key."""
    values = entry.get(key) or []
    if not isinstance(values, list) or (required and not values):
        raise GoldenSetError(f"{where} has no {key}; it could only ever score zero")
    for value in values:
        if not isinstance(value, str) or not value.strip():
            raise GoldenSetError(f"{where} has a blank entry in {key}")
        if value.startswith("/") or ".." in Path(value).parts:
            raise GoldenSetError(
                f"{where}: {value!r} must be repo-relative. It is compared "
                "against RetrievedChunk.file_path, which is always a repo-relative "
                "posix path, so anything else can never match."
            )
    return tuple(values)


def _entry(index: int, entry: Any, path: Path) -> GoldenQuestion:
    where = f"{path} entry {index}"
    if not isinstance(entry, dict):
        raise GoldenSetError(f"{where} must be a mapping, not {type(entry).__name__}")

    unknown = sorted(set(entry) - KEYS)
    if unknown:
        # An entire relabelling pass once landed under a key nothing read,
        # because unknown keys were ignored. Silence is the expensive failure
        # here: the file looks right, the loader is happy, and the labels score
        # nothing at all.
        raise GoldenSetError(
            f"{where} has unknown key(s) {unknown}; expected some of {sorted(KEYS)}"
        )

    question = entry.get("question")
    if not isinstance(question, str) or not question.strip():
        raise GoldenSetError(f"{where} has no question")

    kind = entry.get("kind")
    if kind not in KINDS:
        raise GoldenSetError(f"{where} has kind {kind!r}; expected one of {list(KINDS)}")

    files = _paths(entry, "expect_files", where, required=True)
    docs = _paths(entry, "expect_docs", where, required=False)

    symbols = entry.get("expect_symbols") or []
    if not isinstance(symbols, list):
        raise GoldenSetError(f"{where} has a non-list expect_symbols")

    return GoldenQuestion(
        question=question.strip(),
        expect_files=files,
        expect_symbols=tuple(str(symbol) for symbol in symbols),
        expect_docs=docs,
        kind=kind,
    )
