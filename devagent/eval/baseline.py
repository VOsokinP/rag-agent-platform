"""The recorded floor, and the gate that defends it.

Without this the numbers get printed once, admired, and never looked at again.

The gate has to distinguish two kinds of red. A retrieval change that made
things worse is the one worth failing on. An embedding-model snapshot moving
underneath the corpus is not: `text-embedding-3-small` is an alias, OpenAI
reissues snapshots behind it, and when that happens every stored vector is from
the old snapshot and every query vector from the new one. A gate that cannot
tell them apart gets muted the first time it cries wolf, so the model string is
recorded and compared.
"""

import json
from pathlib import Path
from typing import Any

from devagent.eval.runner import KS, Report, Scores

# On overall recall@5. Not zero: see the module docstring. Re-baselining is a
# deliberate commit, so this only has to absorb noise, not real movement.
TOLERANCE = 0.02
GATE_K = 5


def _scores(scores: Scores) -> dict[str, Any]:
    # JSON object keys are strings, so the k values are stringified here rather
    # than at every read site.
    return {
        "n": scores.n,
        "recall": {str(k): scores.recall[k] for k in KS},
        "mrr": scores.mrr,
    }


def to_baseline(
    report: Report,
    recorded: str,
    repo: str | None = None,
    corpus_commit: str | None = None,
) -> dict[str, Any]:
    """Render a Report as the committed baseline document."""
    return {
        "recorded": recorded,
        "embedding_model": report.embedding_model,
        "k": report.k,
        "repo": repo,
        "corpus_commit": corpus_commit,
        "overall": _scores(report.overall),
        "by_kind": {kind: _scores(scores) for kind, scores in report.by_kind.items()},
        # Per question, not just aggregates. This is what makes Milestone 2's
        # comparison paired -- win/loss/tie over the same questions -- which
        # detects a real gain at a sample size where two means detect nothing.
        "ranks": {result.question: result.rank for result in report.results},
        # Recorded, never gated. Hybrid search shifts hits between the two
        # halves of the corpus, and once it has landed the pre-hybrid split
        # cannot be recovered without reverting it -- so it is captured at the
        # only moment it is free. A shift is a thing to look at, not a
        # regression to fail on, so `check` ignores it.
        "hit_source": {result.question: result.hit_source for result in report.results},
    }


def load_baseline(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def check(
    report: Report, baseline: dict[str, Any], *, repo: str | None = None
) -> list[str]:
    """Return one string per gate failure. An empty list means the gate passed."""
    failures = []

    recorded_model = baseline.get("embedding_model")
    if recorded_model != report.embedding_model:
        failures.append(
            f"embedding model changed: baseline recorded {recorded_model!r}, "
            f"this run used {report.embedding_model!r}. The corpus was embedded "
            "with the old one, so these numbers are not comparable -- re-ingest "
            "and re-baseline rather than reading this as a regression."
        )

    recorded_k = baseline.get("k")
    if recorded_k != report.k:
        failures.append(
            f"retrieval depth changed: baseline recorded k={recorded_k}, this run "
            f"used k={report.k}. recall@{GATE_K} is not comparable across depths -- "
            "re-run at the recorded k, or re-baseline deliberately."
        )

    recorded_repo = baseline.get("repo")
    if recorded_repo != repo:
        failures.append(
            f"corpus changed: baseline recorded repo {recorded_repo!r}, this run "
            f"used {repo!r}. These are different corpora, so the numbers are not "
            "comparable."
        )
    # `corpus_commit` is recorded but deliberately not gated here: a re-ingest at
    # a newer commit is an expected, legitimate move whose numbers are supposed
    # to change, and failing on it would make the gate cry wolf.

    floor = baseline.get("overall", {}).get("recall", {}).get(str(GATE_K))
    if floor is None:
        failures.append(
            f"baseline has no overall recall@{GATE_K}; it is malformed or from "
            "an older format -- re-record it"
        )
    else:
        current = report.overall.recall[GATE_K]
        if current < floor - TOLERANCE:
            failures.append(
                f"recall@{GATE_K} fell to {current:.2f} from a recorded {floor:.2f} "
                f"(tolerance {TOLERANCE:.2f})"
            )
    return failures
