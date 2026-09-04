"""Command-line client for a running DevAgent server.

This drives the HTTP API rather than calling the library in process, so using
the CLI exercises the same path any other client would take: if `devagent query`
works, `/query` works. The value it adds over a raw HTTP call is in the three
places that path is easy to get wrong -- the read timeout on a long ingest, the
`batches_failed` count that silently means stale content, and the difference
between "the server is down" and "the corpus is empty".

`devagent eval` is the one deliberate exception. It is an operator tool rather
than a service feature -- it reads a labelled file off disk, spends money on
embeddings, and gates a commit -- so it has no endpoint to drive and runs in
process. Every other command earns its place by being something a client would
call; this one earns its place by being something only the operator would run.
"""

import argparse
import json
import os
import subprocess
import sys
from datetime import date
from pathlib import Path
from typing import Any

import httpx

from devagent.config import get_settings
from devagent.db.session import session_scope
from devagent.eval.baseline import check, load_baseline, to_baseline
from devagent.eval.dataset import KINDS, GoldenSetError, load_golden
from devagent.eval.runner import KS, RETRIEVAL_K, Report, retriever, run_eval
from devagent.llm.provider import get_provider

DEFAULT_URL = "http://localhost:8000"

# Ingest embeds an entire repository and legitimately runs for minutes. httpx's
# 5s default read timeout would abandon the request while the server carries on
# working, reporting a failure for an ingest that actually succeeds.
INGEST_TIMEOUT = httpx.Timeout(connect=10.0, read=None, write=30.0, pool=10.0)
QUERY_TIMEOUT = httpx.Timeout(connect=10.0, read=120.0, write=30.0, pool=10.0)

START_SERVER_HINT = "Start it with:\n    uvicorn devagent.api.main:app --reload"


SYMBOL_WIDTH = 60
# Tool results get more room than a table column: they are printed one per
# line, with nothing to stay aligned with.
RESULT_WIDTH = 100


def _shorten(symbol: str, width: int = SYMBOL_WIDTH) -> str:
    """Trim a symbol to a width the table can hold.

    Markdown chunks use their full heading path as the symbol, which routinely
    runs past 200 characters and wraps every row into an unreadable block.

    The marker is ASCII "..." rather than a single ellipsis character on
    purpose: the Windows console defaults to cp1252, where a Unicode ellipsis
    prints as a replacement box or raises UnicodeEncodeError outright.
    """
    if len(symbol) <= width:
        return symbol
    return symbol[: width - 3].rstrip() + "..."


def _headline(result: str) -> str:
    """The first line of a tool result, which is where every tool puts its verdict.

    `run_tests` returns a summary line followed by the pytest output tail, and a
    one-line-per-step list can hold only the summary. Truncating the whole blob
    on width instead cut the counts and the duration off the end -- the two
    things that distinguish a test that ran from a model that guessed.
    """
    return _shorten(result.split("\n", 1)[0], RESULT_WIDTH)


def _fail(message: str) -> int:
    print(message, file=sys.stderr)
    return 1


def _request(
    base_url: str,
    method: str,
    path: str,
    timeout: httpx.Timeout,
    transport: httpx.BaseTransport | None,
    json: dict[str, Any] | None = None,
) -> tuple[Any, int | None]:
    """Perform one API call, returning `(payload, status)`.

    Returns `(None, None)` after printing a diagnosis when the server could not
    be reached at all, which is a different problem from any HTTP status.
    """
    with httpx.Client(base_url=base_url, transport=transport) as client:
        try:
            response = client.request(method, path, json=json, timeout=timeout)
        except httpx.ConnectError:
            _fail(f"No DevAgent server at {base_url}.\n{START_SERVER_HINT}")
            return None, None
        except httpx.ReadTimeout:
            _fail(f"{base_url}{path} timed out. The server may still be working.")
            return None, None

    try:
        return response.json(), response.status_code
    except ValueError:
        return response.text, response.status_code


def _detail(payload: Any, status: int) -> str:
    if isinstance(payload, dict) and "detail" in payload:
        return str(payload["detail"])
    return f"HTTP {status}: {payload}"


def _health(args, transport) -> int:
    payload, status = _request(args.url, "GET", "/health", QUERY_TIMEOUT, transport)
    if status is None:
        return 1
    if status != 200:
        return _fail(_detail(payload, status))
    print(f"{args.url} -> {payload.get('status', payload)}")
    return 0


def _ingest(args, transport) -> int:
    body = {"repo": args.repo}
    print(
        f"Ingesting via {args.url} -- this embeds the whole repository and takes a while."
    )
    payload, status = _request(
        args.url, "POST", "/ingest", INGEST_TIMEOUT, transport, json=body
    )
    if status is None:
        return 1
    if status != 200:
        return _fail(_detail(payload, status))

    print(f"repo:            {payload['repo']}")
    print(f"files processed: {payload['files_processed']}")
    print(f"chunks written:  {payload['chunks_written']}")

    skipped = payload.get("files_skipped") or []
    if skipped:
        print(f"files skipped:   {len(skipped)}")
        for path in skipped:
            print(f"  - {path}")

    failed = payload.get("batches_failed", 0)
    if failed:
        # A failed batch does not delete that file's previous chunks, by design:
        # a partial ingest leaves old content rather than a hole. The corpus is
        # therefore stale rather than incomplete, which is invisible unless the
        # count is surfaced -- so this is a failure exit, not a footnote.
        return _fail(
            f"\n{failed} embedding batch(es) failed. Those files kept their previous "
            "chunks, so the corpus may be stale. Re-run the ingest."
        )
    return 0


def _query(args, transport) -> int:
    question = args.question.strip()
    if not question:
        return _fail("The question is empty.")

    body: dict[str, Any] = {"question": question, "k": args.k}
    if args.repo is not None:
        body["repo"] = args.repo
    if not args.retrieval:
        body["retrieval"] = False

    payload, status = _request(
        args.url, "POST", "/query", QUERY_TIMEOUT, transport, json=body
    )
    if status is None:
        return 1
    if status == 409:
        return _fail(
            f"{_detail(payload, status)}\nIngest the repository first:\n    devagent ingest"
        )
    if status != 200:
        return _fail(_detail(payload, status))

    # Before the answer, not after: the label has to be read first to be a
    # label at all. An ungrounded answer that arrives looking like a grounded
    # one is exactly what citations exist to prevent.
    if payload.get("retrieval_used") is False:
        print("[no retrieval] answered from the model alone -- nothing grounds this.\n")

    print(payload["answer"])

    citations = payload.get("citations") or []
    if citations and not args.brief:
        print("\nSources:")
        width = max(len(c["file_path"]) for c in citations)
        for citation in citations:
            symbol = _shorten(citation.get("symbol") or "-")
            lines = f"{citation['start_line']}-{citation['end_line']}"
            print(
                f"  {citation['file_path']:<{width}}  {lines:>11}  "
                f"{citation['score']:+.2f}  {symbol}"
            )
    return 0


def _ask(args, transport) -> int:
    question = args.question.strip()
    if not question:
        return _fail("The question is empty.")

    body: dict[str, Any] = {"question": question, "k": args.k}
    if args.patch:
        try:
            body["patch"] = Path(args.patch).read_text(encoding="utf-8")
        except OSError as exc:
            return _fail(f"Could not read the patch file: {exc}")

    # Agent runs make several model calls and at least one container test run,
    # so they get the ingest timeout rather than the query one.
    payload, status = _request(
        args.url, "POST", "/agent", INGEST_TIMEOUT, transport, json=body
    )
    if status is None:
        return 1
    if status != 200:
        return _fail(_detail(payload, status))

    print(payload["answer"])

    steps = payload.get("steps") or []
    if steps and not args.brief:
        print("\nSteps:")
        for number, step in enumerate(steps, start=1):
            print(
                f"  {number}. {step['tool']}({step['input']}) -> {_headline(step['result'])}"
            )

    usage = payload.get("usage") or {}
    if usage:
        print(
            f"\n{usage['llm_calls']} model calls, "
            f"{usage['prompt_tokens'] + usage['completion_tokens']} tokens, "
            f"~${usage['cost_usd']:.3f} (estimated)"
        )
    if payload.get("budget_exhausted"):
        print(
            "\nWarning: the step budget was exhausted; this answer may be incomplete."
        )
    return 0


def format_report(report: Report, kind: str | None = None) -> str:
    """Render an eval report as a small table.

    Every row carries `n=`. A recall figure without its sample size is how a
    population too thin to resolve a change gets read as a result anyway.
    """
    rows = [("overall", report.overall)]
    rows += [(name, scores) for name, scores in report.by_kind.items()]
    if kind is not None:
        rows = [(name, scores) for name, scores in rows if name == kind]

    header = "  ".join(f"r@{k:<5}" for k in KS)
    lines = [
        f"model: {report.embedding_model}   retrieved k={report.k}",
        "",
        f"{'population':<12}  {'n':>5}  {header}  {'MRR':>6}",
    ]
    for name, scores in rows:
        recalls = "  ".join(f"{scores.recall[k]:<7.2f}" for k in KS)
        lines.append(f"{name:<12}  n={scores.n:<3}  {recalls}  {scores.mrr:>6.3f}")

    # Which half of the corpus answered. Hybrid search moves hits from docs to
    # code, and an aggregate recall number hides that completely -- this line is
    # the one that makes such a shift legible.
    shown = [r for r in report.results if kind is None or r.kind == kind]
    code = sum(1 for r in shown if r.hit_source == "code")
    docs = sum(1 for r in shown if r.hit_source == "docs")
    missed = sum(1 for r in shown if r.hit_source is None)
    lines += ["", f"hits: {code} code, {docs} docs, {missed} missed"]

    if kind is not None:
        # The table above is filtered; the gate below is not. Without this the
        # operator reads a pass/fail verdict as if it were about the population
        # on screen.
        lines += ["", "(the gate is on overall recall@5, not the filtered population)"]

    return "\n".join(lines)


def _corpus_empty(report: Report) -> bool:
    """True when every question retrieved nothing -- the signature of an empty
    or unreachable corpus, not a retrieval regression. `n == 0` is a different,
    already-visible state (an empty golden set), so it is excluded here."""
    return report.overall.n > 0 and all(not r.retrieved_files for r in report.results)


def _corpus_commit(repo_dir: Path) -> str | None:
    """The ingested checkout's HEAD, recorded so a baseline names the corpus it
    measured. Best-effort: the checkout is a local artifact that may be absent,
    and a baseline without it is still better than one that pins nothing."""
    try:
        completed = subprocess.run(
            ["git", "-C", str(repo_dir), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return completed.stdout.strip() if completed.returncode == 0 else None


def _eval(args, transport) -> int:
    """Run the golden set against retrieval in process.

    `transport` is accepted and ignored so every handler keeps one signature.
    There is no server to talk to: see the module docstring.
    """
    settings = get_settings()
    try:
        questions = load_golden(Path(args.golden))
    except GoldenSetError as exc:
        return _fail(str(exc))

    with session_scope() as session:
        report = run_eval(
            questions,
            retriever(get_provider(), session, repo=args.repo, k=args.k),
            embedding_model=settings.embedding_model,
            k=args.k,
        )

    print(format_report(report, kind=args.kind))

    if _corpus_empty(report):
        return _fail(
            "\nEvery question retrieved nothing. The corpus is empty or "
            "unreachable, not a retrieval regression -- ingest the repository "
            "before running the gate:\n    devagent ingest"
        )

    if args.record:
        payload = to_baseline(
            report,
            recorded=str(date.today()),
            repo=args.repo,
            corpus_commit=_corpus_commit(settings.repo_dir),
        )
        Path(args.baseline).write_text(
            json.dumps(payload, indent=2) + "\n", encoding="utf-8"
        )
        print(f"\nRecorded {args.baseline}. Commit it deliberately.")
        return 0

    baseline_path = Path(args.baseline)
    if not baseline_path.is_file():
        print(f"\nNo baseline at {args.baseline}; run with --record to create one.")
        return 0

    failures = check(report, load_baseline(baseline_path), repo=args.repo)
    if failures:
        return _fail("\n" + "\n".join(f"FAIL: {failure}" for failure in failures))
    print("\nGate passed.")
    return 0


def _add_brief(parser: argparse.ArgumentParser) -> None:
    """One flag, same meaning everywhere: drop the supporting table.

    `query` and `ask` show different tables -- citations and tool calls -- but
    "the answer without the table under it" is one idea, and one idea should
    not cost the reader two flags to remember.
    """
    parser.add_argument(
        "--brief",
        action="store_true",
        help="Answer only, without the sources or steps table.",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="devagent", description="Talk to a running DevAgent server."
    )
    parser.add_argument(
        "--url",
        default=os.environ.get("DEVAGENT_URL", DEFAULT_URL),
        help=f"Server base URL (env: DEVAGENT_URL, default: {DEFAULT_URL}).",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    health = subparsers.add_parser("health", help="Check that the server is reachable.")
    health.set_defaults(handler=_health)

    ingest = subparsers.add_parser(
        "ingest", help="Clone, chunk, embed and store the repo."
    )
    ingest.add_argument("--repo", default=None, help="Label recorded on each row.")
    ingest.set_defaults(handler=_ingest)

    query = subparsers.add_parser(
        "query", help="Ask a question about the ingested repo."
    )
    query.add_argument("question")
    query.add_argument(
        "-k", type=int, default=8, help="Chunks to retrieve (default: 8)."
    )
    query.add_argument(
        "--repo", default=None, help="Restrict retrieval to one repo label."
    )
    _add_brief(query)
    query.add_argument(
        "--no-retrieval",
        dest="retrieval",
        action="store_false",
        help="Baseline: answer from the model alone, with no corpus.",
    )
    query.set_defaults(handler=_query)

    ask = subparsers.add_parser("ask", help="Ask the agent, which can run tests.")
    ask.add_argument("question")
    _add_brief(ask)
    ask.add_argument(
        "--patch", default=None, help="Path to a unified diff to apply first."
    )
    ask.add_argument("-k", type=int, default=8, help="Chunks to retrieve (default: 8).")
    ask.set_defaults(handler=_ask)

    evaluate = subparsers.add_parser(
        "eval", help="Score retrieval against the labelled golden set."
    )
    evaluate.add_argument(
        "-k",
        type=int,
        default=RETRIEVAL_K,
        help=f"Chunks to retrieve (default: {RETRIEVAL_K}).",
    )
    evaluate.add_argument(
        "--kind", default=None, choices=list(KINDS), help="Report only one population."
    )
    evaluate.add_argument(
        "--repo",
        default=None,
        help="Restrict retrieval to one ingested repo; omit to search every repo.",
    )
    evaluate.add_argument(
        "--golden", default="evals/golden.yaml", help="Path to the labelled set."
    )
    evaluate.add_argument(
        "--baseline", default="evals/baseline.json", help="Path to the recorded floor."
    )
    evaluate.add_argument(
        "--record",
        action="store_true",
        help="Overwrite the baseline with this run's numbers.",
    )
    evaluate.set_defaults(handler=_eval)

    return parser


def main(
    argv: list[str] | None = None, *, transport: httpx.BaseTransport | None = None
) -> int:
    """Run one CLI invocation and return its exit code.

    `transport` is injectable so tests can drive the real request-building and
    output formatting through httpx's MockTransport, with no server involved.
    """
    # Answers are model output: em dashes, curly quotes, the occasional arrow.
    # The Windows console is cp1252 by default, where those either print as a
    # replacement box or raise UnicodeEncodeError and take the whole answer
    # with them. `errors="replace"` keeps a terminal that genuinely cannot
    # render a character from losing the rest of the text.
    reconfigure = getattr(sys.stdout, "reconfigure", None)
    if reconfigure is not None:
        reconfigure(encoding="utf-8", errors="replace")

    args = build_parser().parse_args(argv)
    return args.handler(args, transport)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
