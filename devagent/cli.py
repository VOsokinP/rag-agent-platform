"""Command-line client for a running DevAgent server.

This drives the HTTP API rather than calling the library in process, so using
the CLI exercises the same path any other client would take: if `devagent query`
works, `/query` works. The value it adds over a raw HTTP call is in the three
places that path is easy to get wrong -- the read timeout on a long ingest, the
`batches_failed` count that silently means stale content, and the difference
between "the server is down" and "the corpus is empty".
"""

import argparse
import os
import sys
from typing import Any

import httpx

DEFAULT_URL = "http://localhost:8000"

# Ingest embeds an entire repository and legitimately runs for minutes. httpx's
# 5s default read timeout would abandon the request while the server carries on
# working, reporting a failure for an ingest that actually succeeds.
INGEST_TIMEOUT = httpx.Timeout(connect=10.0, read=None, write=30.0, pool=10.0)
QUERY_TIMEOUT = httpx.Timeout(connect=10.0, read=120.0, write=30.0, pool=10.0)

START_SERVER_HINT = (
    "Start it with:\n    uvicorn devagent.api.main:app --reload"
)


SYMBOL_WIDTH = 60


def _shorten(symbol: str) -> str:
    """Trim a symbol to a width the table can hold.

    Markdown chunks use their full heading path as the symbol, which routinely
    runs past 200 characters and wraps every row into an unreadable block.

    The marker is ASCII "..." rather than a single ellipsis character on
    purpose: the Windows console defaults to cp1252, where a Unicode ellipsis
    prints as a replacement box or raises UnicodeEncodeError outright.
    """
    if len(symbol) <= SYMBOL_WIDTH:
        return symbol
    return symbol[: SYMBOL_WIDTH - 3].rstrip() + "..."


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
) -> tuple[Any | None, int | None]:
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
    print(f"Ingesting via {args.url} -- this embeds the whole repository and takes a while.")
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

    print(payload["answer"])

    citations = payload.get("citations") or []
    if citations:
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

    ingest = subparsers.add_parser("ingest", help="Clone, chunk, embed and store the repo.")
    ingest.add_argument("--repo", default=None, help="Label recorded on each row.")
    ingest.set_defaults(handler=_ingest)

    query = subparsers.add_parser("query", help="Ask a question about the ingested repo.")
    query.add_argument("question")
    query.add_argument("-k", type=int, default=8, help="Chunks to retrieve (default: 8).")
    query.add_argument("--repo", default=None, help="Restrict retrieval to one repo label.")
    query.set_defaults(handler=_query)

    return parser


def main(argv: list[str] | None = None, *, transport: httpx.BaseTransport | None = None) -> int:
    """Run one CLI invocation and return its exit code.

    `transport` is injectable so tests can drive the real request-building and
    output formatting through httpx's MockTransport, with no server involved.
    """
    args = build_parser().parse_args(argv)
    return args.handler(args, transport)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
