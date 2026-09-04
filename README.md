# DevAgent

[![tests](https://github.com/VOsokinP/rag-agent-platform/actions/workflows/ci.yml/badge.svg)](https://github.com/VOsokinP/rag-agent-platform/actions/workflows/ci.yml)

A RAG-based coding assistant over a real GitHub repository. It ingests code and docs,
and answers questions grounded in retrieved source with file-level citations. It also
runs an agent that can apply a patch to a throwaway copy of the checkout and execute the
repository's own test suite in a locked-down container, reporting what actually happened
rather than predicting it.

Target repository: [`fastapi/fastapi`](https://github.com/fastapi/fastapi).

## Status

Ingestion, the RAG core, and the agent loop are complete and unit-verified: 282 unit
tests pass with no network, database, Docker, or API key. `pytest -m integration` now
also runs the retrieval regression gate described under Evaluation below.

The end-to-end path is verified twice over.

`pytest -m integration` passes against a live clone of FastAPI, real Postgres, and the
real OpenAI API — clone, embed roughly 2,700 chunks, retrieve, and answer with citations
into real source. That first real run also exposed the index build-order bug described
under Known characteristics: retrieval returned nothing until the index was moved from
IVFFlat to HNSW.

`pytest -m "integration and sandbox"` puts the agent through the same path with a patch
that breaks `fastapi/params.py` at import time. The agent chose to run the tests, and
reported the real result:

> Yes, the change would break tests. The test file `tests/test_params_repr.py`
> encountered an error during collection due to a `RuntimeError` raised in
> `fastapi/params.py` [...] This indicates that the tests cannot be executed
> successfully.

Two model calls, ~1,100 tokens, well inside the 8-step budget.

## Setup

```bash
cp .env.example .env          # then put a real OPENAI_API_KEY in it
docker compose up -d postgres
pip install -e ".[dev]"       # also installs the `devagent` command
python -c "from devagent.db.session import init_db; init_db()"
```

The schema step is required: `init_db()` is deliberately not called automatically by
the API, so the `chunks` table and its pgvector index won't exist until you run it.

On Windows, the project's interpreter lives at `.venv/Scripts/python` — use that (or
`.venv/Scripts/python -m pytest`, etc.) if you're not already inside an activated venv.

## Usage

Start the server:

```bash
uvicorn devagent.api.main:app --reload
```

Then drive it with the `devagent` CLI from a second terminal. It is a thin client
over the HTTP API, so anything it can do, any other client can do too:

```bash
devagent health                              # is the server up?
devagent ingest                              # clone, chunk, embed, store
devagent query "what does Depends do?"       # answer, with sources
devagent ask "what does Depends do?"         # same question, but with tools
```

`devagent ask` runs the agent rather than a single retrieval-and-answer pass. It can
search the corpus, read files, blame lines, and run the repository's tests. Pass a
unified diff with `--patch` to have it answer about code that does not exist yet:

```bash
devagent ask "would this break any tests? check tests/test_params_repr.py" --patch change.diff
```

The patch is applied to a throwaway copy of the checkout, never to the checkout itself,
and the copy is deleted when the request finishes. A diff that does not apply is a 400
and costs nothing, because it fails before the first model call.

Running the tests needs the runner image, built once from the ingested checkout:

```bash
docker build -f docker/runner.Dockerfile -t devagent-runner:fastapi data/repos/fastapi
```

`devagent query` takes `-k` to change how many chunks are retrieved and `--repo` to
restrict retrieval to one label. `--url` (or `DEVAGENT_URL`) points at a server other
than `http://localhost:8000`. Every command exits non-zero on failure, including an
ingest that finished with failed embedding batches -- that case leaves stale content
in the corpus, so it is not reported as success.

The CLI is a convenience, not a required layer. The endpoints take plain JSON:

```bash
curl -X POST localhost:8000/query -H 'content-type: application/json' -d '{"question": "what does Depends do?"}'
```

PowerShell aliases `curl` to `Invoke-WebRequest`, whose arguments differ; there, use
`curl.exe`, `Invoke-RestMethod`, or the CLI above.

## Tests

```bash
pytest                  # unit tests; no network, no database, no Docker, no API key
pytest -m integration   # real repo clone, real Postgres, real OpenAI calls
pytest -m sandbox       # real Docker; builds nothing, but needs the runner image
```

The unit suite is what CI runs, on Python 3.12 and 3.13. It needs no services and no
secrets, which is the point of keeping it offline: a clone of this repository can run it
immediately.

`pytest -m integration` covers two kinds of test. The database-only ones
(`tests/db/test_index_build_order.py`) need just a running Postgres — no key, no
network, no spend — and are worth running after any change to the schema or to
retrieval. The rest require a running Postgres (`docker compose up -d postgres`),
a real `OPENAI_API_KEY`, and network access. It clones and embeds the FastAPI repo, so
it will spend a small amount of real money on embeddings (a few cents at
`text-embedding-3-small` pricing).

`pytest -m sandbox` needs Docker and the runner image above. It proves the container
runs the *patched* copy rather than the package installed in the image — the assumption
every answer about a patch depends on.

Both markers are deselected by default via the `not integration and not sandbox` marker
expression in `pyproject.toml`, so a fresh clone runs the unit suite and nothing else.

## Evaluation

Retrieval is measured, not asserted. `evals/golden.yaml` holds hand-labelled
questions over the ingested FastAPI corpus, split into `identifier` questions
(mostly a literal name) and `conceptual` ones (about behaviour).

```bash
devagent eval                    # the table, then the regression gate
devagent eval --kind identifier  # one population
devagent eval --record           # overwrite the baseline, then commit it
```

`evals/baseline.json` records the floor, and `pytest -m integration` fails if
overall recall@5 drops more than 0.02 below it. "Recall@k" here is really
success@k: the fraction of *questions* with a hit somewhere in the top k, not
the fraction of expected files retrieved. The baseline also stores the rank of
every individual question and the embedding model that produced them — the
ranks so a retrieval change can be compared question by question rather than
as two averages, and the model because `text-embedding-3-small` is an alias whose
snapshot can move underneath a corpus. A red gate says which of the two it is.

The baseline also records which repo it was measured against and the commit
that repo's checkout was at when the baseline was recorded. The gate fails if
the repo changes — a different repo is a different corpus, and the numbers
stop being comparable — but not if the commit does: a re-ingest at a newer
commit is expected to move the numbers, and calls for a deliberate
`devagent eval --record` rather than being read as a regression.

## Known characteristics

- **The embedding index is HNSW, deliberately.** `init_db()` has to create the index
  before anything is ingested, and an IVFFlat index derives its centroids from the rows
  present when it is built. Built on an empty table it has none, so rows inserted later
  are effectively unreachable through it: an index scan returns a fraction of the true
  nearest neighbours, or nothing at all, while the rows sit in the table. That is a
  correctness failure, not a slow query, and it presents as `/query` answering "no
  chunks retrieved" right after a successful ingest. HNSW builds its graph incrementally
  as rows arrive, so it is correct on an empty table and needs no rebuild step. The
  cost is slower inserts and more memory, neither material at this corpus size.
- **Similarity scores span `[-1, 1]`.** A slightly negative score means the chunk is
  anti-correlated with the query, not merely irrelevant. Scores are deliberately not
  clamped to `[0, 1]`.
- **Retrieved repository content is inserted into the model's prompt as data.** The
  prompt instructs the model to treat retrieved chunks as data rather than instructions,
  but the block delimiters around that content are not escaped. This is only
  appropriate for a corpus you trust, and it matters more now that the agent has tools:
  injected content could otherwise influence a tool call rather than just an answer.
  **The tool surface is the boundary that actually holds.** It is deliberately small and
  deliberately dull. `read_file` and `run_tests` are confined to the throwaway copy and
  `git_blame` to the checkout, all by resolved-path containment rather than string
  checks. The test container gets no network, 2 GB, two CPUs, 512 processes, a non-root
  user, a read-only mount and a wall-clock timeout. Test targets must live under
  `tests/`. The loop stops after a fixed number of tool calls. Nothing writes to the
  checkout, and nothing the model says can widen any of that — the tools are bound with
  their context closed over, so the workspace root is not an argument the model can set.
- **The tests that run are the ones in the container image's pinned environment.** The
  runner image installs from the target repository's own lockfile rather than resolving
  its dependencies fresh. Resolving freely once picked a newer `anyio` whose deprecation
  warning, under FastAPI's `filterwarnings = ["error"]`, turned 444 test files into
  collection errors before any patch existed — which would have made every answer about
  a patch a false positive. A useful "did this break anything" needs a green baseline
  more than it needs current dependencies.
- **A test file that will not import counts as a failure, not as a broken runner.**
  pytest exits 2 on a collection error, which is exactly what a patch that breaks a
  module at import time produces. That is a real finding about the code and is reported
  as one. A run that genuinely could not happen — no image, a timeout, a container that
  died — is reported separately and never as "nothing broke".
- **The tutorial tests are refused, not run.** `tests/test_tutorial/` writes into the
  working tree, which is mounted read-only, so those tests fail on `OSError` regardless
  of the patch. Asking for one gets a clear refusal rather than 21 failures that look
  like the patch's fault. Everything else is green in the container: 2,044 passed
  against the unmodified checkout.
- **There are two ways to reach a model, deliberately.** `Provider` wraps the OpenAI SDK
  directly for embeddings and the single-shot `/query` answer; LangChain and LangGraph
  are used for the agent and nowhere else. The split is the point rather than drift:
  ingestion and retrieval stay framework-free and swappable, and the framework is
  confined to the one place that needs tool calling. `chat_model()` is the only
  constructor for it.
- **`/agent` has no concurrency limit.** It is a synchronous endpoint, so FastAPI runs it
  in anyio's thread pool — forty workers by default, with nothing bounding concurrent
  agent runs. Forty in flight would be ~1.4 GB of temporary copies and forty containers
  each permitted 2 GB. That is theoretical for a single-operator local tool, and a queue
  is more machinery than the problem deserves, but it is a real ceiling rather than an
  oversight.
- **`/query` and `/agent` build their citations differently, on purpose.** `/query` goes
  through `answer.Citation`, which means "what the model was actually shown" — a
  narrower claim than "what retrieval returned", and the one worth making when a single
  prompt is the whole answer. `/agent` may search several times across a run, so it
  reports the deduplicated union of what retrieval returned, keeping each span's best
  score.
- **Methods are embedded twice, and that is deliberate.** A class produces one chunk
  for the whole class *and* one chunk per method, so a method's source is embedded both
  standalone and inside its class. Measured on the FastAPI corpus that is 1,224,827
  embedded characters against 766,214 source characters — roughly a 1.6x cost
  multiplier. It is kept because class-level context genuinely helps retrieval: a
  question about a class is answered better by a chunk that shows the class as a whole
  than by an arbitrary one of its methods. Chunk-size capping reduces this for large
  classes, which now contribute a header chunk (decorators, signature, docstring)
  instead of a full body.
- **Chunks are capped at 8,000 characters.** `text-embedding-3-small` rejects the entire
  request if any single input exceeds 8,192 tokens, so one oversized chunk would
  silently discard its whole batch. Oversized functions and Markdown sections are split;
  oversized classes become header chunks.
- **Re-ingestion is not atomic per repository, only per file.** A file's previously
  stored chunks are deleted only after its replacement chunks have been embedded
  successfully. If an embedding batch fails, the affected file's *older* chunks are
  left in place rather than deleted, so a partial ingest doesn't leave you with holes
  in retrievable content — but it does mean stale content can persist. Check
  `batches_failed` in the `/ingest` response to see whether this happened.
- **The eval scores retrieval, not answers.** A hit is a retrieved chunk whose
  file is one a human labelled as containing the answer. It says nothing about
  whether the model then used that chunk well, and nothing about whether the
  answer was faithful — that needs a judge model and is deliberately a separate
  piece of work. Symbol labels are recorded and never scored: chunk boundaries
  move whenever the chunker changes, so a symbol-level metric would measure the
  chunker rather than retrieval.

## Architecture

| Layer | Module | Responsibility |
|---|---|---|
| Ingestion | `devagent/ingestion/` | Clone, walk, chunk (AST for Python, headings for Markdown), embed, upsert |
| Storage | `devagent/db/` | The `chunks` table with a pgvector embedding column |
| Retrieval | `devagent/retrieval/` | Top-k cosine similarity search |
| Generation | `devagent/answer.py` | Grounded prompt construction and citation assembly |
| Providers | `devagent/llm/` | Swappable embedding/completion backend |
| Agent | `devagent/agent/` | The LangGraph loop, its four tools, and the run's trace |
| Sandbox | `devagent/sandbox/` | The throwaway patched copy and the container test runner |
| API | `devagent/api/` | `/health`, `/ingest`, `/query`, `/agent` |
| CLI | `devagent/cli.py` | The `devagent` command; an HTTP client for the API |

The agent alternates two nodes — ask the model, run the tools it asked for — until it
answers or spends its step budget. Its four tools are `search_code` (the same retrieval
`/query` uses), `read_file`, `git_blame`, and `run_tests`. An answer cut short by the
budget is returned flagged rather than dressed up as finished.
