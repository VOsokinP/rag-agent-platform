# DevAgent

A RAG-based coding assistant over a real GitHub repository. It ingests code and docs,
and answers questions grounded in retrieved source with file-level citations.

Target repository: [`fastapi/fastapi`](https://github.com/fastapi/fastapi).

## Status

Ingestion and the RAG core are complete and unit-verified: 155 unit tests pass with no
network or database access.

The end-to-end path is verified: `pytest -m integration` passes all 6 tests against a
live clone of FastAPI, real Postgres, and the real OpenAI API — clone, embed roughly
2,700 chunks, retrieve, and answer with citations into real source. That first real run
also exposed the index build-order bug described under Known characteristics: retrieval
returned nothing until the index was moved from IVFFlat to HNSW.

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
pytest                  # unit tests; no network, no database, no API key required
pytest -m integration   # real repo clone, real Postgres, real OpenAI calls
```

`pytest -m integration` covers two kinds of test. The database-only ones
(`tests/db/test_index_build_order.py`) need just a running Postgres — no key, no
network, no spend — and are worth running after any change to the schema or to
retrieval. The rest require a running Postgres (`docker compose up -d postgres`),
a real `OPENAI_API_KEY`, and network access. It clones and embeds the FastAPI repo, so
it will spend a small amount of real money on embeddings (a few cents at
`text-embedding-3-small` pricing). It is deselected by default via the
`not integration` marker expression in `pyproject.toml`.

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
  appropriate for a corpus you trust. It matters more once the agent gains tools that
  can act on the codebase, where prompt-injected content could otherwise influence tool
  calls.
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

## Architecture

| Layer | Module | Responsibility |
|---|---|---|
| Ingestion | `devagent/ingestion/` | Clone, walk, chunk (AST for Python, headings for Markdown), embed, upsert |
| Storage | `devagent/db/` | The `chunks` table with a pgvector embedding column |
| Retrieval | `devagent/retrieval/` | Top-k cosine similarity search |
| Generation | `devagent/answer.py` | Grounded prompt construction and citation assembly |
| Providers | `devagent/llm/` | Swappable embedding/completion backend |
| API | `devagent/api/` | `/health`, `/ingest`, `/query` |
| CLI | `devagent/cli.py` | The `devagent` command; an HTTP client for the API |
