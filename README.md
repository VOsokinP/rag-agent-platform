# DevAgent

A RAG-based coding assistant over a real GitHub repository. It ingests code and docs,
and answers questions grounded in retrieved source with file-level citations.

Target repository: [`fastapi/fastapi`](https://github.com/fastapi/fastapi).

## Status

The Milestone 1 implementation (ingestion + RAG core) is complete and unit-verified:
122 unit tests pass with no network or database access. End-to-end verification
against the live FastAPI repository, a live Postgres instance, and a real OpenAI API
key is written (`pytest -m integration`) but has not yet been run in this environment
— there is no `OPENAI_API_KEY` available here. That run, and the by-hand `/query`
check against the running API, are pending until a key is supplied. See
`PROJECT_SPEC.md` for the full roadmap.

## Setup

```bash
cp .env.example .env          # then put a real OPENAI_API_KEY in it
docker compose up -d postgres
pip install -e ".[dev]"
python -c "from devagent.db.session import init_db; init_db()"
```

The schema step is required: `init_db()` is deliberately not called automatically by
the API, so the `chunks` table and its pgvector index won't exist until you run it.

On Windows, the project's interpreter lives at `.venv/Scripts/python` — use that (or
`.venv/Scripts/python -m pytest`, etc.) if you're not already inside an activated venv.

## Usage

```bash
uvicorn devagent.api.main:app --reload
```

Ingest the target repository (clones it on first run, then embeds every chunk):

```bash
curl -X POST localhost:8000/ingest -H 'content-type: application/json' -d '{}'
```

Ask a question:

```bash
curl -X POST localhost:8000/query \
  -H 'content-type: application/json' \
  -d '{"question": "what does Depends do?"}'
```

## Tests

```bash
pytest                  # unit tests; no network, no database, no API key required
pytest -m integration   # real repo clone, real Postgres, real OpenAI calls
```

`pytest -m integration` requires a running Postgres (`docker compose up -d postgres`),
a real `OPENAI_API_KEY`, and network access. It clones and embeds the FastAPI repo, so
it will spend a small amount of real money on embeddings (a few cents at
`text-embedding-3-small` pricing). It is deselected by default via the
`not integration` marker expression in `pyproject.toml`.

## Known characteristics

- **The IVFFlat index is degenerate until rebuilt.** It's built on an empty `chunks`
  table (schema creation happens before any ingest), so its clusters don't reflect
  real data. This affects query *speed*, not correctness — searches still return the
  true nearest neighbours, just without the intended speedup. After a large ingest, if
  retrieval latency becomes noticeable, rebuild it:
  ```sql
  REINDEX INDEX ix_chunks_embedding;
  ```
- **Similarity scores span `[-1, 1]`.** A slightly negative score means the chunk is
  anti-correlated with the query, not merely irrelevant. Scores are deliberately not
  clamped to `[0, 1]`.
- **Retrieved repository content is inserted into the model's prompt as data.** The
  prompt instructs the model to treat retrieved chunks as data rather than instructions,
  but the block delimiters around that content are not escaped. This is only
  appropriate for a corpus you trust. It matters more once the agent gains tools that
  can act on the codebase (Milestone 3), where prompt-injected content could otherwise
  influence tool calls.
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
