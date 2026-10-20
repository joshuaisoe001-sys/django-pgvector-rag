# Django pgvector RAG

Ask questions about your own documents and get answers that cite the exact passages they come from.

Upload text, Markdown or PDF files into a collection. The service splits them into chunks, embeds each chunk
and stores the vectors in PostgreSQL with pgvector. A question is embedded the same way, the closest chunks are
found by cosine similarity, and Claude answers from those chunks only, with citations that point back to
specific chunks in the database.

**Stack:** Python 3.12+, Django 6.1, Django REST Framework, PostgreSQL + pgvector (HNSW index), Anthropic
Claude API with native citations, Voyage AI embeddings, pytest, GitHub Actions.

![Asking a question about the fictional sample handbook](docs/screenshot.png)

*The screenshot shows offline mode (no API keys), where the answer is the best-matching sentence quoted verbatim.
With `ANTHROPIC_API_KEY` set, Claude writes the answer and cites each claim.*

## How it works

```mermaid
flowchart LR
    U[Upload .txt / .md / .pdf] --> X[Extract text] --> C[Chunk with offsets] --> E[Embed chunks] --> P[(PostgreSQL + pgvector)]
    Q[Question] --> QE[Embed question] --> R[Cosine search, top k, same embedding model] --> P
    R -->|no passage above threshold| NF[Answer: not found, no model call]
    R -->|passages| A[Claude: each passage as a cited document] --> ANS[Answer + citations to chunk IDs] --> L[(Question log: tokens, latency)]
```

1. **Extraction** (`knowledge/services/extraction.py`): UTF-8 text, UTF-16 with a byte-order mark, Markdown,
   and text-based PDFs via pypdf. Encrypted or corrupt PDFs are rejected with a clear message.
2. **Chunking** (`chunking.py`): about 900 characters per chunk with 150 characters of overlap, split at
   paragraph and then sentence boundaries. Every chunk stores its start and end offsets in the source text.
3. **Embedding** (`embeddings.py`): Voyage AI `voyage-3.5` at 1024 dimensions, the embedding provider
   Anthropic recommends (Claude itself does not produce embeddings). Without a key, an offline hashing embedder
   (stemmed words and word pairs, stop-words removed) keeps everything runnable. It matches shared words, not
   meaning.
4. **Storage** (`models.py`): `Chunk.embedding` is a pgvector `vector(1024)` column with an HNSW index using
   `vector_cosine_ops`. Each document records which embedding model produced its vectors.
5. **Retrieval** (`retrieval.py`): one SQL query ordering by cosine distance, limited to the collection and to
   chunks embedded by the same model as the question (vectors from different models are not comparable).
   Passages below `RAG_MIN_SIMILARITY` are dropped.
6. **Answering** (`answering.py`): each passage goes to Claude as a separate `document` content block with
   citations enabled. The response's `char_location` citations carry a `document_index`, which maps straight
   back to a chunk ID, so the API returns the cited text and the chunk it came from. If nothing relevant was
   retrieved, the API answers "not found" without calling the model at all.
7. **Logging** (`qa.py`): every question is stored with its answer, citations, retrieved chunk IDs, provider,
   model, input and output tokens, and latency.

### Claude integration details

- Model `claude-opus-5-5` by default (`CLAUDE_MODEL`), with `output_config.effort` set explicitly (`CLAUDE_EFFORT`,
  default `medium`).
- Server-side refusal fallbacks are on (`fallbacks="default"` with the `server-side-fallback-2026-07-01` beta),
  so a safety decline is retried on a fallback model inside the same call. A final `refusal` stop reason is
  reported as "declined", never shown as an answer.
- **Prompt-injection defence:** uploaded files are untrusted. Fixed rules live only in the system prompt, the
  passages are sent as document blocks (data, not instructions), and the model is given no tools, so its output
  cannot trigger any action. A test checks that hostile text stays inside the document data.
- API errors are mapped to readable messages (invalid key, rate limit, HTTP status, network) and returned as 503.
- An answer with no valid citations is marked `grounded: false`.

## API

All endpoints need authentication: a token (`Authorization: Token <key>`) or a session. Collections belong to one
user; other users get 404, never 403, so a collection's existence is not revealed.

| Method | Path | Purpose |
| --- | --- | --- |
| POST | `/api/auth/token/` | Exchange username and password for a token |
| GET, POST | `/api/collections/` | List or create collections |
| GET, PATCH, DELETE | `/api/collections/{id}/` | One collection |
| GET, POST | `/api/collections/{id}/documents/` | List documents, or upload one (multipart `file`, optional `title`) |
| GET, DELETE | `/api/documents/{id}/` | One document (deleting removes its chunks) |
| POST | `/api/collections/{id}/ask/` | `{"question": "..."}` returns the answer, citations and sources (rate-limited per user) |
| POST | `/api/collections/{id}/search/` | Retrieval only: the passages a question would use, no model call |
| GET | `/api/collections/{id}/questions/` | Question history with token usage |

Uploading the same bytes twice into a collection returns the existing document (`200` instead of `201`) and does
no embedding work.

```bash
TOKEN=$(curl -s -X POST localhost:8000/api/auth/token/ -d username=demo -d password=... | python -c "import sys,json;print(json.load(sys.stdin)['token'])")
curl -s -X POST localhost:8000/api/collections/1/ask/ \
  -H "Authorization: Token $TOKEN" -H "Content-Type: application/json" \
  -d '{"question": "How many days of annual leave do employees get?"}'
```

```json
{
  "answer": "Full-time employees get 21 working days of paid annual leave per year [1].",
  "grounded": true,
  "citations": [{"chunk_id": 4, "document_id": 2, "document_title": "leave-policy", "cited_text": "Every full-time employee is entitled to 21 working days of paid annual leave per calendar year."}],
  "sources": [{"chunk_id": 4, "document_id": 2, "document_title": "leave-policy", "similarity": 0.61, "text": "..."}],
  "provider": "claude", "model": "claude-opus-5-5", "input_tokens": 1834, "output_tokens": 41, "latency_ms": 2210
}
```

(An illustration of the response shape; the values are not from a recorded run.)

There is also a small browser interface at `/` (Django template plus plain ES modules, no build step) that
calls the same API: create collections, upload files, ask questions, and see which passages were cited.

## Run it locally

Requirements: Python 3.12 or later and Node.js 18 or later. No PostgreSQL install is needed for development.

```bash
python -m venv .venv
.venv/Scripts/activate            # macOS/Linux: source .venv/bin/activate
pip install -r requirements-dev.txt

cd devdb && npm ci && cd ..
node devdb/server.mjs             # PostgreSQL 18 + pgvector in WebAssembly, port 5499 (leave running)

python manage.py migrate
python manage.py seed_demo        # demo user + fictional sample handbook; prints the password
python manage.py runserver        # http://localhost:8000
```

`devdb/server.mjs` runs [PGlite](https://pglite.dev) (PostgreSQL compiled to WebAssembly) with the pgvector
extension and serves it over the normal PostgreSQL protocol, so Django connects to it like any server. It keeps
data in memory unless `PGLITE_DATA` is set, and handles one connection at a time, which is fine for development
and tests. For anything else, point `DATABASE_URL` at a real PostgreSQL server with pgvector installed:

```bash
export DATABASE_URL=postgres://user:password@host:5432/dbname
```

To use the real models, set `ANTHROPIC_API_KEY` (answers) and `VOYAGE_API_KEY` (embeddings). All settings are
listed in [.env.example](.env.example). Documents keep the embedding model they were indexed with, so re-upload
them after switching embedding providers.

## Tests

```bash
pytest
```

71 tests: chunking offsets and overlap, text and PDF extraction, both embedders (Voyage through a mocked HTTP
transport), uploads and deduplication, size and type limits, cross-user isolation on every endpoint, pgvector
ranking and thresholds against a real PostgreSQL, the not-found short circuit, rate limiting, the Claude request
shape, citation mapping, refusals and API errors (using the real SDK response types with a fake client), the web
page and the demo seed.

Tests run inside the configured database (each in a rolled-back transaction) and refuse to run against a
non-local host. CI runs lint, format check, Django system checks, a missing-migrations check and the test suite
against the official `pgvector/pgvector:pg17` image on every push.

The test suite never calls the Claude or Voyage APIs. The request and response handling is tested against the
SDK's own types, but a live run with real keys is not part of CI.

## Limitations

- Uploads are extracted, chunked and embedded inside the request (up to 5 MB). Larger volumes need a background
  worker such as Celery or Django-Q.
- Scanned PDFs without a text layer are stored with status `empty`; there is no OCR.
- The offline hashing embedder is lexical. Semantic matching ("time off" for "annual leave") needs Voyage.
- `RAG_MIN_SIMILARITY` needs tuning for each embedding model.
- Collections have a single owner; there is no sharing or team model yet.
- Answers are returned whole. Streaming them over Server-Sent Events or WebSockets is a natural next step.

## Project layout

```
config/                 settings (environment-driven), URLs, WSGI
knowledge/
  models.py             Collection, Document, Chunk (pgvector + HNSW), Question
  services/             extraction, chunking, embeddings, retrieval, answering, ingestion, qa
  views.py, urls.py     DRF viewsets and routes
  templates/, static/   browser interface
  management/commands/  seed_demo
tests/                  pytest suite
devdb/                  PGlite + pgvector development server
samples/                fictional company policies for the demo
```

## Authorship

Written by Joshua Isoe, with Claude Code (Anthropic's AI coding assistant) as a pair programmer. Commits record
this in a `Co-Authored-By` trailer.
