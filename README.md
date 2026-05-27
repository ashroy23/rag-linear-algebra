# Linear Algebra RAG

A FastAPI service that answers questions about linear algebra by retrieving relevant passages from Gilbert Strang's *Linear Algebra and Its Applications* and grounding a Gemini LLM response in them.

## Architecture

```
PDF ──► PyPDFLoader ──► filter empty pages ──► RecursiveCharacterTextSplitter
                                                     │
                                                     ▼
                              HuggingFace all-MiniLM-L6-v2 embeddings
                                                     │
                                                     ▼
                                          FAISS (saved to ./index)
                                                     │
        question ─► similarity_search(top-k) ────────┘
                                │
                                ▼
                  Gemini gemini-2.5-flash-lite ──► grounded answer + sources
```

The FAISS index is built on first startup from the PDF and persisted to `./index`. Subsequent starts load the cached index in under a second.

## Source PDF

The PDF itself is **not** checked into the repo (binaries don't belong in git). Drop your own copy of Strang's *Linear Algebra and Its Applications* (or any other linear algebra PDF) into the project root before building the image. By default the app looks for `Gilbert_Strang_Linear_Algebra_and_Its_Applicatio_230928_225121.pdf`; override with the `PDF_PATH` env var if you use a different filename.

## Quick start (Docker)

```bash
docker build -t la-rag .

docker run --rm -p 8000:8000 \
  -e GEMINI_API_KEY=$GEMINI_API_KEY \
  -v $(pwd)/index:/app/index \
  la-rag
```

First start chunks the PDF and builds the index (~30-60s). The mounted `./index` volume keeps it around for next time.

```bash
curl -s localhost:8000/health
# {"status":"ok"}

curl -s -X POST localhost:8000/query \
  -H 'Content-Type: application/json' \
  -d '{"question":"What is an eigenvalue?"}' | jq
```

## API

### `GET /health`
Returns `{"status": "ok"}`.

### `POST /query`
Request:
```json
{ "question": "What is an eigenvalue?", "k": 3 }
```
`k` is optional (default 3, max 20).

Response:
```json
{
  "answer": "An eigenvalue is …",
  "sources": [
    { "source": "Gilbert_Strang_…pdf", "page": 260, "snippet": "…" }
  ]
}
```

### Auto-docs
FastAPI's interactive docs are served at `http://localhost:8000/docs`.

## Local development

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

export GEMINI_API_KEY=...
uvicorn app.main:app --reload --port 8000
```

## Tests

```bash
pip install -r requirements.txt
pytest -q
```

Tests use a fake hash-based embedding and a fake LLM, so they don't download models or call Gemini — safe for CI.

## Configuration

All settings are env vars (`pydantic-settings`, also reads `.env`):

| Variable          | Default                                                       | Description                                    |
|-------------------|---------------------------------------------------------------|------------------------------------------------|
| `GEMINI_API_KEY`  | _(required)_                                                  | Google AI Studio API key                       |
| `PDF_PATH`        | `Gilbert_Strang_Linear_Algebra_and_Its_Applicatio_230928_225121.pdf` | Source PDF                                     |
| `INDEX_DIR`       | `index`                                                       | Where the FAISS index is saved/loaded          |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2`                                            | HuggingFace sentence-transformer model         |
| `LLM_MODEL`       | `gemini-2.5-flash-lite`                                       | Gemini model id                                |
| `CHUNK_SIZE`      | `1000`                                                        | Characters per chunk                           |
| `CHUNK_OVERLAP`   | `200`                                                         | Characters of overlap between chunks           |
| `RETRIEVAL_K`     | `3`                                                           | Default top-k chunks per query                 |

## Project layout

```
app/
  config.py    pydantic-settings env loader
  ingest.py    PDF → chunks → FAISS (build/load with disk cache)
  rag.py       RagService: similarity_search + Gemini answer
  main.py      FastAPI app, lifespan startup, /health, /query
tests/         pytest suite (no network, no API key needed)
Dockerfile     python:3.11-slim, uvicorn on :8000
requirements.txt
```
