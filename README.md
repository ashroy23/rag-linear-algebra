# Linear Algebra RAG (multi-textbook)

A FastAPI service that answers questions grounded in **one or more textbooks you drop into a directory**. Originally built for Gilbert Strang's *Linear Algebra and Its Applications*, but the pipeline is generic — any collection of PDFs works.

## Architecture

```
pdfs/*.pdf ──► PyPDFLoader ──► filter empty pages ──► RecursiveCharacterTextSplitter
                                                              │
                                                              ▼
                                  HuggingFace all-MiniLM-L6-v2 embeddings
                                                              │
                                                              ▼
                                            FAISS (saved to ./index)
                                                              │
            question ──► similarity_search(top-k) ────────────┘
                                    │
                                    ▼
                       Gemini gemini-2.5-flash-lite ──► grounded answer + sources
```

All PDFs in `pdfs/` (recursively) are loaded, chunked, and embedded into a single FAISS index. The index is cached to `./index/` along with a `manifest.json` recording each file's name, size, and mtime — add, remove, or modify a PDF and the next startup automatically rebuilds.

## Drop in your textbooks

```bash
mkdir -p pdfs
cp /path/to/your/book1.pdf pdfs/
cp /path/to/your/book2.pdf pdfs/
# … as many as you like, including subdirectories
```

PDFs are gitignored — they stay local to your checkout.

## Quick start (Docker)

```bash
docker build -t la-rag .

docker run --rm -p 8000:8000 \
  -e GEMINI_API_KEY=$GEMINI_API_KEY \
  -v $(pwd)/pdfs:/app/pdfs \
  -v $(pwd)/index:/app/index \
  la-rag
```

Mounting `pdfs/` means you can swap textbooks without rebuilding the image. Mounting `index/` persists the FAISS cache across container restarts (first start chunks + embeds, ~30-60s; subsequent starts load in under a second).

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
    { "source": "pdfs/strang.pdf", "page": 260, "snippet": "…" }
  ]
}
```

### Auto-docs
FastAPI's interactive docs are at `http://localhost:8000/docs`.

## Local development

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

mkdir -p pdfs && cp /path/to/textbook.pdf pdfs/

export GEMINI_API_KEY=...
uvicorn app.main:app --reload --port 8000
```

## Tests

```bash
pip install -r requirements.txt
pytest -q
```

Tests use a fake hash-based embedding and a fake LLM — no model downloads, no API key required, safe for CI.

## Configuration

All settings are env vars (`pydantic-settings`, also reads `.env`):

| Variable          | Default                | Description                                                |
|-------------------|------------------------|------------------------------------------------------------|
| `GEMINI_API_KEY`  | _(required)_           | Google AI Studio API key                                   |
| `PDF_DIR`         | `pdfs`                 | Directory scanned recursively for `*.pdf` files            |
| `INDEX_DIR`       | `index`                | Where the FAISS index and manifest are saved/loaded        |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2`     | HuggingFace sentence-transformer model                     |
| `LLM_MODEL`       | `gemini-2.5-flash-lite`| Gemini model id                                            |
| `CHUNK_SIZE`      | `1000`                 | Characters per chunk                                       |
| `CHUNK_OVERLAP`   | `200`                  | Characters of overlap between chunks                       |
| `RETRIEVAL_K`     | `3`                    | Default top-k chunks per query                             |

## Project layout

```
app/
  config.py    pydantic-settings env loader
  ingest.py    scan pdfs/, chunk, build/load FAISS with manifest cache
  rag.py       RagService: similarity_search + Gemini answer
  main.py      FastAPI app, lifespan startup, /health, /query
pdfs/          drop your *.pdf files here (gitignored)
tests/         pytest suite (no network, no API key needed)
Dockerfile     python:3.11-slim, uvicorn on :8000
requirements.txt
```
