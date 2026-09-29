---
title: Zepto Support
emoji: 🛒
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
---

# Zepto Data & AI Platform — Module 3

**Support Assistant (RAG + LangGraph + FastAPI)**


This module implements a retrieval-augmented generation (RAG) support assistant that answers Zepto policy questions.

The assistant is built with:

- **FastAPI** — HTTP service exposing `POST /ask`
- **LangGraph** — state machine that routes each question before answering
- **ChromaDB** — local persistent vector store
- **sentence-transformers** — `all-MiniLM-L6-v2` embeddings
- **Pydantic** — strict validation of both requests and responses

The corpus is a set of eight hand-written Zepto policy documents covering delivery, returns and refunds, membership, order tracking, cancellation, damaged items, gift cards and support hours.

---

## Project Structure

```
support_assistant/
│
├── docs/
│   ├── doc_01.txt      Delivery time, fees and serviceability
│   ├── doc_02.txt      Return windows and refund timelines
│   ├── doc_03.txt      Account tiers (Basic / Pass / Pass+)
│   ├── doc_04.txt      Order tracking and stalled orders
│   ├── doc_05.txt      Order cancellation
│   ├── doc_06.txt      Damaged, spoiled or missing items
│   ├── doc_07.txt      Gift cards
│   └── doc_08.txt      Support channels and hours
│
├── main.py              The complete application
├── requirements.txt     Python dependencies
├── Dockerfile          Container build
└── README.md           This document
```

> `chroma_db/` is created automatically on first startup. It holds the embedded
> corpus and is a generated artifact, so it is not tracked in version control.

---

## Setup and Run — Without Docker

### 1\. Create and activate a virtual environment

From the project root:

```
python -m venv venv
```

Windows:

```
venv\Scripts\activate
```

Linux/macOS:

```
source venv/bin/activate
```

### 2\. Install dependencies

```
pip install -r support_assistant/requirements.txt
```

This installs:

```
fastapi
uvicorn
pydantic
sentence-transformers
chromadb
langgraph
```

### 3\. Run the server

Local `uvicorn` runs default to **port 8000**. (The Docker container uses
**7860**, and the optional Gradio client uses **7861** — see the port table
below.)

| Surface | Port | Notes |
|---|---|---|
| FastAPI backend, local `uvicorn` | **8000** | default for local commands and transcripts below |
| FastAPI backend, Docker container | **7860** | also the HF Spaces `app_port` |
| Gradio web UI (`chat_ui.py`) | **7861** | optional local testing client |

From inside the `support_assistant` directory:

```
uvicorn main:app --host 0.0.0.0 --port 8000
```

Or from the project root:

```
uvicorn main:app --host 0.0.0.0 --port 8000 --app-dir support_assistant
```

The first startup downloads the `all-MiniLM-L6-v2` model, reads the eight
documents, chunks and embeds them, and stores the vectors in `chroma_db/`.
This takes roughly 30–60 seconds on a cold cache. Later startups reuse the
existing collection and start much faster.

The service is then available at:

```
http://localhost:8000
```

Interactive API documentation is served by FastAPI at:

```
http://localhost:8000/docs
```

#### Health check

```
curl http://localhost:8000/health
```

```json
{"status": "ok", "mock_llm": true, "collection": "zepto_support_policies", "chunks": 15}
```

#### 3a\. Optional: Gradio web UI

The repository root ships a small Gradio client, `chat_ui.py`, for interactive
manual testing against a locally running backend. It is a convenience wrapper
only — it is not part of the module and is not required for grading.

Start the backend on port `8000` as above, then in a second terminal from the
project root:

```
python chat_ui.py
```

Gradio serves on <http://127.0.0.1:7861>. `chat_ui.py` reads `API_URL` and
`UI_PORT` from the environment and parses the backend port out of `API_URL`, so
the connection error messages and startup preflight follow the backend
automatically. To point it at a Docker container instead, start the container
and set the backend URL:

```
docker run -p 7860:7860 zepto-support-assistant
$env:API_URL="http://127.0.0.1:7860"   # PowerShell
python chat_ui.py
```

---

## Setup and Run — With Docker

### 1\. Build the image

From inside the `support_assistant` directory:

```
docker build -t zepto-support-assistant .
```

### 2\. Run the container

```
docker run -p 7860:7860 zepto-support-assistant
```

The `Dockerfile` sets `MOCK_LLM=1` by default, so the container runs the
deterministic offline baseline with no API key required.

### 3\. Verify

Against the Docker container on port **7860**:

```
curl http://localhost:7860/health
```

(For a local `uvicorn` process on port **8000**, use
`curl http://localhost:8000/health` instead.)

#### Running with the real LLM in Docker

```
docker run -p 7860:7860 -e MOCK_LLM=0 -e GROQ_API_KEY=your_key_here zepto-support-assistant
```

---

## API

### `POST /ask`

**Request**

```json
{"query": "What is the delivery fee for orders under INR 149?"}
```

**Response**

| Field | Type | Description |
| --- | --- | --- |
| `answer` | string | The grounded answer |
| `sources` | list of strings | Source filenames supporting the answer |
| `confidence` | float | Confidence score between `0.0` and `1.0` |

The same Pydantic model (`GroundedAnswer`) validates the LLM's structured
output and serialises the HTTP response, so any `200` response is guaranteed to
match this schema. Both models are strict (`extra="forbid"`), and a request
containing unknown fields is rejected with `422`.

---

## Example Requests and Responses

Both examples below were captured live against the local backend on port `8000`
with `MOCK_LLM` left at its default value (unset, which is equivalent to
`MOCK_LLM=1`). The JSON is byte-exact, including the truncated 200-character
snippet and the fixed `confidence` of `1.0`. Swap the port to `7860` if you are
calling the Docker container instead of a local `uvicorn` process.

### Example 1 — Policy question

This question contains `delivery`, so the router classifies it as
`policy_question` and the graph retrieves from ChromaDB before answering.

```
curl -X POST http://localhost:8000/ask -H "Content-Type: application/json" -d "{\"query\": \"What is the delivery fee for orders under INR 149?\"}"
```

Raw response:

```json
{"answer":"Based on the retrieved context: Zepto delivers grocery and household essentials to serviceable pin codes within 10 to 30 minutes of order confirmation, depending on the customer's delivery zone and current order volume. Standard del","sources":["doc_01.txt","doc_03.txt"],"confidence":1.0}
```

`doc_01.txt` is the correct source, because the delivery threshold is defined
there. The mock answer is deliberately raw — it is the leading ~200 characters
of the top-ranked chunk, not a fluent sentence, so it stops mid-word at
`Standard del`. Sources are listed in retrieval-rank order. `confidence` is the
fixed `MOCK_CONFIDENCE = 1.0`, which is the expected baseline output in mock
mode.

### Example 2 — General question

This question contains none of the routing keywords, so the router classifies
it as `general_question`. The graph takes the `direct_answer` branch, skips
retrieval entirely and returns the fixed refusal.

```
curl -X POST http://localhost:8000/ask -H "Content-Type: application/json" -d "{\"query\": \"Can you help me write a python script?\"}"
```

Raw response:

```json
{"answer":"I can only answer questions about Zepto policies right now.","sources":[],"confidence":1.0}
```

#### PowerShell note

In Windows PowerShell, `curl` is an alias for `Invoke-WebRequest` and nested
quotes behave differently. Either use `curl.exe` with a payload file:

```powershell
'{"query": "What is the delivery fee for orders under INR 149?"}' | Set-Content -Encoding utf8 payload.json
curl.exe -s -X POST http://localhost:8000/ask -H "Content-Type: application/json" --data-binary "@payload.json"
```

or use `Invoke-RestMethod`:

```powershell
Invoke-RestMethod -Uri http://localhost:8000/ask -Method Post -ContentType "application/json" -Body '{"query": "What is the delivery fee for orders under INR 149?"}'
```

---

## Architecture

### Data flow

```
docs/*.txt
    │
    │  load_documents()            stage 1 — ingestion
    ▼
List[{"source", "text"}]
    │
    │  split_sentences() + chunk_text()
    ▼
List[{id, source, chunk_index, text}]     8 documents → 15 chunks
    │
    │  create_embedding_model()            stage 2 — embedding
    │  embed_texts()   (all-MiniLM-L6-v2, L2-normalised)
    ▼
ChromaDB persistent collection "zepto_support_policies"   (cosine space)
    │
    │  retrieve()  top-3, nearest first      stage 3 — retrieval
    ▼
List[{source, text, distance}]
    │
    │  build_context() → ANSWER_PROMPT       stage 4 — generation
    ▼
GroundedAnswer { answer, sources, confidence }
    │
    ▼
POST /ask  →  HTTP 200 JSON
```

### Which component handles which stage

| Stage | Function | Runs |
| --- | --- | --- |
| Ingestion | `load_documents()` | Once, at startup |
| Chunking | `split_sentences()`, `chunk_text()`, `build_chunks()` | Once, at startup |
| Embedding | `create_embedding_model()`, `embed_texts()` | Once, at startup |
| Storage | `get_chroma_client()`, `open_collection()`, `ingest_documents()` | Once, at startup |
| Retrieval | `retrieve()` | Per request, policy branch only |
| Intent routing | `classify_intent()` node | Per request |
| Generation | `generate_mock_answer()` / `generate_llm_answer()` | Per request, policy branch only |
| Refusal | `direct_answer()` node | Per request, general branch only |
| Validation | `GroundedAnswer`, `AskRequest` | Per request |
| Transport | `ask()` endpoint | Per request |

### Ingestion and embedding

`initialize()` runs once from the FastAPI `lifespan` hook, so the corpus is
embedded exactly one time per process and never on the request path.

`load_documents()` reads every `docs/*.txt` as UTF-8. The UTF-8 encoding is
required because `doc_02.txt` contains an en dash in "3–5 business days".

`chunk_text()` splits on sentence boundaries using a lookbehind on `.`, `!` and
`?`, then packs sentences into chunks of at most 420 characters with 60
characters of overlap. Splitting on sentences rather than at a fixed offset
keeps policies intact — the INR 1000 damaged-item threshold in `doc_06.txt` is
never cut away from the sentence that explains it. Any single sentence longer
than the chunk size is hard-split so that no policy text is silently dropped.

The current corpus yields **15 chunks** across the 8 documents.

`embed_texts()` encodes chunks with `all-MiniLM-L6-v2` using
`normalize_embeddings=True`, so the stored vectors are unit length and cosine
similarity is computed directly. Embeddings are passed to ChromaDB explicitly
rather than relying on a registered embedding function, which keeps the vector
space under the application's control.

### ChromaDB initialization

ChromaDB is the component most likely to fail silently or loudly on the first
run, so initialisation is handled defensively:

- The store is a `PersistentClient` rooted at `support_assistant/chroma_db/`,
  created with `anonymized_telemetry=False`.
- The collection uses `metadata={"hnsw:space": "cosine"}` so that the distance
  returned by a query is `1 - cosine_similarity`.
- `open_collection()` calls `get_or_create_collection()`. ChromaDB raises if a
  collection already exists with *different* metadata, so any failure falls
  back to a delete-and-recreate. This makes a stale or corrupted store
  self-healing instead of a startup crash.
- Ingestion is **idempotent and content-addressed**. A SHA-256 fingerprint of
  every chunk's id and text is stored in the collection metadata. On startup the
  fingerprint is compared against the stored one; if they differ, or if the
  chunk count changed, the collection is dropped and rebuilt. An unchanged
  corpus is reused, which is why only the first startup is slow.

The startup log reports which path was taken:

```
[ingest] 8 documents -> 15 chunks in collection 'zepto_support_policies' (reused=False)
[ingest] 8 documents -> 15 chunks in collection 'zepto_support_policies' (reused=True)
```

### Retrieval

`retrieve()` embeds the question with the same model used at ingestion and
queries ChromaDB for the top 3 chunks. The query embeddings are L2-normalised
in exactly the same way as the stored ones, so the cosine distances are
comparable.

`score_to_confidence()` converts the top hit's distance into confidence as
`1 - distance`, clamped to `[0, 1]` and rounded to three decimals. Because the
model, the corpus and the query are all fixed, this value is **deterministic**:
the same question always yields the same confidence.

This helper is used by the **real LLM path (`MOCK_LLM=0`)**. The **mock
baseline** does not use it: it reports the fixed `MOCK_CONFIDENCE = 1.0`, so an
offline graded run always emits `confidence: 1.0` regardless of how close the
top chunk happens to be. Both branches are reproducible — the mock branch
simply has a single possible value.

`dedupe_sources()` returns unique filenames in retrieval rank order, so
`sources` shows which policies are most relevant first.

### Generation

In mock mode, `generate_mock_answer()` returns:

```python
f"Based on the retrieved context: {top_snippet}"
```

where `top_snippet` is the first 200 characters of the highest-ranked chunk with
whitespace collapsed. This is intentionally unsophisticated — it proves the
retrieval stage is returning the right policy text without involving a model.

In real mode, `generate_llm_answer()` renders the structured `ANSWER_PROMPT`
with the numbered context block and the question, and validates the completion
into `GroundedAnswer`.

#### The prompt

`ANSWER_PROMPT` is assembled from six labelled sections:

| Section | Purpose |
| --- | --- |
| `ROLE` | Establishes the assistant as Zepto policy support |
| `CONTEXT` | The retrieved chunks, numbered with their source filename |
| `TASK` | Answer the question using the context |
| `FORMAT` | A single JSON object with `answer`, `sources`, `confidence` |
| `LENGTH` | 2–4 short sentences, no preamble or sign-off |
| `CONSTRAINT` | **Do not answer using information not present in the provided context** |
| `EXAMPLE` | One worked few-shot example with context, question and JSON output |

The `CONSTRAINT` line is the negative instruction: the model is explicitly
forbidden from falling back on pretrained knowledge of Zepto, and is
instructed to answer from the retrieved context alone.

### The LangGraph state machine

State is a `TypedDict` (`SupportState`) carrying `query`, `intent`, `route`,
`hits`, `answer`, `sources` and `confidence`. Each node returns a partial dict,
which LangGraph merges into the accumulated state.

```
                        ┌──────────────────┐
                 START ─▶  classify_intent │
                        └────────┬─────────┘
                                 │
                    conditional edge on intent
                    ┌────────────────┴────────────────┐
          policy_question                    general_question
                    │                                 │
                    ▼                                 ▼
        ┌───────────────────────┐           ┌────────────────┐
        │  retrieve_and_answer  │           │  direct_answer │
        └───────────┬───────────┘           └───────┬────────┘
                    │                               │
                    └───────────────┬───────────────┘
                                    ▼
                                   END
```

- `classify_intent()` — sets `intent` and `route`.
- `route_after_classification()` — the conditional edge, returning
  `"retrieve_and_answer"` or `"direct_answer"`.
- `retrieve_and_answer()` — always runs retrieval, then generates.
- `direct_answer()` — returns the fixed refusal with `sources=[]` and
  `confidence=1.0` (the fixed `MOCK_CONFIDENCE`), and performs **no** retrieval.

`retrieve_and_answer()` retrieving in *both* modes is deliberate: the graded
requirement is that the retrieval stage always executes for a policy question,
and keeping retrieval unconditional means the mock baseline genuinely
exercises the vector store rather than stubbing it out.

### How the MOCK_LLM toggle changes behaviour

`MOCK_LLM` is read once at import time as `os.getenv("MOCK_LLM", "1") != "0"`.
Unset or `"1"` selects the mock baseline; only `"0"` selects the real LLM path.
Any other value is treated as mock.

#### Node `classify_intent`

| Mode | Behaviour |
| --- | --- |
| `MOCK_LLM` unset / `1` | `classify_by_keyword()` — pure substring match against `POLICY_KEYWORDS`. **No LLM call, no network.** |
| `MOCK_LLM=0` | `classify_with_llm()` — calls the LLM with `INTENT_PROMPT` and validates into the `IntentDecision` schema. |

The keyword list is fixed and documented in the source:

```
delivery, return, refund, membership,
tracking, cancel, gift card, support hours
```

A query containing any of these routes to `policy_question`; everything else
routes to `general_question`. Matching is case-insensitive.

`classify_with_llm()` falls back to the keyword router if the API is
unreachable or its output fails validation, so a routing failure degrades
instead of returning a 500.

#### Node `retrieve_and_answer`

| Mode | Behaviour |
| --- | --- |
| `MOCK_LLM` unset / `1` | Retrieval runs. Generation is `generate_mock_answer()`: a fixed prefix plus the top chunk's first ~200 characters, with the fixed `confidence=1.0`. Deterministic, offline. |
| `MOCK_LLM=0` | Retrieval runs identically. Generation is `generate_llm_answer()`: `ANSWER_PROMPT` → LLM → `GroundedAnswer` with the 2-retry corrective loop, and `confidence` from `score_to_confidence()`. |

Retrieval is identical in both modes. The toggle only swaps the generation
step.

#### Node `direct_answer`

| Mode | Behaviour |
| --- | --- |
| `MOCK_LLM` unset / `1` | Fixed string: `"I can only answer questions about Zepto policies right now."` |
| `MOCK_LLM=0` | **Identical.** |

This node makes no LLM call in either mode. There is nothing to generate, since
the answer is a fixed refusal — a model call would add latency, cost and a
failure mode for a constant string. Both modes return `sources=[]` and
`confidence=1.0`.

#### The 2-retry corrective loop

`generate_structured()` enforces the output schema with up to
`LLM_CORRECTIVE_RETRIES = 2` corrective attempts, so at most 3 API calls:

1. **Attempt 0** — send the prompt, parse the completion with
   `extract_json_object()`, validate with `GroundedAnswer.model_validate_json()`.
2. **Attempt 1 (corrective)** — append the *exact* Pydantic validation error
   plus the model-generated JSON schema to the conversation and retry.
3. **Attempt 2 (corrective)** — same, with the latest error.
4. If all three fail, raise a `RuntimeError` naming the schema and the number
   of retries; the endpoint returns `500` rather than leaking unvalidated text.

`extract_json_object()` strips ` ```json ` fences and any surrounding prose
before locating the outermost `{` ... `}`, so a chatty completion is still
recoverable. `ValidationError` subclasses `ValueError` in Pydantic v2, so
schema failures and parse failures are both caught by the same handler.

Transport failures are **not** retried — an `HTTPError` or `URLError` raises
immediately, because retrying a network fault three times only adds latency.

---

## Configuration

| Variable | Default | Purpose |
| --- | --- | --- |
| `MOCK_LLM` | `1` | `1`/unset = mock baseline, `0` = real LLM |
| `GROQ_API_KEY` | `""` | API key for the real LLM path |
| `GROQ_MODEL` | `llama-3.3-70b-versatile` | Model used when `MOCK_LLM=0` |

Other tunables are module-level constants in `main.py`: `CHUNK_SIZE`,
`CHUNK_OVERLAP`, `TOP_K`, `SNIPPET_CHARS`, `EMBEDDING_MODEL_NAME` and
`COLLECTION_NAME`.

---

## Expected Startup Output

```
INFO:     Started server process
[ingest] 8 documents -> 15 chunks in collection 'zepto_support_policies' (reused=False)
[startup] ready (MOCK_LLM=1)
INFO:     Uvicorn running on http://0.0.0.0:8000
```

Each request logs its routing decision:

```
[classify_intent] 'What is the delivery fee for orders under INR 149?' -> policy_question
```

---

## Technologies Used

- **Python**
- **FastAPI** — HTTP API
- **Uvicorn** — ASGI server
- **Pydantic** — strict request, response and LLM-output validation
- **LangGraph** — stateful routing graph
- **ChromaDB** — local persistent vector store
- **sentence-transformers** — `all-MiniLM-L6-v2` embeddings
- **Docker** — containerization

---

## Troubleshooting

### `GROQ_API_KEY is not set`

Expected when `MOCK_LLM=0` without a key. Either export a key or leave
`MOCK_LLM` at its default of `1`.

### `LLM output failed GroundedAnswer validation after 2 corrective retries`

The model produced unusable output three times in a row. Lower
`LLM_TEMPERATURE` (already `0.0`) or try a different `GROQ_MODEL`.

### ChromaDB raises on `get_or_create_collection`

`open_collection()` catches this and recreates the collection. If startup still
fails, delete `chroma_db/` and restart — it is regenerated from `docs/`.

### `chroma_db` is locked or corrupt

Another process is holding the store, or the index is damaged. Stop all
instances and delete the `chroma_db/` directory, then restart.

### First startup is slow

Expected. The `all-MiniLM-L6-v2` model is downloaded on first use and the
corpus is embedded. Subsequent startups print `reused=True` and are fast.

### Port 8000 already in use

Pick a free local port and use it consistently in the server command, the curl
examples and the Gradio client:

```
uvicorn main:app --host 0.0.0.0 --port 8080
$env:API_URL="http://127.0.0.1:8080"   # if using chat_ui.py
```

The Docker container port (`7860`) and the Gradio port (`7861`) are unaffected.

---

## Conclusion

This module demonstrates a complete retrieval-augmented generation workflow:

```
Policy Documents
      ↓
Ingestion and Chunking
      ↓
Embedding
      ↓
Vector Storage
      ↓
Intent Routing
      ↓
Retrieval
      ↓
Grounded Generation
      ↓
Validated API Response
```

It combines deterministic offline evaluation (`MOCK_LLM=1`) with a production
LLM path (`MOCK_LLM=0`) behind one toggle, while guaranteeing through Pydantic
that both modes emit the same strictly-typed response schema.
