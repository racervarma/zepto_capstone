"""
Zepto Capstone - Module 3
Support Assistant (RAG + LangGraph + FastAPI)

This module implements a retrieval-augmented support assistant for Zepto
policy questions.

The pipeline:

1. INGESTION   - reads the plain-text policy documents from ./docs.
2. CHUNKING    - splits every document into overlapping sentence-aware chunks.
3. EMBEDDING   - encodes chunks with sentence-transformers (all-MiniLM-L6-v2).
4. STORAGE     - persists vectors in a local ChromaDB collection.
5. RETRIEVAL   - fetches the top-3 chunks for an incoming question.
6. GENERATION  - renders a grounded answer from the retrieved context.

Stages 1-4 run once at application startup. Stages 5-6 run per request inside
a LangGraph StateGraph with three nodes:

    START -> classify_intent -> [conditional edge]
                               |-> retrieve_and_answer -> END
                               |-> direct_answer       -> END

MOCK_LLM
--------
The environment variable ``MOCK_LLM`` selects between two code paths that are
wired into every node of the graph:

    MOCK_LLM unset or "1"  -> deterministic graded mock baseline (no network)
    MOCK_LLM=0            -> real LLM (Groq) with a 2-retry corrective loop
                             that repairs output failing Pydantic validation

The API contract is identical in both modes:
    POST /ask {"query": str} -> {"answer": str, "sources": [str], "confidence": float}
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import urllib.error
import urllib.request
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Sequence, TypedDict

import chromadb
from chromadb.config import Settings as ChromaSettings
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, field_validator
from sentence_transformers import SentenceTransformer

from langgraph.graph import END, START, StateGraph


# ============================================================
# Configuration
# ============================================================

SUPPORT_ASSISTANT_DIR = Path(__file__).resolve().parent
DOCS_DIR = SUPPORT_ASSISTANT_DIR / "docs"
CHROMA_DIR = SUPPORT_ASSISTANT_DIR / "chroma_db"

COLLECTION_NAME = "zepto_support_policies"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

CHUNK_SIZE = 420
CHUNK_OVERLAP = 60

TOP_K = 3
SNIPPET_CHARS = 200

MOCK_LLM = os.getenv("MOCK_LLM", "1").strip() != "0"

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
GROQ_ENDPOINT = "https://api.groq.com/openai/v1/chat/completions"
LLM_TIMEOUT_SECONDS = 45
LLM_TEMPERATURE = 0.0
LLM_CORRECTIVE_RETRIES = 2

Intent = Literal["policy_question", "general_question"]


# ============================================================
# Pydantic schemas (API contract + structured LLM output target)
# ============================================================

class AskRequest(BaseModel):
    """Incoming request body for POST /ask."""

    model_config = ConfigDict(extra="forbid")

    query: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="The end-user question about Zepto policies.",
    )

    @field_validator("query")
    @classmethod
    def query_must_not_be_blank(cls, value: str) -> str:
        cleaned = value.strip()

        if not cleaned:
            raise ValueError("query must not be blank")

        return cleaned


class GroundedAnswer(BaseModel):
    """
    Strict output schema shared by the LLM and the HTTP response.

    The same model validates structured LLM output and serialises the API
    response, so a successful graph run is guaranteed to satisfy it.
    """

    model_config = ConfigDict(extra="forbid")

    answer: str = Field(..., min_length=1, description="The grounded answer.")
    sources: List[str] = Field(
        default_factory=list,
        description="Document filenames that support the answer.",
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Confidence in the answer, between 0 and 1.",
    )


class AskResponse(GroundedAnswer):
    """Response body for POST /ask (identical shape to GroundedAnswer)."""


class IntentDecision(BaseModel):
    """Strict output schema used when the real LLM performs classification."""

    model_config = ConfigDict(extra="forbid")

    intent: Intent


# ============================================================
# Prompt templates
# ============================================================

ANSWER_PROMPT = """\
ROLE
You are the Zepto Support Assistant. You answer customer questions about
Zepto policies using only the supplied context.

CONTEXT
{context}

TASK
Answer the customer question below. Every factual claim in your answer must be
supported by the context above.

QUESTION
{query}

FORMAT
Respond with a single JSON object and nothing else:
{{
  "answer": "<the grounded answer, 2-4 short sentences>",
  "sources": ["<source filename>", "..."],
  "confidence": <float between 0 and 1>
}}

LENGTH
Keep the answer between 2 and 4 short sentences. No preamble, no sign-off.

CONSTRAINT
Do not answer using information not present in the provided context.

EXAMPLE
Context: "Zepto delivers within 10 to 30 minutes. Standard delivery is free on
orders over INR 149."
Question: "Is delivery free on a small order?"
Output: {{"answer": "Standard delivery is free on orders over INR 149. Orders
below that threshold incur a flat INR 25 delivery fee.",
"sources": ["doc_01.txt"], "confidence": 0.92}}
"""


INTENT_PROMPT = """\
ROLE
You are the intent router for the Zepto Support Assistant.

TASK
Classify the customer question as exactly one of:
- "policy_question": about Zepto policies such as delivery, returns, refunds,
  membership, order tracking, cancellation, gift cards or support hours.
- "general_question": anything else (chit-chat, capabilities, unrelated topics).

QUESTION
{query}

FORMAT
Respond with a single JSON object and nothing else:
{{"intent": "<policy_question|general_question>"}}

CONSTRAINT
Do not answer using information not present in the provided context.
"""


MOCK_DIRECT_ANSWER = "I can only answer questions about Zepto policies right now."


# ============================================================
# Ingestion: loading and chunking
# ============================================================

def load_documents(docs_dir: Path = DOCS_DIR) -> List[Dict[str, str]]:
    """Read every .txt policy document from the docs directory."""
    if not docs_dir.is_dir():
        raise FileNotFoundError(f"Docs directory not found: {docs_dir}")

    documents: List[Dict[str, str]] = []

    for path in sorted(docs_dir.glob("*.txt")):
        text = path.read_text(encoding="utf-8").strip()

        if not text:
            continue

        documents.append({"source": path.name, "text": text})

    if not documents:
        raise RuntimeError(f"No readable documents found in: {docs_dir}")

    return documents


def split_sentences(text: str) -> List[str]:
    """Split text into sentences so chunks break on natural boundaries."""
    normalized = " ".join(text.split())

    return [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?])\s+", normalized)
        if sentence.strip()
    ]


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE,
               chunk_overlap: int = CHUNK_OVERLAP) -> List[str]:
    """
    Pack sentences into overlapping chunks of at most `chunk_size` characters.

    A sentence longer than `chunk_size` is hard-split so that no single policy
    sentence can be dropped during ingestion.
    """
    chunks: List[str] = []
    current: List[str] = []
    current_length = 0

    def flush() -> None:
        nonlocal current, current_length

        if current:
            chunks.append(" ".join(current))
            current = []
            current_length = 0

    for sentence in split_sentences(text):
        if len(sentence) > chunk_size:
            flush()

            for start in range(0, len(sentence), chunk_size - chunk_overlap):
                piece = sentence[start:start + chunk_size].strip()

                if piece:
                    chunks.append(piece)

            continue

        if current_length + len(sentence) + 1 > chunk_size:
            tail = current[-1] if current else ""
            flush()

            if tail and len(tail) < chunk_size:
                current = [tail]
                current_length = len(tail) + 1

        current.append(sentence)
        current_length += len(sentence) + 1

    flush()

    return chunks


def build_chunks(documents: Sequence[Dict[str, str]]) -> List[Dict[str, Any]]:
    """Turn documents into chunk records carrying source and chunk index."""
    chunks: List[Dict[str, Any]] = []

    for document in documents:
        for index, chunk in enumerate(chunk_text(document["text"])):
            chunks.append({
                "id": f"{Path(document['source']).stem}__{index:02d}",
                "source": document["source"],
                "chunk_index": index,
                "text": chunk,
            })

    if not chunks:
        raise RuntimeError("Chunking produced no chunks from the corpus.")

    return chunks


def corpus_fingerprint(chunks: Sequence[Dict[str, Any]]) -> str:
    """Stable hash of the corpus, used to detect stale ChromaDB collections."""
    digest = hashlib.sha256()

    for chunk in chunks:
        digest.update(chunk["id"].encode("utf-8"))
        digest.update(b"\x00")
        digest.update(chunk["text"].encode("utf-8"))
        digest.update(b"\x00")

    return digest.hexdigest()


# ============================================================
# Embedding + ChromaDB storage
# ============================================================

def create_embedding_model() -> SentenceTransformer:
    """Load the local sentence-transformers embedding model."""
    return SentenceTransformer(EMBEDDING_MODEL_NAME)


def embed_texts(model: SentenceTransformer, texts: Sequence[str]):
    """Embed texts into L2-normalised vectors so cosine similarity is exact."""
    return model.encode(
        list(texts),
        normalize_embeddings=True,
        show_progress_bar=False,
    ).tolist()


def get_chroma_client():
    """Create the persistent local ChromaDB client."""
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)

    return chromadb.PersistentClient(
        path=str(CHROMA_DIR),
        settings=ChromaSettings(anonymized_telemetry=False),
    )


def open_collection(client) -> Any:
    """
    Open the collection, tolerating first run and metadata drift.

    ChromaDB raises if an existing collection was created with different
    metadata, so a failed get_or_create is retried after a clean recreate.
    """
    try:
        return client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
    except Exception:
        try:
            client.delete_collection(name=COLLECTION_NAME)
        except Exception:
            pass

        return client.create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )


def reset_collection(client) -> Any:
    """Delete and recreate the collection with the expected metadata."""
    try:
        client.delete_collection(name=COLLECTION_NAME)
    except Exception:
        pass

    return client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


def collection_is_stale(collection, chunks: Sequence[Dict[str, Any]],
                       fingerprint: str) -> bool:
    """
    Decide whether the on-disk collection must be rebuilt.

    A rebuild is required when the collection is empty, the document count
    changed, or the corpus content hash no longer matches the stored hash.
    """
    if collection.count() != len(chunks):
        return True

    stored = collection.metadata.get("corpus_fingerprint")

    return stored != fingerprint


def ingest_documents(model: SentenceTransformer, client) -> Any:
    """
    Ingestion -> chunking -> embedding -> storage.

    Runs at startup and is idempotent: an unchanged corpus is not re-embedded.
    """
    documents = load_documents()
    chunks = build_chunks(documents)
    fingerprint = corpus_fingerprint(chunks)

    collection = open_collection(client)
    reused = not collection_is_stale(collection, chunks, fingerprint)

    if not reused:
        collection = reset_collection(client)
        collection.add(
            ids=[chunk["id"] for chunk in chunks],
            documents=[chunk["text"] for chunk in chunks],
            embeddings=embed_texts(model, [chunk["text"] for chunk in chunks]),
            metadatas=[
                {
                    "source": chunk["source"],
                    "chunk_index": chunk["chunk_index"],
                }
                for chunk in chunks
            ],
        )
        collection.modify(
            metadata={"corpus_fingerprint": fingerprint},
        )

    print(
        f"[ingest] {len(documents)} documents -> {collection.count()} chunks "
        f"in collection '{COLLECTION_NAME}' (reused={reused})"
    )

    return collection


# ============================================================
# Retrieval
# ============================================================

def retrieve(model: SentenceTransformer, collection, query: str,
             top_k: int = TOP_K) -> List[Dict[str, Any]]:
    """Return the top-k chunks for a query, best match first."""
    if collection.count() == 0:
        raise RuntimeError("ChromaDB collection is empty; ingestion failed.")

    results = collection.query(
        query_embeddings=embed_texts(model, [query]),
        n_results=min(top_k, collection.count()),
        include=["documents", "metadatas", "distances"],
    )

    documents = (results.get("documents") or [[]])[0]
    metadatas = (results.get("metadatas") or [[]])[0]
    distances = (results.get("distances") or [[]])[0]

    hits: List[Dict[str, Any]] = []

    for position, document in enumerate(documents):
        metadata = metadatas[position] if position < len(metadatas) else {}

        hits.append({
            "source": metadata.get("source", "unknown"),
            "chunk_index": metadata.get("chunk_index", position),
            "text": document,
            "distance": float(distances[position]) if position < len(distances) else 1.0,
        })

    return hits


MOCK_CONFIDENCE = 1.0


def score_to_confidence(distance: float) -> float:
    """
    Convert a cosine distance into a bounded confidence score.

    ChromaDB returns distance = 1 - cosine_similarity for the cosine space, so
    confidence is deterministic for a given (corpus, query) pair.

    Used by the real LLM path. The mock baseline reports the fixed
    MOCK_CONFIDENCE instead, so a graded offline run always emits 1.0.
    """
    return round(max(0.0, min(1.0, 1.0 - distance)), 3)


def dedupe_sources(hits: Sequence[Dict[str, Any]]) -> List[str]:
    """Unique source filenames, ordered by retrieval rank."""
    seen: List[str] = []

    for hit in hits:
        if hit["source"] not in seen:
            seen.append(hit["source"])

    return seen


def build_context(hits: Sequence[Dict[str, Any]]) -> str:
    """Render retrieved chunks into the numbered context block for the prompt."""
    return "\n\n".join(
        f"[{index}] (source: {hit['source']}) {hit['text']}"
        for index, hit in enumerate(hits, start=1)
    )


# ============================================================
# Real LLM path (MOCK_LLM=0)
# ============================================================

def call_llm(system_prompt: str, user_prompt: str) -> str:
    """
    Call the Groq chat-completions endpoint.

    The Groq API is OpenAI-compatible, so the request is issued over stdlib
    urllib to keep the dependency list limited to the required packages.
    """
    if not GROQ_API_KEY:
        raise RuntimeError(
            "GROQ_API_KEY is not set. Export it or run with MOCK_LLM=1."
        )

    payload = json.dumps({
        "model": GROQ_MODEL,
        "temperature": LLM_TEMPERATURE,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    }).encode("utf-8")

    request = urllib.request.Request(
        GROQ_ENDPOINT,
        data=payload,
        headers={
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    with urllib.request.urlopen(request, timeout=LLM_TIMEOUT_SECONDS) as response:
        body = json.loads(response.read().decode("utf-8"))

    return body["choices"][0]["message"]["content"]


def extract_json_object(raw_output: str) -> str:
    """
    Pull a JSON object out of a raw completion.

    Models often wrap JSON in a ```json fence or add surrounding prose, so the
    outermost braces are located before parsing.
    """
    text = (raw_output or "").strip()

    fenced = re.search(r"```(?:json)?\s*(.+?)```", text, re.DOTALL)

    if fenced:
        text = fenced.group(1).strip()

    start = text.find("{")
    end = text.rfind("}")

    if start == -1 or end == -1 or end <= start:
        raise ValueError("No JSON object found in the model output.")

    return text[start:end + 1]


def generate_structured(prompt: str, schema: type[BaseModel],
                        user_prompt: str) -> BaseModel:
    """
    Generate output and enforce the schema with a corrective retry loop.

    Attempt 0 is the initial call. Each subsequent attempt appends the exact
    Pydantic validation error to the prompt, steering the model back to a
    schema-valid object. At most LLM_CORRECTIVE_RETRIES corrective attempts are
    made before the error is raised.
    """
    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": user_prompt},
    ]

    last_error: Optional[Exception] = None

    for attempt in range(LLM_CORRECTIVE_RETRIES + 1):
        if attempt > 0:
            messages.append({
                "role": "user",
                "content": (
                    "Your previous response was rejected. "
                    f"Validation error: {last_error}\n"
                    "Return only a valid JSON object matching this schema:\n"
                    f"{json.dumps(schema.model_json_schema())}"
                ),
            })

        payload = json.dumps({
            "model": GROQ_MODEL,
            "temperature": LLM_TEMPERATURE,
            "messages": messages,
        }).encode("utf-8")

        request = urllib.request.Request(
            GROQ_ENDPOINT,
            data=payload,
            headers={
                "Authorization": f"Bearer {GROQ_API_KEY}",
                "Content-Type": "application/json",
            },
            method="POST",
        )

        try:
            with urllib.request.urlopen(
                request, timeout=LLM_TIMEOUT_SECONDS
            ) as response:
                body = json.loads(response.read().decode("utf-8"))

            raw_output = body["choices"][0]["message"]["content"]

            messages.append({"role": "assistant", "content": raw_output})

            return schema.model_validate_json(extract_json_object(raw_output))
        except urllib.error.HTTPError as error:
            raise RuntimeError(
                f"LLM request failed with HTTP {error.code}."
            ) from error
        except urllib.error.URLError as error:
            raise RuntimeError(
                f"LLM request failed: {error.reason}"
            ) from error
        except (ValueError, RuntimeError) as error:
            last_error = error

    raise RuntimeError(
        f"LLM output failed {schema.__name__} validation after "
        f"{LLM_CORRECTIVE_RETRIES} corrective retries: {last_error}"
    )


# ============================================================
# Intent classification
# ============================================================

POLICY_KEYWORDS = (
    "delivery",
    "return",
    "refund",
    "membership",
    "tracking",
    "cancel",
    "gift card",
    "support hours",
)


def classify_by_keyword(query: str) -> Intent:
    """Graded baseline router: keyword match, no LLM call."""
    lowered = query.lower()

    if any(keyword in lowered for keyword in POLICY_KEYWORDS):
        return "policy_question"

    return "general_question"


def classify_with_llm(query: str) -> Intent:
    """
    Real-LLM router used when MOCK_LLM=0.

    Falls back to the keyword router if the API is unreachable or the output
    cannot be validated, so a routing failure degrades instead of erroring.
    """
    try:
        decision = generate_structured(
            prompt=INTENT_PROMPT,
            schema=IntentDecision,
            user_prompt=query,
        )

        return decision.intent
    except Exception as error:
        print(f"[classify_intent] LLM classification failed: {error}")

        return classify_by_keyword(query)


# ============================================================
# Generation strategies (one per MOCK_LLM mode)
# ============================================================

def generate_mock_answer(hits: Sequence[Dict[str, Any]]) -> GroundedAnswer:
    """
    Deterministic mock baseline for MOCK_LLM unset or "1".

    Always prefixes the highest-ranked chunk snippet with a fixed label and
    reports the fixed MOCK_CONFIDENCE of 1.0. No network call, no randomness
    and no dependence on retrieval distance, so the response is fully
    reproducible.
    """
    top_snippet = " ".join(hits[0]["text"].split())[:SNIPPET_CHARS]

    return GroundedAnswer(
        answer=f"Based on the retrieved context: {top_snippet}",
        sources=dedupe_sources(hits),
        confidence=MOCK_CONFIDENCE,
    )


def generate_llm_answer(query: str, hits: Sequence[Dict[str, Any]]) -> GroundedAnswer:
    """Grounded generation through the real LLM, used when MOCK_LLM=0."""
    result = generate_structured(
        prompt=ANSWER_PROMPT,
        schema=GroundedAnswer,
        user_prompt=f"CONTEXT\n{build_context(hits)}\n\nQUESTION\n{query}",
    )

    return result


# ============================================================
# LangGraph state and nodes
# ============================================================

class SupportState(TypedDict, total=False):
    """State shared across the nodes of the support graph."""

    query: str
    intent: Intent
    route: str
    hits: List[Dict[str, Any]]
    answer: str
    sources: List[str]
    confidence: float


def classify_intent(state: SupportState) -> Dict[str, Any]:
    """
    Node 1 - route the question.

    MOCK_LLM unset or "1": pure keyword match, no LLM call.
    MOCK_LLM=0: LLM classification with keyword fallback.
    """
    query = state["query"]
    intent = classify_by_keyword(query) if MOCK_LLM else classify_with_llm(query)

    print(f"[classify_intent] '{query}' -> {intent}")

    return {"intent": intent, "route": intent}


def retrieve_and_answer(state: SupportState) -> Dict[str, Any]:
    """
    Node 2 - retrieve then answer a policy question.

    ChromaDB retrieval always runs, in both modes. Only the generation step
    differs: the mock returns a deterministic snippet, the real path calls the
    LLM with the 2-retry corrective loop.
    """
    query = state["query"]

    model: SentenceTransformer = _EMBEDDING_MODEL
    hits = retrieve(model, _COLLECTION, query, top_k=TOP_K)

    if MOCK_LLM:
        result = generate_mock_answer(hits)
    else:
        result = generate_llm_answer(query, hits)

    return {
        "hits": hits,
        "answer": result.answer,
        "sources": result.sources,
        "confidence": result.confidence,
    }


def direct_answer(state: SupportState) -> Dict[str, Any]:
    """
    Node 3 - handle questions that are out of scope.

    Skips retrieval entirely and returns the fixed refusal string. Confidence
    is the fixed MOCK_CONFIDENCE, because the refusal itself is a certain,
    hardcoded outcome rather than a probabilistic claim about the corpus.
    """
    return {
        "hits": [],
        "answer": MOCK_DIRECT_ANSWER,
        "sources": [],
        "confidence": MOCK_CONFIDENCE,
    }


def route_after_classification(state: SupportState) -> str:
    """Conditional edge: policy questions retrieve, everything else refuses."""
    return (
        "retrieve_and_answer"
        if state.get("intent") == "policy_question"
        else "direct_answer"
    )


def build_graph():
    """Compile the three-node support graph."""
    builder = StateGraph(SupportState)

    builder.add_node("classify_intent", classify_intent)
    builder.add_node("retrieve_and_answer", retrieve_and_answer)
    builder.add_node("direct_answer", direct_answer)

    builder.add_edge(START, "classify_intent")

    builder.add_conditional_edges(
        "classify_intent",
        route_after_classification,
        {
            "retrieve_and_answer": "retrieve_and_answer",
            "direct_answer": "direct_answer",
        },
    )

    builder.add_edge("retrieve_and_answer", END)
    builder.add_edge("direct_answer", END)

    return builder.compile()


# ============================================================
# Application resources
# ============================================================

_EMBEDDING_MODEL: Optional[SentenceTransformer] = None
_COLLECTION: Any = None
_GRAPH: Any = None


def initialize() -> None:
    """Load the embedding model, ingest the corpus and compile the graph."""
    global _EMBEDDING_MODEL, _COLLECTION, _GRAPH

    mode = "MOCK" if MOCK_LLM else "REAL LLM"

    print(f"[startup] support assistant initialising in {mode} mode")

    _EMBEDDING_MODEL = create_embedding_model()
    _COLLECTION = ingest_documents(_EMBEDDING_MODEL, get_chroma_client())
    _GRAPH = build_graph()

    print(f"[startup] ready (MOCK_LLM={'1' if MOCK_LLM else '0'})")


def run_graph(query: str) -> GroundedAnswer:
    """Execute the compiled graph and validate the result into AskResponse."""
    if _GRAPH is None:
        raise RuntimeError("Graph not initialised; call initialize() first.")

    final_state = _GRAPH.invoke({"query": query})

    # Defensive defaults only: both branches always set all three keys. The
    # fallback mirrors mock semantics so a mock run can never emit 0.0.
    fallback_confidence = MOCK_CONFIDENCE if MOCK_LLM else 0.0

    return GroundedAnswer(
        answer=final_state.get("answer", MOCK_DIRECT_ANSWER),
        sources=list(final_state.get("sources", [])),
        confidence=float(final_state.get("confidence", fallback_confidence)),
    )


# ============================================================
# FastAPI application
# ============================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Run ingestion and graph compilation once, before serving traffic."""
    initialize()

    yield


app = FastAPI(
    title="Zepto Support Assistant",
    description=(
        "Retrieval-augmented support assistant for Zepto policy questions, "
        "built with LangGraph, ChromaDB and sentence-transformers."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/")
def read_root() -> Dict[str, Any]:
    """Service metadata and the routes exposed by the API."""
    return {
        "service": "Zepto Support Assistant",
        "version": "1.0.0",
        "mock_llm": MOCK_LLM,
        "endpoints": {
            "ask": "POST /ask",
            "health": "GET /health",
            "docs": "GET /docs",
        },
    }


@app.get("/health")
def read_health() -> Dict[str, Any]:
    """Readiness probe reporting collection size and corpus state."""
    ready = _COLLECTION is not None and _GRAPH is not None

    return JSONResponse(
        status_code=200 if ready else 503,
        content={
            "status": "ok" if ready else "starting",
            "mock_llm": MOCK_LLM,
            "collection": COLLECTION_NAME,
            "chunks": _COLLECTION.count() if _COLLECTION is not None else 0,
        },
    )


@app.post("/ask", response_model=AskResponse, response_model_exclude_none=False)
def ask(request: AskRequest) -> GroundedAnswer:
    """
    Answer a customer question.

    Input:  {"query": "..."}
    Output: {"answer": "...", "sources": ["doc_01.txt"], "confidence": 0.87}
    """
    try:
        return run_graph(request.query)
    except HTTPException:
        raise
    except Exception as error:
        print(f"[ask] request failed: {error}")

        raise HTTPException(
            status_code=500,
            detail=f"Failed to answer the query: {error}",
        ) from error
