# Zepto Data & AI Platform

---

## Executive Summary

This repository is a connected data and AI platform built around a single
retail domain — quick commerce — delivered as three independent but
complementary modules.

**Module 1 — Data Pipeline (`/data_pipeline`)** scrapes a live book catalogue,
validates and cleans the records, converts currency with a fixed baseline rate,
and loads the result into a normalized SQLite schema. It finishes by proving the
relational join is correct using a second, independent implementation.

**Module 2 — Analytics Pipeline (`/analytics`)** loads a dataset, profiles it,
handles missing values by severity, performs exploratory and correlation
analysis, compares three classification models, addresses class imbalance,
tunes the winner, fits a regression side-task, and persists a complete
sklearn pipeline to disk. The artifact is then reloaded and used for inference
to prove it is genuinely deployable rather than merely saved.

**Module 3 — Support Assistant (`/support_assistant`)** is a
retrieval-augmented generation service. Eight Zepto policy documents are
chunked, embedded with `all-MiniLM-L6-v2`, and stored in a local ChromaDB
vector store. A LangGraph state machine routes each incoming question, retrieves
the relevant policy chunks, and returns a strictly-validated answer over a
FastAPI endpoint. A `MOCK_LLM` toggle switches between a deterministic offline
baseline and a real LLM path without changing the API contract.

The modules are separated by concern rather than by deployment: Module 1
produces data, Module 2 derives insight from data, and Module 3 serves
data-derived knowledge. Each has its own `README.md` with full detail.

### Project structure

```
zepto_capstone/
│
├── data_pipeline/
│   ├── pipeline.py           Scraping, cleaning, validation, SQLite, SQL
│   ├── zepto_catalog.db      Normalized SQLite database
│   ├── sql_outputs.txt       Captured SQL query output
│   └── README.md             Module 1 documentation
│
├── analytics/
│   ├── analytics_pipeline.py EDA, modelling, tuning, persistence
│   ├── titanic.csv           Raw dataset, saved on first load
│   ├── univariate.png        Univariate distributions
│   ├── corr_heatmap.png      Correlation matrix
│   ├── story_1.png           Survival by sex
│   ├── story_2.png           Survival by class
│   ├── story_3.png           Survival by sex and class
│   ├── story_4.png           Survival by fare scatter
│   ├── tree.png              Decision tree visualization
│   ├── residuals.png         Regression residual plot
│   ├── model_pipeline.pkl    Persisted complete sklearn pipeline
│   └── README.md             Module 2 documentation
│
├── support_assistant/
│   ├── docs/                 8 Zepto policy documents
│   ├── main.py               FastAPI + LangGraph + ChromaDB service
│   ├── requirements.txt      Module-local dependencies
│   ├── Dockerfile            Container build
│   └── README.md             Module 3 documentation
│
├── venv/
├── requirements.txt          Consolidated dependencies
└── README.md                 This file
```

---

## Setup Instructions

### 1. Create a virtual environment

From the project root:

```
python -m venv venv
```

Activate it.

Windows:

```
venv\Scripts\activate
```

Linux/macOS:

```
source venv/bin/activate
```

### 2. Install dependencies

```
pip install -r requirements.txt
```

The consolidated `requirements.txt` covers all three modules:

| Package | Used by |
| --- | --- |
| `requests` | Module 1 — HTTP scraping |
| `beautifulsoup4` | Module 1 — HTML parsing |
| `pandas` | Modules 1 and 2 — data manipulation |
| `seaborn` | Module 2 — visualization |
| `scikit-learn` | Module 2 — preprocessing, modelling, evaluation |
| `imbalanced-learn` | Module 2 — SMOTE oversampling |
| `matplotlib` | Module 2 — plotting |
| `joblib` | Module 2 — model persistence |
| `fastapi` | Module 3 — HTTP API |
| `uvicorn` | Module 3 — ASGI server |
| `pydantic` | Module 3 — strict schema validation |
| `sentence-transformers` | Module 3 — embeddings |
| `chromadb` | Module 3 — vector store |
| `langgraph` | Module 3 — routing graph |

SQLite is used through Python's built-in `sqlite3` standard library module, so
it requires no separate installation.

> Module 3 downloads the `all-MiniLM-L6-v2` model on first startup. The first
> run takes noticeably longer than subsequent runs.

### 3. Run the modules

Each module is independent and can be run in any order.

```
python data_pipeline/pipeline.py
python analytics/analytics_pipeline.py
uvicorn support_assistant.main:app --port 8000
```

---

## Module 1 — Data Pipeline

**Location:** `/data_pipeline`
**Entry point:** `pipeline.py`

### How to run

```
python data_pipeline/pipeline.py
```

### What it does

Scrapes the first five catalogue pages of `https://books.toscrape.com/` at 20
books per page, producing **100 books**. Each record is parsed, validated,
converted to INR, and written to SQLite. Six SQL queries are then executed and
their output is captured to `data_pipeline/sql_outputs.txt`.

The most recent run produced:

```
Raw rows:              100
Clean rows:            100
Invalid titles:        0
Invalid prices:        0
Invalid ratings:       0
Invalid availability:  0
Invalid categories:    0
```

### Fixed baseline currency conversion rate

All GBP-to-INR conversion uses the required fixed baseline rate:

```
1 GBP = 105.50 INR
```

Conversions are computed with Python's `Decimal` type using `ROUND_HALF_UP`
rather than binary floating point, so results are exact and reproducible. For
example, `£51.77` converts to `51.77 × 105.50 = 5461.735`, which rounds to
`5461.74 INR`. The same Decimal-based calculation is reused during validation,
so the stored `price_inr` is independently re-derived and checked.

### Design decisions

**Normalized two-table schema.** The database uses `categories` and `books`
rather than one wide table:

```
categories
    |
    | category_id
    |
    v
  books
```

`categories` holds unique category names; `books` holds book fields plus a
`category_id` foreign key. This avoids repeating the same category string on
every book row and keeps the data in third normal form.

**Dropping malformed records over imputing them.** Records missing a required
title, price, rating, availability, or category are removed from the dataset
rather than being filled with invented placeholder values. For a catalogue
pipeline, a fabricated price or rating is a correctness defect that silently
propagates into every downstream analysis. A smaller, trustworthy dataset is a
better outcome than a larger, subtly wrong one. In this run no records were
actually rejected, so the 100 scraped rows survived intact.

**Cross-verification with SQL JOIN and `pandas.merge`.** The SQL `JOIN` result
is reproduced independently with `pandas.merge()` and the two are compared for
equality. This is a genuine second opinion rather than a restatement: the two
implementations use different join mechanisms, so agreement is evidence the
relational schema and the key mapping are correct. The pipeline asserts and
reports:

```
SQL JOIN == pandas.merge: True
```

---

## Module 2 — Analytics Pipeline

**Location:** `/analytics`
**Entry point:** `analytics_pipeline.py`

### How to run

```
python analytics/analytics_pipeline.py
```

All metrics below are real output from a verified run of the pipeline.

### Missing-value handling

Missing values are treated according to the percentage of affected rows, so
the intervention scales with severity:

| Column | Missing | Action applied | Rationale |
| --- | --- | --- | --- |
| `deck` | 77.22% | Encode as `"missing"` | Too sparse to drop or impute reliably; recorded as its own category |
| `age` | 19.87% | Median impute | Material enough to distort the model if left null, small enough to impute safely |
| `embarked` | 0.22% | Drop affected rows | Negligible loss |
| `embark_town` | 0.22% | Drop affected rows | Negligible loss |

Cleaned shape: `(889, 15)` from an original `(891, 15)`. The two dropped rows
are the only ones lost.

The `deck` column is a Pandas `Categorical`, so a missing value cannot simply be
assigned into it. The category is added first, then filled, which avoids:

```
TypeError: Cannot setitem on a Categorical with a new category (missing)
```

### Skewness of `fare`

`fare` is **strongly right-skewed**, with a skewness of approximately **4.79**,
against a mean of 32.10 and a median of 14.45. The mean sitting more than twice
the median is the signature of a small number of very high fares pulling the
mean upward. By contrast `age` is near-symmetric at a skewness of roughly
0.39.

IQR outlier analysis quantifies the tails:

| Feature | Q1 | Q3 | IQR | Lower | Upper | Outliers |
| --- | --- | --- | --- | --- | --- | --- |
| `Age` | 22.0 | 35.0 | 13.0 | 2.5 | 54.5 | 65 |
| `Fare` | 7.8958 | 31.0 | 23.1042 | -26.7605 | 65.6563 | 114 |

The 114 flagged fares are inspected rather than removed. Premium passenger
class legitimately produces high fares, and `pclass` correlates with `fare` at
**-0.548** — the strongest pairwise correlation in the dataset. Those outliers
carry real signal about passenger class, so deleting them would discard
information.

### Bivariate findings

Survival rate by sex:

| Sex | Survival rate |
| --- | --- |
| female | 0.740 |
| male | 0.189 |

Survival rate by class:

| Class | Survival rate |
| --- | --- |
| 1 | 0.626 |
| 2 | 0.473 |
| 3 | 0.242 |

Survival rate by sex and class:

| Sex | Class | Survival rate |
| --- | --- | --- |
| female | 1 | 0.967 |
| female | 2 | 0.921 |
| female | 3 | 0.500 |
| male | 1 | 0.369 |
| male | 2 | 0.157 |
| male | 3 | 0.135 |

Sex is the dominant factor, with a gap of roughly 0.55 in absolute survival
rate. Class is a clear but weaker gradient. The joint view shows the two effects
compounding: first-class women reached 0.967 survival, while third-class men
reached 0.135, a 7x difference between the extremes. Notably, third-class women
survived at 0.500 — the class penalty is sharpest for women, whose overall rate
rests heavily on first and second class.

Two strongest absolute correlations in the dataset:

| Feature 1 | Feature 2 | Correlation | Absolute |
| --- | --- | --- | --- |
| `pclass` | `fare` | -0.548193 | 0.548193 |
| `sibsp` | `parch` | 0.414542 | 0.414542 |

### Class imbalance

The target is moderately imbalanced at 61.75% negative against 38.25% positive.
A stratified 80/20 split keeps the test distribution representative. The model
was fitted at 711 rows, with 178 held out.

### Model comparison

| Model | Accuracy | Precision | Recall | F1 | ROC AUC |
| --- | --- | --- | --- | --- | --- |
| Logistic Regression | 0.8090 | 0.7833 | 0.6912 | 0.7344 | 0.8610 |
| Decision Tree | 0.7640 | 0.7600 | 0.5588 | 0.6441 | 0.8374 |
| Random Forest | 0.8090 | 0.7656 | 0.7206 | 0.7424 | 0.8196 |

Random Forest and Logistic Regression tie on accuracy at 0.8090, but they get
there differently. Random Forest has the better F1 (0.7424 vs 0.7344) and
higher recall (0.7206 vs 0.6912), meaning it catches more actual survivors,
whereas Logistic Regression has the better precision (0.7833 vs 0.7656) and the
best ranking quality by ROC AUC (0.8610 vs 0.8196). The Decision Tree is
clearly weakest, and its recall of 0.5588 is the weakest signal of the three —
it misses too many survivors.

### Imbalance strategies

| Strategy | Precision | Recall | F1 |
| --- | --- | --- | --- |
| Baseline RF | 0.7656 | 0.7206 | 0.7424 |
| Balanced RF | 0.7656 | 0.7206 | 0.7424 |
| SMOTE RF | 0.7612 | 0.7500 | 0.7556 |

`class_weight="balanced"` produced results identical to the baseline. That is
the expected outcome at this mild imbalance level — there is too little signal
for reweighting to move the decision boundary. SMOTE is the only strategy that
changed anything, trading a small amount of precision for a meaningful recall
gain, and it produced the best F1 of the three at 0.7556.

Crucially, SMOTE is applied **inside** the imbalanced-learn pipeline, after the
train/test split, rather than to the full dataset beforehand. Applying it before
splitting would synthesise training rows from test data and leak information
across the boundary, inflating every reported metric.

### Grid search

A `GridSearchCV` with `cv=5` searched `n_estimators`, `max_depth`, and
`max_features`. Best parameters:

```
{'model__max_depth': 5, 'model__max_features': 'sqrt', 'model__n_estimators': 200}
```

Tuned Random Forest on the held-out test set:

| Metric | Value |
| --- | --- |
| Accuracy | 0.8315 |
| Precision | 0.8654 |
| Recall | 0.6618 |
| F1 | 0.7500 |
| ROC AUC | 0.8389 |
| OOB score | 0.8214 |

### Regression side-task on `fare`

A multivariable Linear Regression was fitted on `pclass`, `sex`, `age`, `sibsp`,
`parch`, and `embarked`:

| Metric | Value |
| --- | --- |
| MAE | 21.14 |
| RMSE | 41.75 |
| R² | 0.3468 |
| Adjusted R² | 0.3118 |

The model explains only about 35% of fare variance. This is a weak fit, and
that is an honest result rather than a defect: passenger class is the dominant
driver and does not fully determine fare, which also varies with cabin position
and family size in ways this feature set does not capture. The large gap
between MAE and RMSE indicates the residual distribution is heavy-tailed, which
is consistent with the `fare` skewness identified earlier. This side-task is
included to demonstrate the regression workflow end to end, and is not
recommended for production use.

### Deployment recommendation

`analytics/model_pipeline.pkl` stores the **complete** tuned pipeline —
preprocessing, scaling, one-hot encoding, and the Random Forest estimator
together as a single artifact. The deployment recommendation is to load and use
this artifact rather than reimplementing the feature steps in application code:

```python
pipeline = joblib.load("analytics/model_pipeline.pkl")
prediction = pipeline.predict(raw_input)
```

Two reasons. First, correctness: fitting preprocessing inside the pipeline
guarantees the exact transforms used at training time are applied at inference
time, so a hand-written inference path cannot silently drift from the trained
model. Second, practicality: the preprocessor cannot be applied to a raw input
DataFrame that lacks the training columns, which the pipeline handles
automatically.

The reload test confirms this works. Supplying a raw record with no prior
preprocessing:

```
   pclass   sex   age  sibsp  parch  fare embarked
0       3  male  30.0      0      0  10.0        S
Prediction: [0]
Survival probability: 0.118115
```

A third-class male passenger aged 30 travelling alone is predicted not to
survive with probability 0.118, which is consistent with the 0.135 survival
rate observed for third-class men in the bivariate analysis.

---

## Module 3 — Support Assistant

**Location:** `/support_assistant`
**Entry point:** `main.py`

### How to run locally

From the project root:

```
uvicorn support_assistant.main:app --port 8000
```

Then query it:

```
curl -X POST http://localhost:8000/ask -H "Content-Type: application/json" -d "{\"query\": \"What is the delivery fee for orders under INR 149?\"}"
```

Interactive API documentation is served at `http://localhost:8000/docs`.

A health check confirms readiness and the size of the loaded collection:

```
curl http://localhost:8000/health
```

```json
{"status": "ok", "mock_llm": true, "collection": "zepto_support_policies", "chunks": 15}
```

### How to run with Docker

Build the image from the project root:

```
docker build -t zepto-support ./support_assistant
```

Run the container:

```
docker run -p 7860:7860 zepto-support
```

The service is then available at `http://localhost:7860`. The `Dockerfile`
sets `MOCK_LLM=1` by default, so the container needs no API key.

> The Dockerfile defaults to port 7860 while the local command above uses 8000.
> Change the port mapping on either side to keep them consistent.

To run the container against the real LLM:

```
docker run -p 7860:7860 -e MOCK_LLM=0 -e GROQ_API_KEY=your_key_here zepto-support
```

### Test transcripts

Both transcripts were captured from a running server with `MOCK_LLM` left at
its default value (unset, which is equivalent to `MOCK_LLM=1`).

#### Transcript 1 — policy question

The query contains `delivery`, so the router classifies it as
`policy_question`, the graph retrieves from ChromaDB, and the answer is
generated from the retrieved context.

Request:

```
curl -X POST http://localhost:8000/ask -H "Content-Type: application/json" -d "{\"query\": \"What is the delivery fee for orders under INR 149?\"}"
```

Response:

```json
{"answer":"Based on the retrieved context: Zepto delivers grocery and household essentials to serviceable pin codes within 10 to 30 minutes of order confirmation, depending on the customer's delivery zone and current order volume. Standard del","sources":["doc_01.txt","doc_03.txt"],"confidence":0.531}
```

`doc_01.txt` is the correct source, since it is the document that defines the
INR 149 delivery threshold. The mock answer is deliberately raw — the leading
~200 characters of the top-ranked chunk rather than a fluent sentence. That is
the expected baseline output in mock mode and is what makes the result
reproducible without any model call.

#### Transcript 2 — general question

The query contains none of the routing keywords, so the router classifies it as
`general_question`. The graph takes the `direct_answer` branch, skips retrieval
entirely, and returns the fixed refusal.

Request:

```
curl -X POST http://localhost:8000/ask -H "Content-Type: application/json" -d "{\"query\": \"Can you help me write a python script?\"}"
```

Response:

```json
{"answer":"I can only answer questions about Zepto policies right now.","sources":[],"confidence":0.0}
```

The empty `sources` array and zero `confidence` are the correct result for a
question that has no supporting policy context.

> **PowerShell note.** In Windows PowerShell, `curl` is an alias for
> `Invoke-WebRequest` and nested quotes behave differently. Use `curl.exe` with a
> payload file, or use `Invoke-RestMethod`:
>
> ```powershell
> Invoke-RestMethod -Uri http://localhost:8000/ask -Method Post `
>   -ContentType "application/json" `
>   -Body '{"query": "What is the delivery fee for orders under INR 149?"}'
> ```

### Architecture

The service is a retrieval-augmented generation pipeline:

```
docs/doc_01.txt ... doc_08.txt
        │
        │  ingestion + sentence-aware chunking
        ▼
   8 documents → 15 chunks
        │
        │  embedding: all-MiniLM-L6-v2, L2-normalised
        ▼
   ChromaDB persistent collection (cosine space)
        │
        │  LangGraph: classify_intent
        │        │
        │        ├── policy_question ──▶ retrieve_and_answer ──▶ END
        │        │                          (ChromaDB top-3,
        │        │                           Pydantic formatting)
        │        │
        │        └── general_question ──▶ direct_answer ─────▶ END
        ▼
   POST /ask → {"answer", "sources", "confidence"}
```

**Ingestion.** Eight UTF-8 policy documents are read from `support_assistant/docs/`
and split into overlapping, sentence-aware chunks of at most 420 characters.
Splitting on sentence boundaries rather than at a fixed offset keeps each policy
intact — the INR 1000 damaged-item threshold is never separated from the
sentence that explains it. Ingestion and embedding run once at startup, never on
the request path. The ChromaDB collection is content-addressed with a SHA-256
fingerprint of the chunk set, so an unchanged corpus is reused and only genuine
drift triggers a rebuild.

**Embedding.** Chunks are encoded with `all-MiniLM-L6-v2` using
`normalize_embeddings=True`, so stored vectors are unit length and cosine
similarity is computed directly. Vectors are passed to ChromaDB explicitly
rather than through a registered embedding function, keeping the vector space
under the application's control. The collection uses the `cosine` HNSW space.

**Retrieval.** The incoming question is embedded with the same model and used
to query ChromaDB for the **top 3** chunks, nearest first. Query and document
vectors are normalised identically, so distances are directly comparable.
Confidence is derived as `1 - cosine_distance` of the top hit, clamped to
`[0, 1]` and rounded to three decimals — a deterministic function of the
corpus and query, which is precisely what makes the mock baseline
reproducible.

**LangGraph routing.** A `StateGraph` with a `TypedDict` state drives three
nodes. `classify_intent` sets the intent, and a conditional edge routes to
either `retrieve_and_answer` (which always performs retrieval, then generates)
or `direct_answer` (which skips retrieval and returns the fixed refusal).
`retrieve_and_answer` retrieves in **both** modes by design, so the mock
baseline genuinely exercises the vector store rather than stubbing it out.

**Pydantic response formatting.** A single strict model, `GroundedAnswer`,
validates the LLM's structured output and serialises the HTTP response. Because
both the mock and real paths terminate in the same model, any `200` response is
guaranteed to match the schema `{answer, sources, confidence}` with unknown
fields forbidden. Requests are validated by a matching `AskRequest` model, and
malformed requests are rejected with `422` before reaching the graph.

### `MOCK_LLM` — deterministic execution by default

**`MOCK_LLM=1` is the default and requires no third-party API key, no network
access, and no account.** When `MOCK_LLM` is unset or set to `"1"`, every stage
executes deterministically and offline. The variable is read once at import time
as `os.getenv("MOCK_LLM", "1") != "0"`, so only an explicit `"0"` selects the
real LLM path.

The toggle changes behaviour as follows:

| Node | `MOCK_LLM` unset / `1` | `MOCK_LLM=0` |
| --- | --- | --- |
| `classify_intent` | Keyword match against a fixed list. No LLM call, no network. | LLM classification via `INTENT_PROMPT`, validated against the `IntentDecision` schema. Falls back to keywords if the API is unreachable. |
| `retrieve_and_answer` | Retrieval runs. Generation returns a fixed prefix plus the top chunk's first ~200 characters. | Retrieval runs identically. Generation calls the LLM with the structured `ANSWER_PROMPT` and a 2-retry corrective loop. |
| `direct_answer` | Fixed refusal string. | Identical — no LLM call in either mode. |

The keyword router matches `delivery`, `return`, `refund`, `membership`,
`tracking`, `cancel`, `gift card`, and `support hours`, case-insensitively.
A query containing any of them routes to `policy_question`; anything else routes
to `general_question`.

`direct_answer` is intentionally identical in both modes. It returns a constant
string, so a model call would add latency, cost, and a failure mode with no
benefit.

Setting `MOCK_LLM=0` additionally requires `GROQ_API_KEY` and enables a
**2-retry corrective loop**: the initial completion is parsed and validated
against `GroundedAnswer`, and on failure the exact Pydantic validation error
plus the required JSON schema are fed back for up to two corrective attempts
(at most three API calls total) before the request fails with a clear error.
This keeps malformed model output from ever reaching a client.

### API reference

`POST /ask` accepts `{"query": "<question>"}` and returns:

| Field | Type | Description |
| --- | --- | --- |
| `answer` | string | The grounded answer |
| `sources` | list of strings | Source filenames supporting the answer |
| `confidence` | float | Confidence between `0.0` and `1.0` |

---

## Technologies Used

| Technology | Role |
| --- | --- |
| Python | Core language across all modules |
| Requests | HTTP scraping |
| Beautiful Soup | HTML parsing |
| Pandas | Data manipulation and validation |
| SQL / SQLite | Normalized storage and querying |
| Decimal | Exact fixed-rate currency arithmetic |
| NumPy | Numerical computation |
| Seaborn | Statistical visualization |
| Matplotlib | Plotting |
| Scikit-learn | Preprocessing, modelling, tuning, evaluation |
| imbalanced-learn | SMOTE oversampling |
| Joblib | Model persistence and reloading |
| FastAPI | HTTP API framework |
| Uvicorn | ASGI server |
| Pydantic | Strict schema validation |
| LangGraph | Stateful routing graph |
| ChromaDB | Local persistent vector store |
| sentence-transformers | `all-MiniLM-L6-v2` embeddings |
| Docker | Containerization |

---

## Generated Artifacts

Running the pipelines produces the following files.

| File | Module | Purpose |
| --- | --- | --- |
| `data_pipeline/zepto_catalog.db` | 1 | Normalized SQLite database |
| `data_pipeline/sql_outputs.txt` | 1 | Captured SQL output |
| `analytics/titanic.csv` | 2 | Raw dataset |
| `analytics/univariate.png` | 2 | Univariate distributions |
| `analytics/corr_heatmap.png` | 2 | Correlation matrix |
| `analytics/story_1.png` | 2 | Survival by sex |
| `analytics/story_2.png` | 2 | Survival by class |
| `analytics/story_3.png` | 2 | Survival by sex and class |
| `analytics/story_4.png` | 2 | Survival by fare scatter |
| `analytics/tree.png` | 2 | Decision tree visualization |
| `analytics/residuals.png` | 2 | Regression residual plot |
| `analytics/model_pipeline.pkl` | 2 | Persisted complete sklearn pipeline |
| `support_assistant/chroma_db/` | 3 | Generated vector store (untracked) |

---

## Reproducibility

Fixed random seeds (`random_state=42`) are used for train/test splitting,
Decision Tree and Random Forest initialisation, SMOTE, and grid search, so
Module 2 results are repeatable across runs.

Module 3 produces identical responses for identical queries in mock mode,
because both the retrieved chunk order and the derived confidence are
deterministic functions of a fixed corpus and a fixed embedding model.

Module 1 uses Decimal arithmetic with `ROUND_HALF_UP` rather than floating
point, so currency conversion is exact and independent of platform.

---

## Conclusion

This platform demonstrates three connected competencies end to end:

- **Data engineering** — scraping, validation, exact currency conversion,
  normalized persistence, and independent cross-verification
- **Machine learning** — statistical reasoning about missing data and skew,
  model selection against stated evidence, leakage-free imbalance handling,
  and a persisted artifact proven to work on raw input
- **Generative AI** — retrieval-augmented generation with a deliberate
  deterministic evaluation mode, a routing graph that constrains the model to
  its sources, and schema enforcement that makes the API contract reliable

Each module documents its own tradeoffs in detail in its local `README.md`.
