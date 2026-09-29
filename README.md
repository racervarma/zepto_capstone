# Zepto Data & AI Platform

## Executive Summary

This repository contains an end-to-end data and AI platform designed for Zepto's analytics and support guilds. It consists of three fully integrated modules:

**Data Pipeline (/data_pipeline):** Scrapes raw product catalog data, normalizes it, enforces a strict currency conversion, and loads it into a relational SQLite database.

**Analytics Pipeline (/analytics):** Performs EDA, handles missing values/imbalances, and trains a deployable machine learning classification model on a customer dataset.

**Support Assistant (/support_assistant):** A Retrieval-Augmented Generation (RAG) API that embeds Zepto's internal policies using sentence-transformers and ChromaDB, orchestrated via LangGraph, and served through FastAPI.

## Setup Instructions

### 1. Create a Virtual Environment

```bash
python -m venv venv
```

```powershell
venv\Scripts\activate  # On Windows
```

### 2. Install Dependencies

All dependencies are consolidated in the root `requirements.txt`.

```bash
pip install -r requirements.txt
```

## Module 1: Data Pipeline (/data_pipeline)

### How to run:

```bash
python data_pipeline/pipeline.py
```

### Design Decisions:

**Currency Baseline:** A fixed baseline conversion rate of 1 GBP = 105.50 INR was enforced programmatically. No external API calls were made.

**Database Schema:** A normalized, two-table SQLite schema (categories and books) was designed with a Primary Key / Foreign Key relationship to mimic enterprise data modeling.

**Validation:** Cross-verification of the schema was proven by ensuring a standard SQL JOIN produced the exact same output as a `pandas.merge()`.

## Module 2: Analytics Pipeline (/analytics)

### How to run:

```bash
python analytics/analytics_pipeline.py
```

### Design Decisions & Findings:

**Cleaning Thresholds:** Handled missing values using strict percentage rules (<5% dropped, 5-30% imputed).

**Skewness & Bivariate Insights:** `fare` is heavily right-skewed (mean > median). Survival was most strongly correlated with `sex` and `pclass`.

**Model Comparison:** Evaluated Logistic Regression, Decision Tree, and Random Forest.

**Imbalance Handling:** SMOTE was applied strictly to the training fold to eliminate data leakage.

**Recommendation:** Random Forest is recommended for deployment due to its superior ROC-AUC score and resistance to overfitting, captured and saved as a reloadable `model_pipeline.pkl` artifact.

## Module 3: Support Assistant (/support_assistant)

### Architecture:

**Ingestion:** Reads 8 Zepto policy text files from the `docs/` folder.

**Embedding:** sentence-transformers (`all-MiniLM-L6-v2`) converts text into semantic vectors.

**Retrieval:** ChromaDB performs local cosine-similarity vector search to find the top 3 chunks.

**Generation & Routing:** LangGraph creates a 3-node graph (`classify_intent`, `retrieve_and_answer`, `direct_answer`).

**Offline Mode (`MOCK_LLM=1`):** Provides a strict, deterministic fallback without paid LLM API calls, returning canned templates based on retrieved data.

### How to run locally:

```bash
uvicorn support_assistant.main:app --port 8000
```

### How to run with Docker:

```bash
docker build -t zepto-support ./support_assistant
docker run -p 7860:7860 zepto-support
```

### Web UI (Optional)

A Gradio-based web interface is included for easy testing.

Start the backend:

```bash
uvicorn support_assistant.main:app --port 8000
```

Start the frontend:

```bash
python chat_ui.py
```

Navigate to <http://127.0.0.1:7861>

### Live Cloud Deployment (Stretch Goal)

The FastAPI backend has been successfully containerized via Docker and deployed live using the Free CPU tier on Hugging Face Spaces.

Live Endpoint URL:
<https://huggingface.co/spaces/racervarma/zepto_support>

### Test Transcripts (MOCK_LLM=1 Baseline)

**Policy Query (Routes to Retrieval):**

```bash
curl -X POST http://127.0.0.1:8000/ask -H "Content-Type: application/json" -d "{\"query\": \"How much does delivery cost?\"}"
```

```json
{"answer":"Based on the retrieved context: Zepto delivers grocery and household essentials to serviceable pin codes within 10 to 30 minutes...","sources":["doc_01.txt","doc_05.txt","doc_03.txt"],"confidence":1.0}
```

**General Query (Bypasses Retrieval):**

```bash
curl -X POST http://127.0.0.1:8000/ask -H "Content-Type: application/json" -d "{\"query\": \"What is the capital of France?\"}"
```

```json
{"answer":"I can only answer questions about Zepto policies right now.","sources":[],"confidence":1.0}
```
