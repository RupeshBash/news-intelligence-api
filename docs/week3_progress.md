# Week 3 Progress

## Day 15 — FastAPI Application Setup

### What I completed

- installed and configured FastAPI
- created `app/fastapi_app.py`
- created the main FastAPI application
- added an application lifespan
- loaded MiniLM and ChromaDB once during startup
- stored shared resources for future endpoints
- added `GET /health`
- tested the API using PowerShell
- verified the interactive API documentation

### Workflow

```text
Start FastAPI
        ↓
lifespan startup
        ↓
load MiniLM
        ↓
load ChromaDB
        ↓
store shared resources
        ↓
accept HTTP requests
```

### Health endpoint

```text
GET /health
```

Returns API and resource status, including:

```text
service status
embedding-model readiness
ChromaDB collection name
indexed article count
```

### Key concepts

- FastAPI exposes Python functions through HTTP endpoints.
- Shared ML resources should not reload for every request.
- Application lifespan handles startup and shutdown work.
- API responses should use predictable JSON-ready data.
- `/health` provides a simple readiness check.

### Current limitation

Only the `/health` endpoint is implemented.

Planned next:

```text
Day 16 → POST /predict
Day 17 → POST /similar
Day 18 → POST /analyze
```

---

## Day 16 — Prediction API Endpoint

### What I completed

- created a Pydantic request model for prediction input
- validated and cleaned `news_text`
- created prediction response models
- added `POST /predict`
- reused `predict_category_with_confidence()`
- returned class index, category, and confidence
- tested valid and invalid API requests
- verified the endpoint through FastAPI `/docs`

### Prediction workflow

```text
POST /predict
        ↓
PredictRequest
        ↓
validate news_text
        ↓
predict_category_with_confidence()
        ↓
class index + category + confidence
        ↓
PredictResponse
        ↓
JSON response
```

### Example request

```json
{
  "news_text": "A company introduced a new computer chip that uses less electricity."
}
```

### Example response

```json
{
  "input_text": "A company introduced a new computer chip that uses less electricity.",
  "prediction": {
    "class_index": 4,
    "category": "Sci/Tech",
    "confidence_percent": 89.94
  }
}
```

### Key concepts

- Pydantic models define API input and output contracts.
- Validation should happen before model processing.
- API routes should reuse existing ML functions.
- Prediction responses should use JSON-friendly Python values.
- Invalid request bodies return validation errors before prediction runs.

### Current endpoints

```text
GET  /health
POST /predict
```

### Next step

Day 17 will add:

```text
POST /similar
```

## Day 18 — Combined Analyze Endpoint

### What I completed

- Added `POST /analyze`.
- Reused the existing `analyze_news()` workflow.
- Reused the MiniLM model and ChromaDB collection loaded at API startup.
- Returned classification and semantic-search results in one response.
- Added request and response validation with Pydantic.

### Workflow

`news_text + top_k`
→ FastAPI validation
→ `analyze_news()`
→ classification + semantic search
→ validated JSON response

### Actual result

The `/analyze` endpoint successfully returned:

- predicted class index
- predicted category
- confidence percentage
- requested result count
- similar articles returned
- search time
- ranked similar articles

Validation correctly rejected:

- whitespace-only text
- `top_k < 1`
- `top_k > 10`

### Key concepts

- Reuse existing processing functions instead of duplicating logic.
- Shared ML resources should load once during API startup.
- Internal result fields can be mapped to cleaner API field names.
- Pydantic validates both incoming requests and outgoing responses.

### Mistake pattern

A misspelled response key caused a `ResponseValidationError`.

`similar_articles_returned` must match the response model exactly.

### Current limitation

The API currently runs locally and has not yet been containerized.


---

## Day 19 — Dockerize FastAPI Application

### What I completed

- created a basic `Dockerfile`
- created `.dockerignore`
- built the `ai-news-intelligence` Docker image
- ran FastAPI inside a Linux Docker container
- mapped host port `8000` to container port `8000`
- verified `/health`, `/predict`, `/similar`, and `/analyze`
- confirmed MiniLM and ChromaDB load correctly inside Docker

### Workflow

```text
Dockerfile
    ↓
docker build
    ↓
Docker image
    ↓
docker run
    ↓
Linux container
    ↓
Uvicorn
    ↓
FastAPI