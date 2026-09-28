# Release Notes Hybrid RAG

Small reference app for the release-note PDFs in `backend/Release Notes`.
Retrieval uses Sentence Transformers embeddings, BM25-style lexical matching, a local Qdrant client stored in `backend/tmp/qdrant`, and a CrossEncoder reranker. It does not connect to a production Qdrant URL.

## Run the API

```powershell
cd backend
uv sync
Copy-Item .env.example .env
# add GROQ_API_KEY to .env
uv run uvicorn app.main:app --reload --port 8000
```

## Run the frontend

```powershell
cd frontend
npm install
npm run dev
```

Open http://localhost:3000. The only API is `POST /api/chat` with `{ "message": "..." }`.

## Test

```powershell
cd backend
uv run pytest
```
