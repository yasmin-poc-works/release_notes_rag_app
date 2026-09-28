from functools import lru_cache
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from groq import Groq

from .config import GROQ_MODEL, RELEASE_NOTES_DIR, TOP_K
from .rag import HybridRetriever, load_chunks

@asynccontextmanager
async def lifespan(_app: FastAPI):
    # Warm models and the local Qdrant index before the first user request.
    get_retriever()
    yield


app = FastAPI(title="Release Notes Hybrid RAG API", version="0.1.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=4000)


class ChatResponse(BaseModel):
    answer: str
    sources: list[dict[str, str | int]] = Field(default_factory=list)


@lru_cache(maxsize=1)
def get_retriever() -> HybridRetriever:
    return HybridRetriever(load_chunks(RELEASE_NOTES_DIR))


@app.get("/health")
def health() -> dict[str, str | int]:
    return {"status": "ok", "chunks": len(get_retriever().chunks)}


@app.post("/api/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    matches = get_retriever().search(request.message, TOP_K)
    if not matches:
        return ChatResponse(
            answer="This cannot be searched like that. Please ask about a release note, feature, fix, or release date.",
        )
    context = "\n\n---\n\n".join(c.text for c in matches)
    try:
        client = Groq()
        result = client.chat.completions.create(
            model=GROQ_MODEL,
            temperature=0.1,
            messages=[
                {"role": "system", "content": "You are an assistant helping a tester understand client-provided release notes. Answer only from the provided context. Synthesize and summarize all relevant context into one clear answer. Do not mention PDF names, page numbers, citations, sources, or retrieval. If the context does not answer the question, say that the release notes do not provide enough information. If relevant context contains different changes or dates, combine them in a short chronological or thematic summary."},
                {"role": "user", "content": f"Release-note context:\n{context}\n\nQuestion: {request.message}"},
            ],
        )
        answer = result.choices[0].message.content or "I couldn't generate an answer."
    except Exception as exc:
        raise HTTPException(status_code=503, detail="Groq is unavailable. Check GROQ_API_KEY and try again.") from exc
    sources = list(dict.fromkeys((chunk.source, chunk.page) for chunk in matches))
    source_citations = [{"source": source, "page": page} for source, page in sources]
    return ChatResponse(answer=answer, sources=source_citations)
