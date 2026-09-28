from pathlib import Path
import os
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parents[1]
load_dotenv(BASE_DIR / ".env")

RELEASE_NOTES_DIR = BASE_DIR / "Release Notes"
TMP_DIR = BASE_DIR / "tmp"
QDRANT_PATH = TMP_DIR / "qdrant"
QDRANT_COLLECTION = "release_notes"
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
RERANKER_MODEL = os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-2-v2")
MIN_HYBRID_SCORE = float(os.getenv("MIN_HYBRID_SCORE", "0.12"))
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
TOP_K = int(os.getenv("TOP_K", "6"))
