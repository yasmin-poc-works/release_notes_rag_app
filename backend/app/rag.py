from dataclasses import dataclass
from pathlib import Path
import json
import hashlib
import math
import re
from typing import Iterable, Protocol

from pypdf import PdfReader
from qdrant_client import QdrantClient, models
from sentence_transformers import CrossEncoder, SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer

from .config import EMBEDDING_MODEL, MIN_HYBRID_SCORE, QDRANT_COLLECTION, QDRANT_PATH, RERANKER_MODEL, TMP_DIR


@dataclass(frozen=True)
class Chunk:
    text: str
    source: str
    page: int


class Embedder(Protocol):
    def encode(self, sentences: list[str], **kwargs): ...


class Reranker(Protocol):
    def predict(self, pairs: list[list[str]]): ...


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def load_chunks(data_dir: Path, chunk_size: int = 900, overlap: int = 150) -> list[Chunk]:
    chunks: list[Chunk] = []
    for pdf_path in sorted(data_dir.glob("*.pdf")):
        reader = PdfReader(str(pdf_path))
        for page_number, page in enumerate(reader.pages, start=1):
            text = (page.extract_text() or "").strip()
            if not text:
                continue
            start = 0
            while start < len(text):
                end = min(start + chunk_size, len(text))
                if text[start:end].strip():
                    chunks.append(Chunk(text[start:end].strip(), pdf_path.name, page_number))
                if end == len(text):
                    break
                start = max(end - overlap, start + 1)
    return chunks


class HybridRetriever:
    """Sentence Transformer + BM25 retrieval in local Qdrant, then reranking."""

    def __init__(self, chunks: Iterable[Chunk], qdrant_path: Path = QDRANT_PATH,
                 embedder: Embedder | None = None, reranker: Reranker | None = None):
        self.chunks = list(chunks)
        self.documents = [c.text for c in self.chunks]
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words="english")
        self.sparse_matrix = self.vectorizer.fit_transform(self.documents) if self.documents else None
        self.doc_tokens = [_tokens(c.text) for c in self.chunks]
        self.avg_len = sum(map(len, self.doc_tokens)) / max(len(self.doc_tokens), 1)
        self.embedder = embedder or SentenceTransformer(EMBEDDING_MODEL) if self.chunks else embedder
        self.reranker = reranker or CrossEncoder(RERANKER_MODEL) if self.chunks else reranker
        self.qdrant = None
        if self.chunks:
            TMP_DIR.mkdir(parents=True, exist_ok=True)
            qdrant_path.mkdir(parents=True, exist_ok=True)
            self.qdrant = QdrantClient(path=str(qdrant_path))
            dimension = self.embedder.get_sentence_embedding_dimension() if hasattr(self.embedder, "get_sentence_embedding_dimension") else None
            manifest_path = qdrant_path / "index_manifest.json"
            manifest = {"fingerprint": self._data_fingerprint(), "embedding_model": EMBEDDING_MODEL, "dimension": dimension}
            cached = manifest_path.exists() and self.qdrant.collection_exists(f"{QDRANT_COLLECTION}_{dimension}d")
            if cached:
                try:
                    cached_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                    cached = cached_manifest == manifest
                except (OSError, ValueError):
                    cached = False
            if cached:
                self.collection_name = f"{QDRANT_COLLECTION}_{dimension}d"
                return
            vectors = self._encode(self.documents)
            dimension = len(vectors[0])
            # Include the embedding dimension to avoid collisions with a
            # previous local collection created by another model/test.
            self.collection_name = f"{QDRANT_COLLECTION}_{len(vectors[0])}d"
            if self.qdrant.collection_exists(self.collection_name):
                self.qdrant.delete_collection(self.collection_name)
            self.qdrant.create_collection(collection_name=self.collection_name, vectors_config=models.VectorParams(
                size=len(vectors[0]), distance=models.Distance.COSINE))
            self.qdrant.upsert(collection_name=self.collection_name, wait=True, points=[
                models.PointStruct(id=i, vector=vector, payload={"index": i, "text": chunk.text,
                    "source": chunk.source, "page": chunk.page})
                for i, (chunk, vector) in enumerate(zip(self.chunks, vectors))
            ])
            manifest["dimension"] = dimension
            manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    def _data_fingerprint(self) -> str:
        digest = hashlib.sha256()
        for chunk in self.chunks:
            digest.update(f"{chunk.source}|{chunk.page}|{chunk.text}".encode("utf-8"))
        return digest.hexdigest()

    def _encode(self, texts: list[str]) -> list[list[float]]:
        vectors = self.embedder.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return vectors.tolist() if hasattr(vectors, "tolist") else [list(vector) for vector in vectors]

    def _bm25(self, query: str) -> list[float]:
        query_terms = set(_tokens(query))
        df = {term: sum(term in tokens for tokens in self.doc_tokens) for term in query_terms}
        scores = []
        for tokens in self.doc_tokens:
            score = 0.0
            for term in query_terms:
                tf = tokens.count(term)
                if tf:
                    idf = math.log(1 + (len(self.documents) - df[term] + 0.5) / (df[term] + 0.5))
                    score += idf * tf * 2.5 / (tf + 1.5 * (0.25 + 0.75 * len(tokens) / max(self.avg_len, 1)))
            scores.append(score)
        return scores

    def _rerank(self, query: str, candidates: list[tuple[int, float]], top_k: int) -> list[Chunk]:
        pairs = [[query, self.documents[index]] for index, _ in candidates]
        try:
            scores = self.reranker.predict(pairs, batch_size=16, show_progress_bar=False)
        except TypeError:
            # Keeps injected test doubles and compatible reranker adapters working.
            scores = self.reranker.predict(pairs)
        ranked = sorted(zip(scores, candidates), key=lambda item: float(item[0]), reverse=True)
        return [self.chunks[index] for _, (index, _) in ranked[:top_k]]

    def search(self, query: str, top_k: int = 6) -> list[Chunk]:
        if not self.chunks or self.qdrant is None:
            return []
        lexical = self._bm25(query)
        query_vector = self._encode([query])[0]
        points = self.qdrant.query_points(collection_name=self.collection_name, query=query_vector,
                                          limit=min(max(top_k * 3, 10), len(self.chunks)), with_payload=True).points
        dense_scores = {int(point.id): max(float(point.score), 0.0) for point in points}
        candidate_ids = set(dense_scores)
        candidate_ids.update(sorted(range(len(self.chunks)), key=lambda i: lexical[i], reverse=True)[:top_k * 3])
        max_lexical = max(lexical, default=0.0) or 1.0
        max_dense = max(dense_scores.values(), default=0.0) or 1.0
        candidates = [(i, 0.55 * lexical[i] / max_lexical + 0.45 * dense_scores.get(i, 0.0) / max_dense)
                      for i in candidate_ids]
        candidates = [(i, score) for i, score in candidates if score >= MIN_HYBRID_SCORE]
        return self._rerank(query, candidates, top_k) if candidates else []
