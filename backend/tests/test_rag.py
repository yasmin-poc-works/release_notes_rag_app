from pathlib import Path

from app.rag import Chunk, HybridRetriever


class FakeEmbedder:
    def encode(self, texts, **kwargs):
        vocabulary = ["export", "csv", "dashboard", "active", "quantum"]
        return [[float(word in text.lower()) for word in vocabulary] for text in texts]


class FakeReranker:
    def predict(self, pairs):
        return [float(sum(word in text.lower() for word in query.lower().split())) for query, text in pairs]


def test_hybrid_retriever_returns_relevant_chunk():
    retriever = HybridRetriever([
        Chunk("Improved export performance and fixed CSV formatting.", "one.pdf", 1),
        Chunk("Added a new dashboard filter for active users.", "two.pdf", 2),
    ], qdrant_path=Path("tmp/test-qdrant"), embedder=FakeEmbedder(), reranker=FakeReranker())
    assert retriever.search("CSV export", 1)[0].source == "one.pdf"


def test_release_notes_directory_exists():
    assert (Path(__file__).parents[1] / "Release Notes").is_dir()


def test_unrelated_query_returns_no_results():
    retriever = HybridRetriever([
        Chunk("Improved export performance and fixed CSV formatting.", "one.pdf", 1),
    ], qdrant_path=Path("tmp/test-qdrant-empty"), embedder=FakeEmbedder(), reranker=FakeReranker())
    assert retriever.search("quantum banana spaceship", 3) == []
