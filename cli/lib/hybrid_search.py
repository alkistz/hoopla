import os
from dataclasses import dataclass, field
from typing import Any, Optional

from sentence_transformers import CrossEncoder

from .inverted_index import InvertedIndex
from .llm_setup import improve_query, rerank_batch_prompt, score_movie
from .search_utils import ALPHA, load_movies
from .semantic_search import ChunkedSemanticSearch


@dataclass
class MovieDocument:
    """Represents a movie document."""

    id: str
    title: str = ""
    description: str = ""
    genres: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MovieDocument":
        """Create a MovieDocument from a dictionary."""
        return cls(
            id=str(data["id"]),
            title=data["title"],
            description=data.get("description", data.get("document", "")),
            genres=data.get("genres", []),
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert MovieDocument to dictionary."""
        return {
            "id": self.id,
            "title": self.title,
            "description": self.description,
            "genres": self.genres,
        }


@dataclass
class WeightedSearchResult:
    """Result from weighted hybrid search."""

    doc: MovieDocument
    bm25: float
    semantic: float
    hybrid_score: float


@dataclass
class RRFSearchResult:
    """Result from RRF (Reciprocal Rank Fusion) search."""

    doc: MovieDocument
    rrf_score: float
    bm25_rank: Optional[int] = None
    semantic_rank: Optional[int] = None
    llm_score: Optional[str] = None


class HybridSearch:
    def __init__(self, documents: list[dict[str, Any]]):
        self.documents = documents
        self.semantic_search = ChunkedSemanticSearch()
        self.semantic_search.load_or_create_chunk_embeddings(documents)

        self.idx = InvertedIndex()
        if not os.path.exists(self.idx.index_path):
            self.idx.build()
            self.idx.save()

    def _bm25_search(self, query: str, limit: int) -> list[dict[str, Any]]:
        self.idx.load()
        return self.idx.bm25_search(query, limit)

    def weighted_search(
        self, query: str, alpha: float, limit: int = 5
    ) -> list[WeightedSearchResult]:
        final_results: dict[str, dict[str, Any]] = {}

        bm25_results = self._bm25_search(query, limit * 500)
        bm25_scores = normalise([r["score"] for r in bm25_results])

        if not bm25_scores:
            raise ValueError("No bm25 scores were provided!")

        for i, doc in enumerate(bm25_results):
            doc_id = str(doc["id"])
            if not final_results.get(doc_id):
                final_results[doc_id] = {"doc": doc, "bm25": 0, "semantic": 0}
            final_results[doc_id]["bm25"] = bm25_scores[i]

        semantic_results = self.semantic_search.search_chunks(query, limit=limit * 500)
        semantic_scores = normalise([r["score"] for r in semantic_results])

        if not semantic_scores:
            raise ValueError("No semantic scores were provided!")

        for i, doc in enumerate(semantic_results):
            doc_id = str(doc["id"])
            if not final_results.get(doc_id):
                final_results[doc_id] = {"doc": doc, "bm25": 0, "semantic": 0}
            final_results[doc_id]["semantic"] = semantic_scores[i]

        # Build typed results
        results = []
        for doc_id, data in final_results.items():
            bm25_score = data.get("bm25", 0)
            semantic_score = data.get("semantic", 0)
            results.append(
                WeightedSearchResult(
                    doc=MovieDocument.from_dict(data["doc"]),
                    bm25=bm25_score,
                    semantic=semantic_score,
                    hybrid_score=hybrid_score(bm25_score, semantic_score, alpha),
                )
            )

        # Sort by hybrid score
        results.sort(key=lambda x: x.hybrid_score, reverse=True)
        return results[:limit]

    def rrf_search(self, query: str, k: int, limit: int = 10) -> list[RRFSearchResult]:
        final_results: dict[str, dict[str, Any]] = {}
        bm25_results = self._bm25_search(query, limit * 500)
        semantic_results = self.semantic_search.search_chunks(query, limit=limit * 500)

        for i, doc in enumerate(bm25_results):
            doc_id = str(doc["id"])
            rrf_score = calculate_rrf_score(i, k)
            if final_results.get(doc_id):
                final_results[doc_id]["rrf_score"] += rrf_score
                final_results[doc_id]["bm25_rank"] = i
            else:
                final_results[doc_id] = {
                    "doc": doc,
                    "rrf_score": rrf_score,
                    "bm25_rank": i,
                }

        for i, doc in enumerate(semantic_results):
            doc_id = str(doc["id"])
            rrf_score = calculate_rrf_score(i, k)
            if final_results.get(doc_id):
                final_results[doc_id]["rrf_score"] += rrf_score
                final_results[doc_id]["semantic_rank"] = i
            else:
                final_results[doc_id] = {
                    "doc": doc,
                    "rrf_score": rrf_score,
                    "semantic_rank": i,
                }

        # Build typed results
        results = [
            RRFSearchResult(
                doc=MovieDocument.from_dict(data["doc"]),
                rrf_score=data["rrf_score"],
                bm25_rank=data.get("bm25_rank"),
                semantic_rank=data.get("semantic_rank"),
            )
            for data in final_results.values()
        ]

        # Sort by RRF score
        results.sort(key=lambda x: x.rrf_score, reverse=True)
        return results[:limit]


def calculate_rrf_score(rank: int, k: int = 60) -> float:
    """Calculate RRF (Reciprocal Rank Fusion) score for a given rank."""
    return 1 / (k + rank)


def hybrid_score(
    bm25_score: float, semantic_score: float, alpha: float = ALPHA
) -> float:
    """Calculate weighted hybrid score from BM25 and semantic scores."""
    return alpha * bm25_score + (1 - alpha) * semantic_score


def normalise(numbers: list[float]) -> list[float] | list[int] | None:
    if not numbers:
        return

    min_value = min(numbers)
    max_value = max(numbers)

    if min_value == max_value:
        normalised_scores = [1 for val in numbers]
    else:
        normalised_scores = [
            (value - min_value) / (max_value - min_value) for value in numbers
        ]
    return normalised_scores


def weighted_search_command(query: str, alpha: float, limit: int = 5) -> None:
    """Execute weighted hybrid search and print results."""
    docs = load_movies()
    hybrid_search = HybridSearch(docs)
    results = hybrid_search.weighted_search(query, alpha, limit)

    for i, result in enumerate(results):
        print(f"{i + 1}. {result.doc.title}")
        print(f"   Hybrid Score: {result.hybrid_score:.3f}")
        print(f"   BM25: {result.bm25:.3f}")
        print("   " + result.doc.description[:100] + "...")
        print("\n")


def _enhance_query(query: str, enhance_method: Optional[str]) -> str:
    """Enhance the query using the specified method."""
    if not enhance_method:
        return query

    enhanced = improve_query(query, method=enhance_method)
    return enhanced if enhanced else query


def _add_llm_scores(
    query: str, results: list[RRFSearchResult]
) -> list[RRFSearchResult]:
    """Add LLM scores to each result."""
    for result in results:
        result.llm_score = score_movie(query=query, doc=result.doc.to_dict())
    return results


def _rerank_results(
    query: str, results: list[RRFSearchResult], method: str
) -> list[RRFSearchResult]:
    """Rerank results using the specified method."""
    if method == "individual":
        return sorted(results, key=lambda x: x.llm_score or 0, reverse=True)
    elif method == "batch":
        results_dicts = [
            {"doc": r.doc.to_dict(), "rrf_score": r.rrf_score} for r in results
        ]
        ranks = rerank_batch_prompt(query, results_dicts)
        return sorted(results, key=lambda x: ranks.index(x.doc.id))
    elif method == "cross_encoder":
        pairs = []
        for result in results:
            pairs.append([query, f"{result.doc.title} - {result.doc.description}"])
        cross_encoder = CrossEncoder("cross-encoder/ms-marco-TinyBERT-L2-v2")
        scores = cross_encoder.predict(pairs)

        for result, score in zip(results, scores):
            result.llm_score = str(float(score))

        return sorted(results, key=lambda x: float(x.llm_score or 0), reverse=True)

    return results


def print_rrf_results(
    results: list[RRFSearchResult],
    rerank_method: Optional[str],
    query: Optional[str] = None,
    k: Optional[int] = None,
) -> None:
    """Print RRF search results in a formatted manner."""
    if query and k is not None:
        print(f"\nReciprocal Rank Fusion Results for '{query}' (k={k}):")

    for i, result in enumerate(results):
        print(f"{i + 1}. {result.doc.title}")

        if rerank_method == "individual" and result.llm_score is not None:
            print(f"    Rerank Score: {int(result.llm_score):.3f}/10")
        elif rerank_method == "batch":
            print(f"    Rerank Rank: {i + 1}")
        elif rerank_method == "cross_encoder" and result.llm_score is not None:
            print(f"   Cross Encoder Score: {float(result.llm_score):.3f}")

        print(f"   RRF Score: {result.rrf_score:.3f}")
        print(
            f"   BM25 Rank: {result.bm25_rank if result.bm25_rank is not None else 'N/A'}, "
            f"Semantic Rank: {result.semantic_rank if result.semantic_rank is not None else 'N/A'}"
        )
        print("   " + result.doc.description[:100] + "...")
        print()


def rrf_search_command(
    query: str,
    k: int = 60,
    enhance: Optional[str] = None,
    rerank_method: Optional[str] = None,
    limit: int = 5,
) -> list[RRFSearchResult]:
    """Execute RRF search with optional query enhancement and reranking.

    Returns:
        List of RRF search results.
    """
    docs = load_movies()
    hybrid_search = HybridSearch(docs)

    search_limit = (
        limit * 5 if rerank_method in ("individual", "cross_encoder") else limit
    )

    query = _enhance_query(query, enhance)
    results = hybrid_search.rrf_search(query, k, search_limit)

    if rerank_method:
        if rerank_method != "cross_encoder":
            results = _add_llm_scores(query, results)
        results = _rerank_results(query, results, rerank_method)
        results = results[:limit]

    return results
