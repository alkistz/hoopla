import os
from typing import Optional

from .inverted_index import InvertedIndex
from .llm_setup import improve_query, rerank_batch_prompt, score_movie
from .search_utils import ALPHA, load_movies
from .semantic_search import ChunkedSemanticSearch


class HybridSearch:
    def __init__(self, documents):
        self.documents = documents
        self.semantic_search = ChunkedSemanticSearch()
        self.semantic_search.load_or_create_chunk_embeddings(documents)

        self.idx = InvertedIndex()
        if not os.path.exists(self.idx.index_path):
            self.idx.build()
            self.idx.save()

    def _bm25_search(self, query, limit):
        self.idx.load()
        return self.idx.bm25_search(query, limit)

    def weighted_search(self, query, alpha, limit=5):
        final_results = {}

        bm25_results = self._bm25_search(query, limit * 500)
        bm25_scores = normalise([r["score"] for r in bm25_results])

        if not bm25_scores:
            raise ValueError("No bm25 scores were provided!")

        for i, doc in enumerate(bm25_results):
            doc_id = doc["id"]
            if not final_results.get(doc_id):
                final_results[doc_id] = {"doc": doc, "bm25": 0, "semantic": 0}
            final_results[doc_id]["bm25"] = bm25_scores[i]

        semantic_results = self.semantic_search.search_chunks(query, limit=limit * 500)
        semantic_scores = normalise([r["score"] for r in semantic_results])

        if not semantic_scores:
            raise ValueError("No semantic scores were provided!")

        for i, doc in enumerate(semantic_results):
            doc_id = doc["id"]
            if not final_results.get(doc_id):
                final_results[doc_id] = {"doc": doc, "bm25": 0, "semantic": 0}
            final_results[doc_id]["semantic"] = semantic_scores[i]

        for doc_id in final_results:
            bm25_score = final_results[doc_id].get("bm25", 0)
            semantic_score = final_results[doc_id].get("semantic", 0)
            final_results[doc_id]["hybrid_score"] = hybrid_score(
                bm25_score, semantic_score, alpha
            )
        sorted_results = sorted(
            final_results.values(), key=lambda x: x["hybrid_score"], reverse=True
        )

        return sorted_results[:limit]

    def rrf_search(self, query, k, limit=10):
        final_results = {}
        bm25_results = self._bm25_search(query, limit * 500)
        semantic_results = self.semantic_search.search_chunks(query, limit=limit * 500)

        for i, doc in enumerate(bm25_results):
            doc_id = doc["id"]
            rrf_score = calculate_rrf_score(i, k)
            if final_results.get(doc_id):
                final_results[doc_id]["rrf_score"] = (
                    final_results[doc_id]["rrf_score"] + rrf_score
                )
                final_results[doc_id]["bm25_rank"] = i
            else:
                final_results[doc_id] = {
                    "doc": doc,
                    "rrf_score": rrf_score,
                    "bm25_rank": i,
                }

        for i, doc in enumerate(semantic_results):
            doc_id = doc["id"]
            rrf_score = calculate_rrf_score(i, k)
            if final_results.get(doc_id):
                final_results[doc_id]["rrf_score"] = (
                    final_results[doc_id]["rrf_score"] + rrf_score
                )
                final_results[doc_id]["semantic_rank"] = i
            else:
                final_results[doc_id] = {
                    "doc": doc,
                    "rrf_score": rrf_score,
                    "semantic_rank": i,
                }

        sorted_results = sorted(
            final_results.values(), key=lambda x: x["rrf_score"], reverse=True
        )

        return sorted_results[:limit]


def calculate_rrf_score(rank, k=60):
    return 1 / (k + rank)


def hybrid_score(bm25_score, semantic_score, alpha=ALPHA):
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


def weighted_search_command(query, alpha, limit=5):
    docs = load_movies()
    hybrid_search = HybridSearch(docs)
    results = hybrid_search.weighted_search(query, alpha, limit)

    for i, result in enumerate(results):
        print(f"{i + 1}. {result['doc']['title']}")
        print(f"   Hybrid Score: {result['hybrid_score']:.3f}")
        print(f"   BM25: {result['bm25']:.3f}")
        print("   " + result["doc"]["description"][:100] + "...")
        print("\n")


def _enhance_query(query: str, enhance_method: Optional[str]) -> str:
    """Enhance the query using the specified method."""
    if not enhance_method:
        return query

    enhanced = improve_query(query, method=enhance_method)
    return enhanced if enhanced else query


def _add_llm_scores(query: str, results: list) -> list:
    """Add LLM scores to each result."""
    for result in results:
        result["llm_score"] = score_movie(query=query, doc=result["doc"])
    return results


def _rerank_results(query: str, results: list, method: str) -> list:
    """Rerank results using the specified method."""
    if method == "individual":
        return sorted(results, key=lambda x: x["llm_score"], reverse=True)
    elif method == "batch":
        ranks = rerank_batch_prompt(query, results)
        return sorted(results, key=lambda x: ranks.index(str(x["doc"]["id"])))
    
    return results


def _print_rrf_results(results: list, rerank_method: Optional[str]):
    """Print RRF search results in a formatted manner."""
    for i, result in enumerate(results):
        print(f"{i + 1}. {result['doc']['title']}")

        if rerank_method == "individual":
            print(f"    Rerank Score: {int(result['llm_score']):.3f}/10")
        elif rerank_method == "batch":
            print(f"    Rerank Rank: {i + 1}")

        print(f"    RRF Score: {result['rrf_score']}")
        print(
            f"    BM25 Rank: {result.get('bm25_rank', 'N/A')}, "
            f"Semantic Rank: {result.get('semantic_rank', 'N/A')}"
        )
        print("   " + result["doc"]["description"][:100] + "...")
        print()


def rrf_search_command(
    query: str,
    k: int = 60,
    enhance: Optional[str] = None,
    rerank_method: Optional[str] = None,
    limit: int = 5,
):
    """Execute RRF search with optional query enhancement and reranking."""
    docs = load_movies()
    hybrid_search = HybridSearch(docs)

    search_limit = (
        limit * 5 if rerank_method in ("individual", "cross_encoder") else limit
    )

    query = _enhance_query(query, enhance)
    results = hybrid_search.rrf_search(query, k, search_limit)

    if rerank_method:
        results = _add_llm_scores(query, results)
        results = _rerank_results(query, results, rerank_method)
        results = results[:limit]

    _print_rrf_results(results, rerank_method)
