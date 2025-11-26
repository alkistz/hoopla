import os

from .inverted_index import InvertedIndex
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

        return sorted_results[:5]

    def rrf_search(self, query, k, limit=10):
        raise NotImplementedError("RRF hybrid search is not implemented yet.")


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
