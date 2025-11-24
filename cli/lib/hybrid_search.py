import os

from cli.lib.search_utils import ALPHA

from .inverted_index import InvertedIndex
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
        bm25_results = self._bm25_search(query, limit * 500)
        self.semantic_search.search_chunks(query, limit=limit * 500)
        
        scores = {}

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
