import os

import numpy as np
from sentence_transformers import SentenceTransformer

from .search_utils import CACHE_DIR, load_movies


class SemanticSearch:
    def __init__(self):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.embeddings = None
        self.documents = None
        self.document_map = {}
        self.movie_embeddings_cache_path = os.path.join(
            CACHE_DIR, "movie_embeddings.npy"
        )

    def generate_embedding(self, text: str):
        if not text.strip():
            raise ValueError("The text field is empty")

        return self.model.encode([text])[0]

    def build_embeddings(self, documents: list[dict]):
        self.documents = documents

        string_docs = []
        for doc in documents:
            self.document_map[doc["id"]] = doc
            string_docs.append(f"{doc['title']}: {doc['description']}")

        self.embeddings = self.model.encode(string_docs, show_progress_bar=True)
        np.save(self.movie_embeddings_cache_path, self.embeddings)

        return self.embeddings

    def load_or_create_embeddings(self, documents):
        self.documents = documents
        self.document_map = {doc["id"]: doc for doc in documents}
        if os.path.exists(self.movie_embeddings_cache_path):
            self.embeddings = np.load(self.movie_embeddings_cache_path)
            if len(self.embeddings) == len(self.documents):
                return self.embeddings

        return self.build_embeddings(documents)


def verify_model():
    semantic_search = SemanticSearch()
    print(f"Model loaded: {semantic_search.model}")
    print(f"Max sequence length: {semantic_search.model.max_seq_length}")


def embed_text(text: str):
    semantic_search = SemanticSearch()
    embedding = semantic_search.generate_embedding(text)
    print(f"Text: {text}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Dimensions: {embedding.shape[0]}")


def verify_embeddings():
    semantic_search = SemanticSearch()
    documents = load_movies()
    embeddings = semantic_search.load_or_create_embeddings(documents)
    print(f"Number of docs:   {len(documents)}")
    print(
        f"Embeddings shape: {embeddings.shape[0]} vectors in {embeddings.shape[1]} dimensions"
    )


def embed_query_text(query: str):
    sem_search = SemanticSearch()
    embedding = sem_search.generate_embedding(query)
    print(f"Query: {query}")
    print(f"First 5 dimensions: {embedding[:5]}")
    print(f"Shape: {embedding.shape}")
