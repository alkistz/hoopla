import json
import os
import re

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

    def search(self, query: str, limit: int):
        if self.embeddings is None or self.documents is None:
            raise ValueError(
                "No embeddings loaded. Call `load_or_create_embeddings` first."
            )

        query_embedding = self.generate_embedding(query)
        similarity = [
            (cosine_similarity(query_embedding, embedding), self.documents[i])
            for i, embedding in enumerate(self.embeddings)
        ]
        sorted_similarity = sorted(similarity, key=lambda x: x[0], reverse=True)[:limit]
        results = []
        for result in sorted_similarity:
            results.append(
                {
                    "score": result[0],
                    "title": result[1]["title"],
                    "description": result[1]["description"],
                }
            )

        return results


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


def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)


def search_command(query: str, limit: int = 5):
    semantic_search = SemanticSearch()
    documents = load_movies()
    semantic_search.load_or_create_embeddings(documents)
    results = semantic_search.search(query, limit)
    for i, result in enumerate(results):
        print(f"{i + 1}. {result['title']} (score: {result['score']})")
        print(f"{result['description']}\n")


def chunk_command(text: str, chunk_size: int = 200, overlap: int = 0):
    words = text.rsplit()
    chunks = []

    for i in range(0, len(words), chunk_size):
        chunk = words[i : i + chunk_size]
        if i > 0 and overlap > 0:
            overlap_chunk = words[i - overlap : i]
            chunk = overlap_chunk + chunk
        chunks.append(" ".join(chunk))

    print(f"Chunking {len(text)} characters")
    for i in range(len(chunks)):
        print(f"{i + 1}. {chunks[i]}")


def semantic_chunk(text: str, max_chunk_size: int = 4, overlap: int = 0):
    sentences = re.split(r"(?<=[.!?])\s+", text)

    chunks = []
    for i in range(0, len(sentences), max_chunk_size - overlap):
        chunk_end = min(i + max_chunk_size, len(sentences))
        chunk = sentences[i:chunk_end]
        chunks.append(" ".join(chunk))

        if chunk_end >= len(sentences):
            break

    return chunks


class ChunkedSemanticSearch(SemanticSearch):
    def __init__(self, model_name="all-MiniLM-L6-v2") -> None:
        super().__init__()
        if model_name != "all-MiniLM-L6-v2":
            self.model = SentenceTransformer(model_name)
        self.chunk_embeddings = None
        self.chunk_metadata = None
        self.chunk_embeddings_path = os.path.join(CACHE_DIR, "chunk_embeddings.npy")
        self.chunk_metadata_path = os.path.join(CACHE_DIR, "chunk_metadata.json")

    def build_chunk_embeddings(self, documents):
        self.documents = documents

        all_chunks = []
        chunk_metadata = []

        for doc_index, doc in enumerate(documents):
            if doc["description"] is None or doc["description"].strip() == "":
                continue
            self.document_map[doc["id"]] = doc

            chunks = semantic_chunk(doc["description"], 4, 1)
            all_chunks.extend(chunks)

            for chunk_index, chunk in enumerate(chunks):
                chunk_metadata.append(
                    {
                        "movie_idx": doc_index,
                        "chunk_idx": chunk_index,
                        "total_chunks": len(chunks),
                    }
                )

        self.chunk_metadata = chunk_metadata
        embeddings = self.model.encode(all_chunks)
        self.chunk_embeddings = embeddings
        np.save(self.chunk_embeddings_path, embeddings)
        
        with open(self.chunk_metadata_path, "w") as f:
            json.dump(
                {"chunks": chunk_metadata, "total_chunks": len(all_chunks)}, f, indent=2
            )

        return embeddings

    def load_or_create_chunk_embeddings(self, documents: list[dict]) -> np.ndarray:
        self.documents = documents
        self.document_map = {doc["id"]: doc for doc in documents}
        if os.path.exists(self.chunk_embeddings_path) and os.path.exists(
            self.chunk_metadata_path
        ):
            self.chunk_embeddings = np.load(self.chunk_embeddings_path)
            with open(self.chunk_metadata_path, "r") as f:
                metadata = json.load(f)
                self.chunk_metadata = metadata["chunks"]
            return self.chunk_embeddings

        return self.build_chunk_embeddings(documents)


def embed_chunks_command():
    documents = load_movies()
    chunked_semantic_search = ChunkedSemanticSearch()
    embeddings = chunked_semantic_search.load_or_create_chunk_embeddings(documents)
    print(f"Generated {len(embeddings)} chunked embeddings")
