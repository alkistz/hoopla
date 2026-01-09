import numpy as np
from PIL import Image
from sentence_transformers import SentenceTransformer

from .search_utils import load_movies


class MultimodalSearch:
    def __init__(self, model_name="clip-ViT-B-32", documents=None):
        self.model = SentenceTransformer(model_name)
        self.documents = documents or []

        # Create texts by concatenating title and description
        self.texts = [f"{doc['title']}: {doc['description']}" for doc in self.documents]

        # Generate embeddings for all texts
        if self.texts:
            self.text_embeddings = self.model.encode(self.texts, show_progress_bar=True)
        else:
            self.text_embeddings = []

    def embed_image(self, image_path: str):
        """Generate an embedding for an image at the given path.

        Args:
            image_path: Path to the image file

        Returns:
            The embedding vector for the image
        """
        image = Image.open(image_path)
        embedding = self.model.encode([image])[0]  # type: ignore
        return embedding

    def search_with_image(self, image_path: str):
        """Search for similar documents using an image.

        Args:
            image_path: Path to the image file

        Returns:
            List of top 5 matching documents with similarity scores
        """
        # Generate embedding for the image
        image_embedding = self.embed_image(image_path)

        # Calculate cosine similarity with all text embeddings
        results = []
        for i, text_embedding in enumerate(self.text_embeddings):
            # Cosine similarity
            similarity = np.dot(image_embedding, text_embedding) / (
                np.linalg.norm(image_embedding) * np.linalg.norm(text_embedding)
            )

            doc = self.documents[i]
            results.append(
                {
                    "id": doc["id"],
                    "title": doc["title"],
                    "description": doc["description"],
                    "similarity": float(similarity),
                }
            )

        # Sort by similarity (descending) and return top 5
        results.sort(key=lambda x: x["similarity"], reverse=True)
        return results[:5]


def verify_image_embedding(image_path: str):
    """Generate an embedding for an image and print its shape.

    Args:
        image_path: Path to the image file
    """
    search = MultimodalSearch()
    embedding = search.embed_image(image_path)
    print(f"Embedding shape: {embedding.shape[0]} dimensions")


def image_search_command(image_path: str):
    """Search for movies using an image.

    Args:
        image_path: Path to the image file

    Returns:
        List of matching movies with similarity scores
    """
    # Load the movie dataset using the utility function
    movies = load_movies()

    # Create MultimodalSearch instance with the movie documents
    search = MultimodalSearch(documents=movies)

    # Perform the search
    results = search.search_with_image(image_path)

    return results
