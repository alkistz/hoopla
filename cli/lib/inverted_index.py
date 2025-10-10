import os
import pickle
from collections import defaultdict

from .keyword_search import tokenise_text
from .search_utils import CACHE_DIR, load_movies


class InvertedIndex:
    def __init__(self):
        self.index: dict[str, set] = defaultdict(set)
        self.docmap: dict[int, dict] = {}
        self.index_path = os.path.join(CACHE_DIR, "index.pkl")
        self.docmap_path = os.path.join(CACHE_DIR, "docmap.pkl")

    def __add_document(self, doc_id: int, text: str):
        """
        Tokenize the input text, then add each token to the index with the document ID.
        """
        tokens = tokenise_text(text)
        for token in tokens:
            self.index[token].add(doc_id)

    def get_documents(self, term: str):
        """
        Get the set of documents for a given token, and return them as a list, sorted in ascending order by document ID
        """
        doc_ids = self.index[term.lower()]
        return sorted(
            [self.docmap[doc_id] for doc_id in doc_ids], key=lambda x: x["id"]
        )

    def build(self):
        """
        Iterate over all the movies and add them to both the index and the docmap.
        """
        movies = load_movies()
        for movie in movies:
            self.docmap[movie["id"]] = movie
            self.__add_document(
                doc_id=movie["id"], text=f"{movie['title']} {movie['description']}"
            )

    def save(self):
        """
        Save the index and docmap attributes to disk using the pickle module's dump function
        """
        os.makedirs(CACHE_DIR, exist_ok=True)
        self.__write_to_file(self.index_path, self.index)
        self.__write_to_file(self.docmap_path, self.docmap)

    def __write_to_file(self, pickle_file, data):
        with open(pickle_file, "wb") as file:
            pickle.dump(data, file)

    def load(self):
        if not os.path.exists(self.index_path):
            raise FileExistsError("Index pickle does not exist")

        if not os.path.exists(self.docmap_path):
            raise FileExistsError("Docmap pickle does not exist")

        # with open(self.index_path, 'rb') as file:
        #     self.index = pickle.load()