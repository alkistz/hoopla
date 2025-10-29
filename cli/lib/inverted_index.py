import os
import pickle
from collections import Counter, defaultdict

from .keyword_search import tokenise_text
from .search_utils import CACHE_DIR, load_movies


class InvertedIndex:
    def __init__(self):
        self.index: dict[str, set] = defaultdict(set)
        self.docmap: dict[int, dict] = {}
        self.term_frequencies: dict = defaultdict(Counter)
        self.index_path = os.path.join(CACHE_DIR, "index.pkl")
        self.docmap_path = os.path.join(CACHE_DIR, "docmap.pkl")
        self.term_frequencies_path = os.path.join(CACHE_DIR, "term_frequencies.pkl")

    def get_documents(self, term: str):
        """
        Get the set of documents for a given token, and return them as a list, sorted in ascending order by document ID
        """
        doc_ids = self.index[term.lower()]
        return sorted(
            [self.docmap[doc_id] for doc_id in doc_ids], key=lambda x: x["id"]
        )

    def get_tf(self, doc_id: str, term: str) -> int:
        """
        Return the times the token appears in the document with the given ID. If the term doesn't exist in that document, return 0.
        Be sure to tokenize the term, but assume that there is only one token. If there's more than one, raise an exception.
        """
        tokenised_term = tokenise_text(term)
        if len(tokenised_term) != 1:
            raise ValueError("More than one tokenised term")

        token = tokenised_term[0]

        if token not in self.term_frequencies[int(doc_id)]:
            return 0

        return self.term_frequencies[int(doc_id)].get(token, 0)

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
        self.__write_to_file(self.term_frequencies_path, self.term_frequencies)

    def load(self):
        if not os.path.exists(self.index_path):
            raise FileExistsError("Index pickle does not exist")

        if not os.path.exists(self.docmap_path):
            raise FileExistsError("Docmap pickle does not exist")

        if not os.path.exists(self.term_frequencies_path):
            raise FileExistsError("Docmap pickle does not exist")

        with open(self.index_path, "rb") as file:
            self.index = pickle.load(file)
            print("Index loaded")

        with open(self.docmap_path, "rb") as file:
            self.docmap = pickle.load(file)
            print("docmap loaded")

        with open(self.term_frequencies_path, "rb") as file:
            self.term_frequencies = pickle.load(file)
            print("tf loaded")

    def __add_document(self, doc_id: int, text: str):
        """
        Tokenize the input text, then add each token to the index with the document ID.
        """
        tokens = tokenise_text(text)
        self.term_frequencies[doc_id].update(tokens)
        for token in tokens:
            self.index[token].add(doc_id)

    def __write_to_file(self, pickle_file, data):
        with open(pickle_file, "wb") as file:
            pickle.dump(data, file)
