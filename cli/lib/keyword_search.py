import string

from .search_utils import load_movies


def search_titles(query: str) -> list[dict]:
    movies = load_movies()
    search_result = [
        movie
        for movie in movies
        if preprocess_string(query) in preprocess_string(movie["title"])
    ]
    search_result_sorted = sorted(search_result, key=lambda x: x["id"])
    return search_result_sorted[:5]


def remove_punctuation(input_string: str) -> str:
    table = str.maketrans("", "", string.punctuation)
    result = input_string.translate(table)
    return result


def preprocess_string(input_string: str) -> str:
    return remove_punctuation(input_string.lower())
