import string

from nltk.stem import PorterStemmer

from .search_utils import load_movies, load_stop_wrods


def search_titles(query: str) -> list[dict]:
    movies = load_movies()
    search_result = [
        movie for movie in movies if check_for_match(query, movie["title"])
    ]
    search_result_sorted = sorted(search_result, key=lambda x: x["id"])
    return search_result_sorted[:5]


def remove_punctuation(input_string: str) -> str:
    table = str.maketrans("", "", string.punctuation)
    result = input_string.translate(table)
    return result


def preprocess_string(input_string: str) -> str:
    return remove_punctuation(input_string.lower())


def remove_stop_words(tokens: list[str]) -> list[str]:
    stop_words = set(load_stop_wrods())
    return [token for token in tokens if token not in stop_words]


def tokenise_text(input_text: str) -> list[str]:
    stemmer = PorterStemmer()

    tokens = preprocess_string(input_text).split()
    tokens = remove_stop_words(tokens)
    tokens = [stemmer.stem(token) for token in tokens]

    return tokens


def check_for_match(query: str, movie_title: str) -> bool:
    query_tokens = tokenise_text(query)
    movie_tokens = tokenise_text(movie_title)

    for query_token in query_tokens:
        for movie_token in movie_tokens:
            if query_token in movie_token:
                return True

    return False


