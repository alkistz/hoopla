from .search_utils import load_movies

def search_titles(query: str) -> list[dict]:
    movies = load_movies()
    search_result = [movie for movie in movies if query in movie["title"]]
    search_result_sorted = sorted(search_result, key=lambda x: x["id"])
    return search_result_sorted[:5]
