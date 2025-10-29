import json
import os

DEFAULT_SEARCH_LIMIT = 5
BM25_K1 = 1.5
BM25_B = 0.75

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "movies.json")
STOP_WORDS_PATH = os.path.join(PROJECT_ROOT, "data", "stop_words.txt")

CACHE_DIR = os.path.join(PROJECT_ROOT, "cache")


def load_movies() -> list[dict]:
    with open(DATA_PATH, "r") as f:
        data = json.load(f)
    return data["movies"]


def load_stop_wrods():
    with open(STOP_WORDS_PATH, "r") as f:
        return f.read().splitlines()
    


