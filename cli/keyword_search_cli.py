#!/usr/bin/env python3

import argparse

from lib.keyword_search import search_titles


def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")

    args = parser.parse_args()

    match args.command:
        case "search":
            query = args.query
            print(f"Searching for: {query}")
            search_result = search_titles(query)
            for i, movie in enumerate(search_result):
                print(f"{i + 1}. {movie['title']}")
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
