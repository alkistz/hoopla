#!/usr/bin/env python3

import argparse

from lib.inverted_index import InvertedIndex
from lib.keyword_search import tokenise_text


def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")

    subparsers.add_parser("build", help="Builds and saves the index and docmap to disk")

    args = parser.parse_args()

    match args.command:
        case "search":
            query = args.query
            print(f"Searching for: {query}")
            index = InvertedIndex()
            index.load()
            query_tokens = tokenise_text(query)
            results = []
            for token in query_tokens:
                print(token)
                matches = index.get_documents(token)
                results.extend(matches)
                if len(results) >= 5:
                    for result in results[:5]:
                        print("Title: " + result["title"])
                        print("ID: " + str(result["id"]))
                    

        case "build":
            print("Building the index")
            inverted_index = InvertedIndex()
            inverted_index.build()
            inverted_index.save()
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
