#!/usr/bin/env python3

import argparse

from lib.inverted_index import InvertedIndex, idf_command
from lib.keyword_search import tokenise_text


def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")

    subparsers.add_parser("build", help="Builds and saves the index and docmap to disk")

    tf_parser = subparsers.add_parser(
        "tf", help="Returns the term frequency for a specific term"
    )
    tf_parser.add_argument("doc_id", type=int, help="Document ID")
    tf_parser.add_argument("term", type=str, help="Term to seach")

    idf_parser = subparsers.add_parser("idf", help="Calculate the idf for a given term")
    idf_parser.add_argument("term", help="Term to lookup")

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

        case "tf":
            doc_id = args.doc_id
            term = args.term
            index = InvertedIndex()
            index.load()
            term_frequency = index.get_tf(doc_id, term)
            print(term_frequency)

        case "idf":
            print("Calculatiing IDF")
            idf_score = idf_command(args.term)
            print(f"Inverse document frequency of '{args.term}': {idf_score:.2f}")
            
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
