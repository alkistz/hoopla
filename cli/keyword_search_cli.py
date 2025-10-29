#!/usr/bin/env python3

import argparse

from lib.inverted_index import (
    InvertedIndex,
    bm25_idf_command,
    bm25_tf_command,
    bm25search_command,
    idf_command,
    tf_command,
    tfidf_command,
    tokenise_text,
)
from lib.search_utils import BM25_B, BM25_K1


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

    tfidf_parser = subparsers.add_parser(
        "tfidf", help="Calculates the TF - IDF score for a given term and document"
    )
    tfidf_parser.add_argument("doc_id", help="Document ID")
    tfidf_parser.add_argument("term", help="Specific term")

    bm25_idf_parser = subparsers.add_parser(
        "bm25idf", help="Get BM25 IDF score for a given term"
    )
    bm25_idf_parser.add_argument(
        "term", type=str, help="Term to get BM25 IDF score for"
    )

    bm25_tf_parser = subparsers.add_parser(
        "bm25tf", help="Get BM25 TF score for a given document ID and term"
    )
    bm25_tf_parser.add_argument("doc_id", type=int, help="Document ID")
    bm25_tf_parser.add_argument("term", type=str, help="Term to get BM25 TF score for")
    bm25_tf_parser.add_argument(
        "k1", type=float, nargs="?", default=BM25_K1, help="Tunable BM25 K1 parameter"
    )
    bm25_tf_parser.add_argument(
        "b", type=float, nargs="?", default=BM25_B, help="Tunable BM25 b parameter"
    )

    bm25search_parser = subparsers.add_parser(
        "bm25search", help="Search movies using full BM25 scoring"
    )
    bm25search_parser.add_argument("query", type=str, help="Search query")
    bm25search_parser.add_argument(
        "limit", type=int, nargs="?", default=5, help="Limit of the search results"
    )

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
            print("Calculating TF")
            term_frequency = tf_command(args.doc_id, args.term)
            print(term_frequency)

        case "idf":
            print("Calculatiing IDF")
            idf_score = idf_command(args.term)
            print(f"Inverse document frequency of '{args.term}': {idf_score:.2f}")

        case "tfidf":
            tfidf_score = tfidf_command(args.doc_id, args.term)
            print(
                f"TF-IDF score of '{args.term}' in document '{args.doc_id}': {tfidf_score:.2f}"
            )

        case "bm25idf":
            print("Calculating BM25 IDF")
            bm25idf_score = bm25_idf_command(args.term)
            print(f"BM25 IDF score of '{args.term}': {bm25idf_score:.2f}")

        case "bm25tf":
            print("Calculating the BM25 TF score")
            bm25_tf_score = bm25_tf_command(args.doc_id, args.term, args.k1, args.b)
            print(
                f"BM25 TF score of '{args.term}' in document '{args.doc_id}': {bm25_tf_score:.2f}"
            )

        case "bm25search":
            results = bm25search_command(args.query, args.limit)
            for i, result in enumerate(results):
                print(
                    f"{i + 1}. ({result['id']}) {result['title']} - Score: {result['score']:.2f}"
                )

        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
