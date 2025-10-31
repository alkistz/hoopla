#!/usr/bin/env python3

import argparse

from lib.semantic_search import (
    embed_query_text,
    embed_text,
    verify_embeddings,
    verify_model,
)


def main():
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    subparsers.add_parser(
        "verify",
        help="Verifies that the model is loaded and prints basic model info",
    )

    embed_text_parser = subparsers.add_parser(
        "embed_text",
        help="Prints the result of a text embedding",
    )
    embed_text_parser.add_argument("text", type=str, help="Text to embed")

    subparsers.add_parser(
        "verify_embeddings", help="Loads and  verifies the embeeddings"
    )

    embedquery_parser = subparsers.add_parser(
        "embedquery", help="Embeds a specific query"
    )
    embedquery_parser.add_argument("query", type=str, help="Query to embed")

    args = parser.parse_args()

    match args.command:
        case "verify":
            verify_model()
        case "embed_text":
            embed_text(args.text)
        case "verify_embeddings":
            verify_embeddings()
        case "embedquery":
            embed_query_text(args.query)
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
