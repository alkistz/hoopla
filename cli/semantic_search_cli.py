#!/usr/bin/env python3

import argparse

from lib.semantic_search import (
    chunk_command,
    embed_chunks_command,
    embed_query_text,
    embed_text,
    search_chunked_command,
    search_command,
    semantic_chunk,
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

    search_parser = subparsers.add_parser(
        "search", help="Returns semantic search results"
    )
    search_parser.add_argument("query", type=str, help="Query to search against")
    search_parser.add_argument("--limit", type=int, default=5)

    chunk_parser = subparsers.add_parser("chunk", help="Chunks the documents")
    chunk_parser.add_argument("text", type=str, help="Text to chunk")
    chunk_parser.add_argument(
        "--chunk-size", type=int, default=200, help="The size of the chunk"
    )
    chunk_parser.add_argument("--overlap", type=int, default=0, help="Words to overlap")

    semantic_chunk_parser = subparsers.add_parser(
        "semantic_chunk", help="Does chunking based on sentences"
    )
    semantic_chunk_parser.add_argument("text", type=str, help="Text to chunk")
    semantic_chunk_parser.add_argument(
        "--max-chunk-size", type=int, default=4, help="The size of the chunk"
    )
    semantic_chunk_parser.add_argument(
        "--overlap", type=int, default=0, help="Words to overlap"
    )

    subparsers.add_parser(
        "embed_chunks", help="Create chunk embeddings for all the movies"
    )

    search_chunks_parser = subparsers.add_parser(
        "search_chunked", help="Semantic search with chunks"
    )
    search_chunks_parser.add_argument("query", type=str, help="query to search against")
    search_chunks_parser.add_argument(
        "--limit", type=int, default=5, help="How many results would you like?"
    )

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
        case "search":
            search_command(args.query, args.limit)
        case "chunk":
            chunk_command(args.text, args.chunk_size, args.overlap)
        case "semantic_chunk":
            semantic_chunk(args.text, args.max_chunk_size, args.overlap)
        case "embed_chunks":
            embed_chunks_command()
        case "search_chunked":
            search_chunked_command(args.query, args.limit)
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
