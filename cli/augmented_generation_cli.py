import argparse

from lib.hybrid_search import rrf_search_command
from lib.llm_setup import rag_response


def main():
    parser = argparse.ArgumentParser(description="Retrieval Augmented Generation CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    rag_parser = subparsers.add_parser(
        "rag", help="Perform RAG (search + generate answer)"
    )
    rag_parser.add_argument("query", type=str, help="Search query for RAG")

    args = parser.parse_args()

    match args.command:
        case "rag":
            query = args.query
            results = rrf_search_command(query)
            docs = [result.doc for result in results]
            response = rag_response(query, docs)

            print("Search Results")
            for result in results:
                print(f"- {result.doc.title}")
            print()

            print("RAG Response:")
            print(response)

        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
