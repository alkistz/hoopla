import argparse

from lib.hybrid_search import rrf_search_command
from lib.llm_setup import (
    answer_question,
    answer_with_citations,
    rag_response,
    summarize_results,
)


def main():
    parser = argparse.ArgumentParser(description="Retrieval Augmented Generation CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    rag_parser = subparsers.add_parser(
        "rag", help="Perform RAG (search + generate answer)"
    )
    rag_parser.add_argument("query", type=str, help="Search query for RAG")

    summarize_parser = subparsers.add_parser(
        "summarize", help="Summarizes the results of a query"
    )
    summarize_parser.add_argument(
        "query", type=str, help="Search query for summarisation"
    )
    summarize_parser.add_argument(
        "--limit", type=int, default=5, help="The limit of the search results"
    )

    citation_parser = subparsers.add_parser(
        "citations", help="Summarise query results "
    )
    citation_parser.add_argument(
        "query", type=str, help="Search query for summarisation with citations"
    )
    citation_parser.add_argument(
        "--limit", type=int, default=5, help="The limit of the search results"
    )

    question_parser = subparsers.add_parser(
        "question", help="Answers the users question"
    )
    question_parser.add_argument("question", type=str, help="question to answer")
    question_parser.add_argument(
        "--limit", type=int, default=5, help="The limit of the search results"
    )

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
        case "summarize":
            results = rrf_search_command(args.query, limit=args.limit)
            docs = [result.doc for result in results]
            response = summarize_results(args.query, docs)

            print("Search Results")
            for result in results:
                print(f"- {result.doc.title}")
            print()

            print("LLM Summary")
            print(response)

        case "citations":
            results = rrf_search_command(args.query, limit=args.limit)
            docs = [result.doc for result in results]
            response = answer_with_citations(args.query, docs)

            print("Search Results")
            for result in results:
                print(f"- {result.doc.title}")
            print()

            print("LLM Answer")
            print(response)

        case "question":
            results = rrf_search_command(args.question, limit=args.limit)
            docs = [result.doc for result in results]
            response = answer_question(args.question, docs)

            print("Search Results")
            for result in results:
                print(f"- {result.doc.title}")
            print()

            print("Answer")
            print(response)

        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
