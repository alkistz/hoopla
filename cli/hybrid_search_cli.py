import argparse

from lib.hybrid_search import (
    print_rrf_results,
    rrf_search_command,
    weighted_search_command,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid Search CLI")
    subparser = parser.add_subparsers(dest="command", help="Available commands")

    normalize_parser = subparser.add_parser(
        "normalize", help="Normalise a list of numbers"
    )
    normalize_parser.add_argument(
        "numbers", type=float, nargs="+", help="A list of numbers separated by space"
    )

    weigthed_search_parser = subparser.add_parser(
        "weighted-search",
        help="Performs a hybrid search combining keyword and semantic search",
    )
    weigthed_search_parser.add_argument("query", type=str, help="Search query")
    weigthed_search_parser.add_argument(
        "--alpha",
        type=float,
        default=0.5,
        help="The weight between keyword and semantic scores",
    )
    weigthed_search_parser.add_argument(
        "--limit", type=int, default=5, help="Search results limit"
    )

    rrf_search_parser = subparser.add_parser(
        "rrf-search", help="Hynrid search with Reciprocal Rank Fusion"
    )
    rrf_search_parser.add_argument("query", type=str, help="Search query")
    rrf_search_parser.add_argument("--k", type=int, default=60, help="K parameter")
    rrf_search_parser.add_argument(
        "--limit", type=int, default=5, help="Number of search results returned."
    )

    rrf_search_parser.add_argument(
        "--enhance",
        type=str,
        choices=["spell", "rewrite", "expand"],
        help="Query enhancement method",
    )

    rrf_search_parser.add_argument(
        "--rerank-method",
        type=str,
        choices=["individual", "batch", "cross_encoder"],
        help="LLM based re-ranking",
    )

    rrf_search_parser.add_argument(
        "--evaluate", action="store_true", help="Use an LLM to judge the results"
    )

    args = parser.parse_args()

    match args.command:
        case "normalize":
            numbers = args.numbers
            if not numbers:
                return

            min_value = min(numbers)
            max_value = max(numbers)

            if min_value == max_value:
                normalised_scores = [1 for val in numbers]
            else:
                normalised_scores = [
                    (value - min_value) / (max_value - min_value) for value in numbers
                ]
            for score in normalised_scores:
                print(f"* {score:.4f}")

        case "weighted-search":
            weighted_search_command(args.query, args.alpha, args.limit)
        case "rrf-search":
            # Print reranking message if using cross_encoder
            if args.rerank_method == "cross_encoder":
                initial_limit = args.limit * 5
                print(
                    f"\nReranking top {initial_limit} results using {args.rerank_method} method..."
                )

            results = rrf_search_command(
                args.query, args.k, args.enhance, args.rerank_method, args.limit
            )

            if args.evaluate:
                from lib.llm_setup import evaluate_results

                # Format results for LLM evaluation
                formatted_results = [
                    f"{i + 1}. {result.doc.title} - {result.doc.description[:200]}"
                    for i, result in enumerate(results)
                ]

                # Get LLM scores (returns list of 0-3 scores in same order)
                scores = evaluate_results(args.query, formatted_results)

                # Attach scores to results and sort by score (descending)
                for result, score in zip(results, scores):
                    result.llm_score = str(score)

                results = sorted(
                    results,
                    key=lambda x: int(x.llm_score) if x.llm_score else 0,
                    reverse=True,
                )

                print("\n" + "=" * 80)
                print(f"LLM Evaluation Results for '{args.query}'")
                print("=" * 80)
                for i, result in enumerate(results, 1):
                    print(
                        f"{i}. {result.doc.title} - Relevance Score: {result.llm_score}/3"
                    )
                print("=" * 80 + "\n")

            print_rrf_results(results, args.rerank_method, args.query, args.k)
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
