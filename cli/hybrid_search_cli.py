import argparse


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
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
