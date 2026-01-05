import argparse
import json

from lib.hybrid_search import rrf_search_command
from lib.models import TestCase


def main():
    parser = argparse.ArgumentParser(description="Search Evaluation CLI")
    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Number of results to evaluate (k for precision@k, recall@k)",
    )

    args = parser.parse_args()
    limit = args.limit

    # run evaluation logic here
    with open(r"data/golden_dataset.json") as f:
        golden_dataset = json.load(f)

    test_cases: list[TestCase] = []
    for test_case in golden_dataset["test_cases"]:
        test_cases.append(
            TestCase(query=test_case["query"], relevant_docs=test_case["relevant_docs"])
        )

    print(f"k={limit}\n")

    for test_case in test_cases:
        results = rrf_search_command(query=test_case.query, k=60, limit=limit)

        result_titles = [result.doc.title for result in results]
        golden_titles = test_case.relevant_docs

        result_set = set(result_titles)
        golden_set = set(golden_titles)
        relevant_retrieved = len(result_set.intersection(golden_set))

        precision = relevant_retrieved / limit if limit > 0 else 0.0

        print(f"- Query: {test_case.query}")
        print(f"  - Precision@{limit}: {precision:.4f}")
        print(f"  - Retrieved: {', '.join(result_titles)}")
        print(f"  - Relevant: {', '.join(golden_titles)}")
        print()


if __name__ == "__main__":
    main()
