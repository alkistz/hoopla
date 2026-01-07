import argparse
import json

from lib.hybrid_search import rrf_search_command
from lib.models import EvaluationResult, TestCase


def print_evaluation_result(evaluation_result: EvaluationResult):
    print(f"- Query: {evaluation_result.query}")
    print(f"  - Precision@{evaluation_result.k}: {evaluation_result.precision:.4f}")
    print(f"  - Recall@{evaluation_result.k}: {evaluation_result.recall:.4f}")
    print(f"  - F1 Score: {evaluation_result.f1_score:.4f}")
    print(f"  - Retrieved: {', '.join(evaluation_result.retrieved)}")
    print(f"  - Relevant: {', '.join(evaluation_result.relevant)}")
    print()


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

        evaluation_result = EvaluationResult(
            query=test_case.query,
            retrieved=[result.doc.title for result in results],
            relevant=test_case.relevant_docs,
            k=limit,
        )

        print_evaluation_result(evaluation_result)


if __name__ == "__main__":
    main()
