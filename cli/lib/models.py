from dataclasses import dataclass


@dataclass
class TestCase:
    query: str
    relevant_docs: list[str]
