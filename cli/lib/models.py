from dataclasses import dataclass, field
from typing import Optional


@dataclass
class TestCase:
    query: str
    relevant_docs: list[str]


@dataclass
class EvaluationResult:
    query: str
    retrieved: list[str] = field(default_factory=list)
    relevant: list[str] = field(default_factory=list)
    k: Optional[int] = None

    @property
    def relevant_len(self):
        return len(self.relevant)

    @property
    def retrieved_len(self):
        return len(self.retrieved)

    @property
    def relevant_retrieved_len(self):
        return len(set(self.retrieved).intersection(set(self.relevant)))

    @property
    def precision(self):
        denominator = self.k if self.k is not None else self.retrieved_len
        if denominator == 0:
            return 0.0
        return self.relevant_retrieved_len / denominator

    @property
    def recall(self):
        if self.relevant_len == 0:
            return 0.0
        return self.relevant_retrieved_len / self.relevant_len

    @property
    def f1_score(self):
        return 2 * (self.precision * self.recall) / (self.precision + self.recall)



