"""Abstract contracts so upper layers depend only on these."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Protocol, Dict, List
import numpy as np

from src.schema import DocScore


class Retriever(Protocol):
    def search(self, query: str, k: int) -> List[DocScore]:
        """Return *k* documents ranked for *query*."""

class FeedbackService(Protocol):
    def refine(self, qid: str, q_vec: np.ndarray, doc_vecs: Dict[str, np.ndarray]) -> np.ndarray: ...

class Evaluator(Protocol):
    def evaluate(self, run: Dict[str, Dict[str, float]], qrels: Dict[str, Dict[str, int]]) -> Dict[str, float]: ...