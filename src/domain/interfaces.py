# src/domain/interfaces.py

from __future__ import annotations
from typing import Protocol, Dict, List
import numpy as np

from src.schema import DocScore


class Retriever(Protocol):
    """A generic retriever that supports both text and vector lookup."""
    def search(self, query: str, k: int) -> List[DocScore]:
        """Return the top‐k docs for the given query string."""
        ...

    def doc_text(self, doc_id: str) -> str:
        """Fetch the original text of a document by its ID."""
        ...

    def search_by_vector(self, q_vec: np.ndarray, k: int) -> List[DocScore]:
        """Return the top‐k docs for the given query embedding."""
        ...

    def rerank_subset(self, q_vec: np.ndarray, subset_doc_vecs: Dict[str, np.ndarray], k: int) -> List[DocScore]:
        """Rerank a subset of documents based on their embeddings and a query vector."""
        ...


class FeedbackService(Protocol):
    def refine(
            self,
            qid: str,
            q_vec: np.ndarray,
            doc_vecs: Dict[str, np.ndarray]
    ) -> np.ndarray:
        """Apply relevance feedback (e.g. Rocchio) to produce a new query vector."""
        ...


class Evaluator(Protocol):
    """Score a run against provided qrels and return aggregate metrics."""
    def evaluate(
            self,
            run: Dict[str, Dict[str, float]],
            qrels: Dict[str, Dict[str, int]]
    ) -> Dict[str, Dict[str, float]]:
        ...
