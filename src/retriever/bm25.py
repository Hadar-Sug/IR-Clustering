from __future__ import annotations
from typing import List, Sequence
from rank_bm25 import BM25Okapi
from ..domain.interfaces import Retriever   
from ..schema import DocScore              


class BM25Retriever(Retriever):
    """Pure-Python BM25 using rank_bm25 (no external index)."""
    def __init__(self, corpus: dict[str, str]):
        # save the IDs once for stable ordering / tie-breaking
        self.doc_ids: List[str] = list(corpus.keys())

        # very simple tokenizer; replace with nltk, spacy, etc. if you wish
        def tokenize(text: str) -> List[str]:
            return text.lower().split()

        # store for later use
        self._tokenize = tokenize

        tokenised_corpus: List[Sequence[str]] = [
            self._tokenize(doc) for doc in corpus.values()
        ]
        self._bm25 = BM25Okapi(tokenised_corpus)   # <-- consistent name

    # ------------------------------------------------------------------ #
    # Retriever interface
    # ------------------------------------------------------------------ #
    def search(self, query: str, k: int = 100) -> list[DocScore]:
        """Return up to *k* documents ranked by BM25 score."""
        # short-circuit empty / whitespace query
        if not query.strip():
            return []

        q_tokens = self._tokenize(query)
        scores   = self._bm25.get_scores(q_tokens)

        # tie-break deterministically on doc_id
        ranked = sorted(
            (DocScore(did, score) for did, score in zip(self.doc_ids, scores)),
            key=lambda x: (-x.score, x.doc_id)
        )
        return ranked[:k]
