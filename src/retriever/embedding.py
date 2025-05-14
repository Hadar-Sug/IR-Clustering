# src/retriever/embedding.py

import numpy as np
from typing import Dict, List
from sentence_transformers import SentenceTransformer
import faiss  # type: ignore

from ..domain.interfaces import Retriever
from ..schema import DocScore

class EmbeddingRetriever(Retriever):
    """Embedding-based retriever with FAISS, plus helper methods
    for true-feedback pipelines (doc_text, search_by_vector)."""

    def __init__(self, model_name: str, corpus: Dict[str, str]):
        # raw corpus for doc_text()
        self._corpus: Dict[str, str] = corpus.copy()

        # keep doc IDs in insertion order
        self.doc_ids: List[str] = list(corpus.keys())

        # 1) load encoder
        self.model = SentenceTransformer(model_name)

        # 2) encode all docs
        texts = list(corpus.values())
        emb = self.model.encode(
            texts,
            convert_to_numpy=True,
            show_progress_bar=False,
            normalize_embeddings=True
        )
        # ensure float32, contiguous
        emb = np.asarray(emb, dtype=np.float32)
        if not emb.flags["C_CONTIGUOUS"]:
            emb = np.ascontiguousarray(emb)

        # 3) build inner-product index
        dim = emb.shape[1]
        self._index = faiss.IndexFlatIP(dim)  # type: ignore
        self._index.add(emb)                  # type: ignore

    # ---------------------------------------------------------------- #
    # Pipeline-needed helpers
    # ---------------------------------------------------------------- #
    def doc_text(self, doc_id: str) -> str:
        """Return the original text for a given doc_id."""
        return self._corpus[doc_id]

    def search_by_vector(self, q_vec: np.ndarray, k: int) -> List[DocScore]:
        """Rank by cosine-sim (inner-product) against a precomputed query vector."""
        q = np.asarray(q_vec, dtype=np.float32).reshape(1, -1)
        distances, indices = self._index.search(q, k)  # type: ignore

        results: List[DocScore] = []
        for score, idx in zip(distances[0], indices[0]):
            if idx < 0:
                break
            results.append(DocScore(doc_id=self.doc_ids[idx], score=float(score)))
        return results

    # ---------------------------------------------------------------- #
    # Standard Retriever interface
    # ---------------------------------------------------------------- #
    def search(self, query: str, k: int) -> List[DocScore]:
        """Embed the query string, then proxy to search_by_vector."""
        q_emb = self.model.encode(
            query,
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        return self.search_by_vector(q_emb, k)
