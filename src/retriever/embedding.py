import numpy as np
from typing import Dict, List, Optional
import os
import types

try:  # pragma: no cover - optional dependency
    import torch
except Exception:  # pragma: no cover - torch may be absent
    torch = types.SimpleNamespace(
        cuda=types.SimpleNamespace(is_available=lambda: False),
        float16="float16",
        float32="float32",
    )  # type: ignore[misc]

# Optional heavy dependencies -------------------------------------------------
try:  # pragma: no cover - optional dependency
    from sentence_transformers import SentenceTransformer  # type: ignore
except Exception:  # pragma: no cover - package may be absent in CI
    SentenceTransformer = None  # type: ignore[misc]

try:  # pragma: no cover - optional dependency
    import faiss  # type: ignore
except Exception:  # pragma: no cover - package may be absent in CI
    # Minimal stub so tests can monkeypatch FAISS classes/functions
    faiss = types.SimpleNamespace(
        IndexFlatIP=None,
        StandardGpuResources=None,
        index_cpu_to_gpu=lambda *a, **k: a[2] if len(a) > 2 else None,
        index_gpu_to_cpu=lambda x: x,
        write_index=lambda *a, **k: None,
        read_index=lambda *a, **k: None,
    )  # type: ignore[misc]

from ..domain.interfaces import Retriever
from ..schema import DocScore

class EmbeddingRetriever(Retriever):
    """Embedding-based retriever with FAISS, plus helper methods
    for true-feedback pipelines (doc_text, search_by_vector)."""

    def __init__(
        self, 
        model_name: str, 
        corpus: Dict[str, str], 
        index_path: Optional[str] = None,
        batch_size: int = 64, 
        use_fp16: bool = False,
    ):
        # raw corpus for doc_text()
        self._corpus: Dict[str, str] = corpus.copy()
        self.doc_ids: List[str] = list(corpus.keys())
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.index_path = index_path or "faiss.index"

        # 1) load encoder, send to device
        if SentenceTransformer is None:
            raise ImportError(
                "sentence_transformers is required for EmbeddingRetriever"
            )
        if not hasattr(faiss, "IndexFlatIP"):
            raise ImportError("faiss is required for EmbeddingRetriever")
        self.model = SentenceTransformer(model_name)
        if use_fp16 and self.device == "cuda":
            self.model = self.model.half()

        if os.path.isfile(self.index_path):
            # If the index exists, load it
            print(f"Loading FAISS index from {self.index_path}")
            index = faiss.read_index(self.index_path)
            if self.device == "cuda" and hasattr(faiss, "StandardGpuResources"):
                res = faiss.StandardGpuResources()
                index = faiss.index_cpu_to_gpu(res, 0, index)
            self._index = index
        else:
            # 2) encode all docs using GPU (if available)
            print("Building FAISS index")
            texts = list(corpus.values())
            emb = self.model.encode(
                texts,
                convert_to_numpy=True,
                normalize_embeddings=True,
            )
            # ensure float32 format for faiss, contiguous
            emb = np.asarray(emb, dtype=np.float32)
            if not emb.flags["C_CONTIGUOUS"]:
                emb = np.ascontiguousarray(emb)

            # 3) build inner-product index
            dim = emb.shape[1]
            cpu_index = faiss.IndexFlatIP(dim)  # type: ignore

            # GPU: move index to GPU if available
            if self.device == "cuda" and hasattr(faiss, "StandardGpuResources"):
                res = faiss.StandardGpuResources()
                self._index = faiss.index_cpu_to_gpu(res, 0, cpu_index)
            else:
                self._index = cpu_index

            self._index.add(emb)  # type: ignore

            # Save the index for future runs
            self.save_index(self.index_path)

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

    def rerank_subset(self, q_vec: np.ndarray, subset_doc_vecs: Dict[str, np.ndarray], k: int) -> List[DocScore]:
        """Rerank a subset of documents based on their embeddings and a query vector."""
        if not subset_doc_vecs:
            return []

        q_vec_norm = q_vec / np.linalg.norm(q_vec) if np.linalg.norm(q_vec) > 0 else q_vec

        scored_docs: List[DocScore] = []
        for doc_id, doc_vec in subset_doc_vecs.items():
            doc_vec_norm = doc_vec / np.linalg.norm(doc_vec) if np.linalg.norm(doc_vec) > 0 else doc_vec
            score = np.dot(q_vec_norm, doc_vec_norm).item()
            scored_docs.append(DocScore(doc_id=doc_id, score=float(score)))

        # Sort by score in descending order
        scored_docs.sort(key=lambda x: x.score, reverse=True)
        
        return scored_docs[:k]

    def search(self, query: str, k: int) -> List[DocScore]:
        """Embed the query string, then proxy to search_by_vector."""
        q_emb = self.model.encode(
            query,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return self.search_by_vector(q_emb, k)
    
    def save_index(self, path: str):
        """Save the FAISS index to disk."""
        # If on GPU, bring back to CPU
        index_to_save = self._index
        if self.device == "cuda" and hasattr(faiss, "index_gpu_to_cpu"):
            index_to_save = faiss.index_gpu_to_cpu(self._index)
        if (
            hasattr(faiss, "write_index")
            and getattr(self._index.__class__, "__module__", "").startswith("faiss")
        ):
            faiss.write_index(index_to_save, path)
    
    @classmethod
    def load_index(cls, path: str, model_name: str, corpus: Dict[str, str], batch_size: int = 64, use_fp16: bool = False):
        """Load a retriever from a saved FAISS index on disk."""
        self = cls.__new__(cls)
        self._corpus = corpus.copy()
        self.doc_ids = list(corpus.keys())
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if SentenceTransformer is None:
            raise ImportError(
                "sentence_transformers is required for EmbeddingRetriever"
            )
        if not hasattr(faiss, "IndexFlatIP"):
            raise ImportError("faiss is required for EmbeddingRetriever")
        self.model = SentenceTransformer(model_name)
        if use_fp16 and self.device == "cuda":
            self.model = self.model.half()
        index = faiss.read_index(path)
        if self.device == "cuda" and hasattr(faiss, "StandardGpuResources"):
            res = faiss.StandardGpuResources()
            index = faiss.index_cpu_to_gpu(res, 0, index)
        self._index = index
        return self
