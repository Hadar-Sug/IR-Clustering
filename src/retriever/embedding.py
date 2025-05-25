import numpy as np
from typing import Dict, List
from sentence_transformers import SentenceTransformer
import faiss  # type: ignore
import torch
import os

from ..domain.interfaces import Retriever
from ..schema import DocScore

class EmbeddingRetriever(Retriever):
    """Embedding-based retriever with FAISS, plus helper methods
    for true-feedback pipelines (doc_text, search_by_vector)."""

    def __init__(
        self, 
        model_name: str, 
        corpus: Dict[str, str], 
        index_path: str,
        batch_size: int = 64, 
        use_fp16: bool = False,
    ):
        # raw corpus for doc_text()
        self._corpus: Dict[str, str] = corpus.copy()
        self.doc_ids: List[str] = list(corpus.keys())
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.index_path = index_path

        # 1) load encoder, send to device
        self.model = SentenceTransformer(model_name, device=self.device)
        if use_fp16 and self.device == "cuda":
            self.model = self.model.half()

        if os.path.isfile(self.index_path):
            # If the index exists, load it
            print(f"Loading FAISS index from {self.index_path}")
            index = faiss.read_index(self.index_path)
            if self.device == "cuda":
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
                show_progress_bar=True,
                batch_size=batch_size,
                normalize_embeddings=True,
                device=self.device,
                dtype=torch.float16 if (use_fp16 and self.device == "cuda") else torch.float32
            )
            # ensure float32 format for faiss, contiguous
            emb = np.asarray(emb, dtype=np.float32)
            if not emb.flags["C_CONTIGUOUS"]:
                emb = np.ascontiguousarray(emb)

            # 3) build inner-product index
            dim = emb.shape[1]
            cpu_index = faiss.IndexFlatIP(dim)  # type: ignore

            # GPU: move index to GPU if available
            res = faiss.StandardGpuResources()
            self._index = faiss.index_cpu_to_gpu(res, 0, cpu_index) if self.device == "cuda" else cpu_index

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
            device=self.device,
            dtype=torch.float16 if (hasattr(self.model, "half") and self.device=="cuda") else torch.float32
        )
        return self.search_by_vector(q_emb, k)
    
    def save_index(self, path: str):
        """Save the FAISS index to disk."""
        # If on GPU, bring back to CPU
        index_to_save = self._index
        if self.device == "cuda":
            index_to_save = faiss.index_gpu_to_cpu(self._index)
        faiss.write_index(index_to_save, path)
    
    @classmethod
    def load_index(cls, path: str, model_name: str, corpus: Dict[str, str], batch_size: int = 64, use_fp16: bool = False):
        """Load a retriever from a saved FAISS index on disk."""
        self = cls.__new__(cls)
        self._corpus = corpus.copy()
        self.doc_ids = list(corpus.keys())
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = SentenceTransformer(model_name, device=self.device)
        if use_fp16 and self.device == "cuda":
            self.model = self.model.half()
        index = faiss.read_index(path)
        if self.device == "cuda":
            res = faiss.StandardGpuResources()
            index = faiss.index_cpu_to_gpu(res, 0, index)
        self._index = index
        return self
