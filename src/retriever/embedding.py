import numpy as np
from typing import List, Dict
from sentence_transformers import SentenceTransformer
import faiss
from ..domain.interfaces import Retriever
from ..schema import DocScore


class EmbeddingRetriever(Retriever):
    def __init__(self, model_name: str, corpus: Dict[str, str]):
        # 1) load the encoder
        self.model = SentenceTransformer(model_name)

        # 2) keep doc IDs in order
        self.doc_ids = list(corpus.keys())

        # 3) embed every document, cast to float32 & ensure C-contiguous
        texts = list(corpus.values())
        emb = self.model.encode(
            texts,
            convert_to_numpy=True,
            show_progress_bar=False,
            normalize_embeddings=True
        )
        emb = np.asarray(emb, dtype=np.float32)          # ensure float32
        if not emb.flags['C_CONTIGUOUS']:
            emb = np.ascontiguousarray(emb)

        # 4) build a simple inner-product index
        dim = emb.shape[1]
        self.index = faiss.IndexFlatIP(dim)              # type: ignore
        self.index.add(emb)                              # type: ignore

    def search(self, query: str, k: int) -> List[DocScore]:
        # 5) embed the query
        q_emb = self.model.encode(
            query,
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        q_emb = np.asarray(q_emb, dtype=np.float32)
        q_emb = q_emb.reshape(1, -1)                    # shape (1, d)

        # 6) search the FAISS index
        distances, indices = self.index.search(q_emb, k) # type: ignore

        # 7) map back to DocScore
        results: List[DocScore] = []
        for score, idx in zip(distances[0], indices[0]):
            if idx < 0:  # faiss pads with -1 if fewer than k
                break
            results.append(DocScore(doc_id=self.doc_ids[idx],
                                    score=float(score)))
        return results
