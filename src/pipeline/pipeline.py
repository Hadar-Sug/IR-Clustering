from typing import Dict, List
from ..domain.interfaces import Retriever, FeedbackService, Evaluator, DocScore
import numpy as np

class Pipeline:
    def __init__(self, retriever_init: Retriever, emb_retriever: Retriever, fb: FeedbackService, evaluator: Evaluator, embed_model):
        self.first_stage = retriever_init
        self.emb_retriever = emb_retriever
        self.feedback = fb
        self.evaluator = evaluator
        self.embed_model = embed_model

    def run_query(self, qid: str, query: str, k: int) -> Dict[str, float]:
        # 1) initial BM25 search
        first_hits = self.first_stage.search(query, k)
        # 2) embed query + docs
        q_vec = self.embed_model.encode(query, convert_to_numpy=True, normalize_embeddings=True)
        doc_texts = {ds.doc_id: self.emb_retriever.doc_text(ds.doc_id) for ds in first_hits}
        doc_vecs = {d: self.embed_model.encode(t, convert_to_numpy=True, normalize_embeddings=True) for d, t in doc_texts.items()}
        # 3) true feedback
        q_vec_prime = self.feedback.refine(qid, q_vec, doc_vecs)
        # 4) re‑rank in embedding space
        reranked = self.emb_retriever.search_by_vector(q_vec_prime, k)
        return {d.doc_id: d.score for d in reranked}

    def evaluate(self, queries: Dict[str, str], k: int = 100):
        run: Dict[str, Dict[str, float]] = {}
        for qid, query in queries.items():
            run[qid] = self.run_query(qid, query, k)
        return self.evaluator.evaluate(run)