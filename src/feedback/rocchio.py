from typing import Dict, Optional
import numpy as np

from ..domain.interfaces import FeedbackService

class RocchioTrueFeedback(FeedbackService):
    def __init__(
        self,
        qrels: Dict[str, Dict[str, int]],
        top_k_relevant_docs: int = 3,
        alpha: float = 1.0,
        beta: float = 0.75,
        gamma: float = 0.0,
    ):
        self.qrels = qrels
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.top_k_relevant_docs = top_k_relevant_docs
    def refine(self, qid, q_vec, doc_vecs):
        # Find relevant doc_ids that are present in doc_vecs
        rel_ids = [d for d, r in self.qrels.get(qid, {}).items() if r > 0 and d in doc_vecs.keys()]
        non_ids = [d for d, r in self.qrels.get(qid, {}).items() if r <= 0 and d in doc_vecs.keys()]
        if not rel_ids and not non_ids:
            # Skip refinement when no feedback information is available
            # (no relevant or non-relevant documents for this query)
            pass
            
        scored_rel = [(d, doc_vecs[d]) for d in rel_ids]
        
        # Use only the first k relevant docs
        if self.top_k_relevant_docs is not None:
            scored_rel = scored_rel[:self.top_k_relevant_docs]
        
        top_rel_vecs = [vec for _, vec in scored_rel]
        non_vecs = [doc_vecs[d] for d in non_ids]

        if not top_rel_vecs and not non_vecs:
            return q_vec

        rel_centroid = np.mean(top_rel_vecs, axis=0) if top_rel_vecs else np.zeros_like(q_vec)
        non_centroid = np.mean(non_vecs, axis=0) if non_vecs else np.zeros_like(q_vec)

        return (
            self.alpha * q_vec +
            self.beta * rel_centroid -
            self.gamma * non_centroid
        )
