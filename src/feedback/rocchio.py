from typing import Dict, Optional
import numpy as np
from ..domain.interfaces import FeedbackService

class RocchioTrueFeedback(FeedbackService):
    def __init__(
        self, 
        qrels: Dict[str, Dict[str, int]], 
        alpha=1.0, 
        beta=0.75, 
        gamma=0.15,
        k: Optional[int] = None
    ):
        self.qrels = qrels
        self.alpha, self.beta, self.gamma = alpha, beta, gamma
        self.k = k  # number of top relevant docs to use

    def refine(self, qid, q_vec, doc_vecs):
        # Find relevant doc_ids that are present in doc_vecs
        rel_ids = [d for d, r in self.qrels.get(qid, {}).items() if r > 0 and d in doc_vecs]

        # Sort those relevant doc_ids by their scores (descending), using only those present in doc_vecs
        # The order in doc_vecs is inherited from reranking, so sort by available scores
        scored_rel = [(d, doc_vecs[d]) for d in rel_ids]
        
        # Use only the first k relevant docs
        if self.k is not None:
            scored_rel = scored_rel[:self.k]
        
        top_rel_vecs = [vec for _, vec in scored_rel]

        if not top_rel_vecs:
            return q_vec

        # Compute centroid only over top-k relevant docs
        rel_centroid = np.mean(top_rel_vecs, axis=0)

        # No non-relevant docs used in the centroid
        return self.alpha * q_vec + self.beta * rel_centroid