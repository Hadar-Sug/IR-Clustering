from typing import Dict
import numpy as np
from ..domain.interfaces import FeedbackService

class RocchioTrueFeedback(FeedbackService):
    def __init__(self, qrels: Dict[str, Dict[str, int]], alpha=1.0, beta=0.75, gamma=0.15):
        self.qrels = qrels
        self.alpha, self.beta, self.gamma = alpha, beta, gamma

    def refine(self, qid, q_vec, doc_vecs):
        rel_ids   = [d for d, r in self.qrels.get(qid, {}).items() if r > 0]
        nonrel_ids= [d for d, r in self.qrels.get(qid, {}).items() if r == 0]
        if not rel_ids:   # fallback: no feedback available
            return q_vec
        rel_centroid = np.mean([doc_vecs[d] for d in rel_ids], axis=0)
        non_centroid = np.mean([doc_vecs[d] for d in nonrel_ids], axis=0) if nonrel_ids else 0
        return (self.alpha * q_vec + self.beta * rel_centroid - self.gamma * non_centroid)