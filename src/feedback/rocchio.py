from typing import Dict, Optional
import numpy as np
from examples.clirmatrix_example import qrels

from ..domain.interfaces import FeedbackService

class RocchioTrueFeedback(FeedbackService):
    def __init__(
        self, 
        qrels: Dict[str, Dict[str, int]],
        top_k_relevant_docs: int = 3,
        alpha=1.0, 
        beta=0.75,
    ):
        self.qrels = qrels
        self.alpha, self.beta = alpha, beta
        self.top_k_relevant_docs = top_k_relevant_docs
    def refine(self, qid, q_vec, doc_vecs):
        # Find relevant doc_ids that are present in doc_vecs
        print(qrels.get(qid).items())
        print(doc_vecs.keys())
        rel_ids = [d for d, r in self.qrels.get(qid, {}).items() if r > 0 and d in doc_vecs.keys()]
        if not rel_ids:
            print(f"No relevant documents found for query {qid}")

        scored_rel = [(d, doc_vecs[d]) for d in rel_ids]
        
        # Use only the first k relevant docs
        if self.top_k_relevant_docs is not None:
            scored_rel = scored_rel[:self.top_k_relevant_docs]
        
        top_rel_vecs = [vec for _, vec in scored_rel]

        if not top_rel_vecs:
            return q_vec

        # Compute centroid only over top-k relevant docs
        rel_centroid = np.mean(top_rel_vecs, axis=0)

        # No non-relevant docs used in the centroid
        return self.alpha * q_vec + self.beta * rel_centroid