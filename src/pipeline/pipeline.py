# src/pipeline.py

from typing import Dict
from ..domain.interfaces import Retriever, FeedbackService, Evaluator

class Pipeline:
    def __init__(
            self,
            retriever_init: Retriever,
            emb_retriever: Retriever,
            feedback: FeedbackService,
            evaluator: Evaluator,
            embed_model,
    ):
        self.first_stage   = retriever_init
        self.emb_retriever = emb_retriever
        self.feedback      = feedback
        self.evaluator     = evaluator
        self.embed_model   = embed_model

    def run_query(self, qid: str, query: str, k: int) -> Dict[str, float]:
        # 1) initial BM25 (or sparse) retrieval
        first_hits = self.first_stage.search(query, k)

        # 2) embed query + fetch & embed each doc text
        q_vec = self.embed_model.encode(
            query, convert_to_numpy=True, normalize_embeddings=True
        )
        doc_texts = {
            hit.doc_id: self.emb_retriever.doc_text(hit.doc_id)
            for hit in first_hits
        }
        doc_vecs = {
            doc_id: self.embed_model.encode(
                text, convert_to_numpy=True, normalize_embeddings=True
            )
            for doc_id, text in doc_texts.items()
        }

        # 3) apply true‐feedback (Rocchio with qrels)
        q_vec_prime = self.feedback.refine(qid, q_vec, doc_vecs)

        # 4) re-rank in embedding space
        reranked = self.emb_retriever.search_by_vector(q_vec_prime, k)
        return {hit.doc_id: hit.score for hit in reranked}

    def evaluate(
            self,
            queries: Dict[str, str],
            qrels: Dict[str, Dict[str, int]],
            k: int = 100,
    ) -> Dict[str, float]:
        """
        Run the full pipeline on each query and evaluate
        using the provided qrels.
        """
        run = {
            qid: self.run_query(qid, query, k)
            for qid, query in queries.items()
        }
        return self.evaluator.evaluate(run, qrels)
