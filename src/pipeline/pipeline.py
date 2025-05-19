from typing import Dict
from ..domain.interfaces import Retriever, FeedbackService, Evaluator

from tqdm import tqdm

import torch

class Pipeline:
    def __init__(
            self,
            retriever_init: Retriever,
            emb_retriever: Retriever,
            feedback: FeedbackService,
            evaluator: Evaluator,
            embed_model,
            batch_size: int = 64,
            use_fp16: bool = False,
            doc_corpus: dict = None,
    ):
        self.first_stage   = retriever_init
        self.emb_retriever = emb_retriever
        self.feedback      = feedback
        self.evaluator     = evaluator
        self.embed_model   = embed_model
        self.device        = "cuda" if torch.cuda.is_available() else "cpu"
        self.batch_size    = batch_size
        self.use_fp16      = use_fp16
        self.doc_corpus    = doc_corpus if doc_corpus is not None else {}

    def run_query(self, qid: str, query: str, k: int) -> Dict[str, float]:
        # Case 1: Pure BM25 baseline
        if self.feedback is None and self.emb_retriever is self.first_stage and hasattr(self.first_stage, "search"):
            first_hits = self.first_stage.search(query, k)
            return {hit.doc_id: hit.score for hit in first_hits}
        
        # Case 2: Pure embedding baseline (dense) -- we run embedding retriever directly
        if self.feedback is None and self.first_stage is self.emb_retriever:
            # 'EmbeddingRetriever' must have 'search' working in embedding space
            first_hits = self.emb_retriever.search(query, k)
            return {hit.doc_id: hit.score for hit in first_hits}
        
        # Case 3: Feedback or reranking (hybrids)
        # 1) Initial retrieval (could be BM25 or dense)
        first_hits = self.first_stage.search(query, k)

        # 2) Embed query & docs
        q_vec = self.embed_model.encode(
            query,
            convert_to_numpy=True,
            normalize_embeddings=True,
            device=self.device,
            dtype=torch.float16 if (self.use_fp16 and self.device == "cuda") else torch.float32,
        )
        doc_texts = {hit.doc_id: self.doc_corpus[hit.doc_id]
                     for hit in first_hits if hit.doc_id in self.doc_corpus}
        doc_vecs = {}
        doc_ids = list(doc_texts.keys())
        texts = list(doc_texts.values())
        if texts:
            doc_embeddings = self.embed_model.encode(
                texts,
                batch_size=self.batch_size,
                convert_to_numpy=True,
                normalize_embeddings=True,
                device=self.device,
                dtype=torch.float16 if (self.use_fp16 and self.device == "cuda") else torch.float32,
                show_progress_bar=False,
            )
            for doc_id, emb in zip(doc_ids, doc_embeddings):
                doc_vecs[doc_id] = emb

        # 3) Feedback
        q_vec_prime = self.feedback.refine(qid, q_vec, doc_vecs) if self.feedback else q_vec

        # 4) Embedding rerank
        reranked = self.emb_retriever.search_by_vector(q_vec_prime, k)

        return {hit.doc_id: hit.score for hit in reranked}