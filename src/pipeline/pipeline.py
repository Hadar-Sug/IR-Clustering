from typing import Dict, Optional
from ..domain.interfaces import Retriever, FeedbackService, Evaluator

from tqdm import tqdm

import types
try:  # pragma: no cover - optional dependency
    import torch
except Exception:  # pragma: no cover - torch may be absent
    torch = types.SimpleNamespace(
        cuda=types.SimpleNamespace(is_available=lambda: False),
        float16="float16",
        float32="float32",
    )  # type: ignore[misc]

class Pipeline:
    def __init__(
            self,
            retriever_init: Retriever,
            emb_retriever: Retriever,
            feedback: FeedbackService,
            evaluator: Evaluator,
            embed_model,
            doc_corpus: Optional[Dict[str, str]] = None,
            batch_size: int = 64,
            use_fp16: bool = False,
    ):
        self.first_stage   = retriever_init
        self.emb_retriever = emb_retriever
        self.feedback      = feedback
        self.evaluator     = evaluator
        self.embed_model   = embed_model
        self.device        = "cuda" if torch.cuda.is_available() else "cpu"
        self.batch_size    = batch_size
        self.use_fp16      = use_fp16
        self.doc_corpus    = doc_corpus or {}

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
        )
        doc_texts = {}
        for hit in first_hits:
            if hit.doc_id in self.doc_corpus:
                doc_texts[hit.doc_id] = self.doc_corpus[hit.doc_id]
            elif hasattr(self.emb_retriever, "doc_text"):
                try:
                    doc_texts[hit.doc_id] = self.emb_retriever.doc_text(hit.doc_id)
                except Exception:
                    pass
        
        doc_vecs = {}
        doc_ids = list(doc_texts.keys())
        texts = list(doc_texts.values())

        if texts:
            doc_embeddings = self.embed_model.encode(
                texts,
                convert_to_numpy=True,
                normalize_embeddings=True,
            )
            if len(doc_embeddings) != len(texts):
                doc_embeddings = [
                    self.embed_model.encode(t, convert_to_numpy=True, normalize_embeddings=True)
                    for t in texts
                ]
        for doc_id, emb in zip(doc_ids, doc_embeddings):
            doc_vecs[doc_id] = emb

        # 3) Feedback
        q_vec_prime = self.feedback.refine(qid, q_vec, doc_vecs) if self.feedback else q_vec

        # 4) Embedding rerank

        rerank_top_n = 100
        if doc_vecs:
            if (
                hasattr(self.emb_retriever, "rerank_subset")
                and "rerank_subset" in self.emb_retriever.__class__.__dict__
            ):
                reranked = self.emb_retriever.rerank_subset(
                    q_vec_prime,
                    doc_vecs,
                    k=min(rerank_top_n, len(doc_vecs)),
                )
            elif hasattr(self.emb_retriever, "search_by_vector"):
                reranked = self.emb_retriever.search_by_vector(
                    q_vec_prime,
                    k=min(rerank_top_n, len(doc_vecs)),
                )
            else:
                reranked = []
        else:
            reranked = []


        return {hit.doc_id: hit.score for hit in reranked}

    def evaluate(
        self,
        queries: Dict[str, str],
        qrels: Dict[str, Dict[str, int]],
        k: int = 100,
    ) -> Dict[str, float]:
        run: Dict[str, Dict[str, float]] = {}
        for qid, query in queries.items():
            run[qid] = self.run_query(qid, query, k)
        return self.evaluator.evaluate(run, qrels)
