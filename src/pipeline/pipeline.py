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
            doc_corpus: dict = None,  # pass loaded corpus mapping doc_id -> text
    ):
        self.first_stage   = retriever_init
        self.emb_retriever = emb_retriever
        self.feedback      = feedback
        self.evaluator     = evaluator
        self.embed_model   = embed_model
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.batch_size = batch_size
        self.use_fp16 = use_fp16
        self.doc_corpus = doc_corpus if doc_corpus is not None else {}

    def run_query(self, qid: str, query: str, k: int) -> Dict[str, float]:
        print(f"Running query {qid}: '{query}'")

        # 1) initial BM25 (or sparse) retrieval
        print("Step 1/4: Initial retrieval...")
        first_hits = self.first_stage.search(query, k)

        # 2) embed query + fetch & embed each doc text
        print("Step 2/4: Creating embeddings for query and documents...")
        q_vec = self.embed_model.encode(
            query,
            convert_to_numpy=True,
            normalize_embeddings=True,
            device=self.device,
            dtype=torch.float16 if (self.use_fp16 and self.device == "cuda") else torch.float32
        )
        doc_texts = {
            hit.doc_id: self.doc_corpus[hit.doc_id]
            for hit in first_hits if hit.doc_id in self.doc_corpus
        }
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
                show_progress_bar=False
            )
            for doc_id, emb in zip(doc_ids, doc_embeddings):
                doc_vecs[doc_id] = emb

        # 3) apply true‐feedback (Rocchio with qrels)
        print("Step 3/4: Applying feedback/refinement...")
        q_vec_prime = self.feedback.refine(qid, q_vec, doc_vecs)

        # 4) re-rank in embedding space
        print("Step 4/4: Re-ranking in embedding space...")
        reranked = self.emb_retriever.search_by_vector(q_vec_prime, k)
        print(f"Finished query {qid}")

        return {hit.doc_id: hit.score for hit in reranked}

    def evaluate(
            self,
            queries: Dict[str, str],
            qrels: Dict[str, Dict[str, int]],
            k: int = 100,
    ) -> Dict[str, float]:
        print("Evaluating queries...")
        run = {}
        for qid, query in tqdm(queries.items(), desc="Running queries"):
            run[qid] = self.run_query(qid, query, k)
        print("Evaluation done.")
        return self.evaluator.evaluate(run, qrels)