from typing import Dict
from ..domain.interfaces import Retriever, FeedbackService, Evaluator

# Add tqdm for a progress bar
from tqdm import tqdm

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
        print(f"Running query {qid}: '{query}'")

        # 1) initial BM25 (or sparse) retrieval
        print("Step 1/4: Initial retrieval...")
        first_hits = self.first_stage.search(query, k)

        # 2) embed query + fetch & embed each doc text
        print("Step 2/4: Creating embeddings for query and documents...")
        q_vec = self.embed_model.encode(
            query, convert_to_numpy=True, normalize_embeddings=True
        )
        doc_texts = {
            hit.doc_id: self.emb_retriever.doc_text(hit.doc_id)
            for hit in first_hits
        }
        doc_vecs = {}
        for doc_id, text in tqdm(doc_texts.items(), desc="Encoding documents", leave=False):
            doc_vecs[doc_id] = self.embed_model.encode(
                text, convert_to_numpy=True, normalize_embeddings=True
            )

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
        """
        Run the full pipeline on each query and evaluate
        using the provided qrels.
        """
        print("Evaluating queries...")
        run = {}
        for qid, query in tqdm(queries.items(), desc="Running queries"):
            run[qid] = self.run_query(qid, query, k)
        print("Evaluation done.")
        return self.evaluator.evaluate(run, qrels)