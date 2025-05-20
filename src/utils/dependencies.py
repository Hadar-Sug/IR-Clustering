from sentence_transformers import SentenceTransformer

from .config import Config
from ..data_loader.loader import DataLoader
from ..retriever.bm25 import BM25Retriever
from ..retriever.embedding import EmbeddingRetriever
from ..feedback.rocchio import RocchioTrueFeedback
from ..evaluator.trec_eval import TrecEvaluator
from ..pipeline.pipeline import Pipeline

def build_pipeline(cfg: Config) -> tuple[Pipeline, dict[str, str], dict[str, dict[str, int]]]:
    # Initialize loader with explicit paths from config
    dl = DataLoader(
        queries_tsv=cfg.queries_path,
        top100_gz=cfg.top100_path,
        qrels_txt=cfg.qrels_path,
        docs_trec_gz=cfg.test_docs_path,
    )

    # Load data
    corpus = dl.load_corpus()
    queries = dl.load_queries()
    qrels = dl.load_qrels()

    # Build Retrieval+Feedback+Evaluation pipeline
    bm25 = BM25Retriever(corpus)
    model = SentenceTransformer(cfg.model_name)
    emb_ret = EmbeddingRetriever(cfg.model_name, corpus, cfg.index_path)
    fb = RocchioTrueFeedback(qrels, cfg.alpha, cfg.beta, cfg.gamma)
    evalr = TrecEvaluator(cfg.metrics)

    return Pipeline(bm25, emb_ret, fb, evalr, model), queries, qrels