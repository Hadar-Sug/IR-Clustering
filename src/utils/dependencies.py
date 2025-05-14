from sentence_transformers import SentenceTransformer

from .config import Config
from ..data_loader.loader import DataLoader
from ..retriever.bm25 import BM25Retriever
from ..retriever.embedding import EmbeddingRetriever
from ..feedback.rocchio import RocchioTrueFeedback
from ..evaluator.trec_eval import TrecEvaluator
from ..pipeline.pipeline import Pipeline



def build_pipeline(cfg: Config) -> tuple[Pipeline, dict[str, str], dict[str, dict[str, int]]]:
    dl = DataLoader(cfg.dataset)
    corpus = dl.load_corpus()
    queries = dl.load_queries()
    qrels = dl.load_qrels()

    bm25 = BM25Retriever(corpus)
    model = SentenceTransformer(cfg.model_name)
    emb_ret = EmbeddingRetriever(cfg.model_name, corpus)
    fb = RocchioTrueFeedback(qrels, cfg.alpha, cfg.beta, cfg.gamma)
    evalr = TrecEvaluator(cfg.metrics)

    return Pipeline(bm25, emb_ret, fb, evalr, model), queries, qrels