from sentence_transformers import SentenceTransformer

from .config import Config
from ..data_loader.loader import DataLoader
from ..retriever.bm25 import BM25Retriever
from ..retriever.embedding import EmbeddingRetriever
from ..feedback.rocchio import RocchioTrueFeedback
from ..evaluator.trec_eval import TrecEvaluator
from ..pipeline.pipeline import Pipeline


def build_pipeline(
    cfg: Config,
    *,
    alpha: float | None = None,
    beta: float | None = None,
    rocchio_k: int | None = None,
) -> tuple[Pipeline, dict[str, str], dict[str, dict[str, int]]]:
    """Construct the main retrieval pipeline using the configuration."""
    dl = DataLoader(
        queries_tsv=cfg.queries_path,
        top100_gz=cfg.top100_path,
        qrels_txt=cfg.qrels_path,
        docs_trec_gz=cfg.test_docs_path,
        docs_json_path=cfg.docs_json_path,
    )

    corpus = dl.load_corpus()
    queries = dl.load_queries()
    qrels = dl.load_qrels()

    bm25 = BM25Retriever(corpus)
    model = SentenceTransformer(cfg.model_name)
    emb_ret = EmbeddingRetriever(cfg.model_name, corpus, cfg.index_path)

    fb = RocchioTrueFeedback(
        qrels,
        rocchio_k if rocchio_k is not None else cfg.rocchio_k[0],
        alpha if alpha is not None else cfg.alpha,
        beta if beta is not None else cfg.beta,
    )

    evalr = TrecEvaluator(cfg.metrics)

    return Pipeline(bm25, emb_ret, fb, evalr, model, corpus), queries, qrels
