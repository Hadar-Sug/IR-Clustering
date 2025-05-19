from pathlib import Path
import statistics
from rich import print

from .utils.config import Config
from .utils.dependencies import build_pipeline, RocchioTrueFeedback


if __name__ == "__main__":
    cfg = Config.load(Path("config.yaml"))

    # Build main pipeline and get data
    pipe, queries, qrels = build_pipeline(cfg)

    # Optionally subsample for dev/quick run
    queries = dict(list(queries.items())[:2])
    qrels = {qid: qrels[qid] for qid in queries if qid in qrels}

    # Extract ready-built resources/releases from pipeline for variants
    bm25         = pipe.first_stage
    emb_retriever= pipe.emb_retriever
    embed_model  = pipe.embed_model
    feedback     = pipe.feedback
    evaluator    = pipe.evaluator
    corpus       = pipe.doc_corpus

    # Build feedback variants if needed
    rocchio3 = RocchioTrueFeedback(qrels, cfg.alpha, cfg.beta, cfg.gamma)
    rocchio5 = RocchioTrueFeedback(qrels, cfg.alpha, cfg.beta, cfg.gamma)
    # (If RocchioTrueFeedback takes different arguments for different k, supply as needed)

    # Assemble configurations for ablation
    variants = [
        ("BM25 retrieval only",      type(pipe)(bm25, bm25,            None,     evaluator, embed_model, doc_corpus=corpus)),
        ("E5 dense only",            type(pipe)(emb_retriever, emb_retriever, None,     evaluator, embed_model, doc_corpus=corpus)),
        ("E5 + Rocchio (k=3)",       type(pipe)(emb_retriever, emb_retriever, rocchio3, evaluator, embed_model, doc_corpus=corpus)),
        ("E5 + Rocchio (k=5)",       type(pipe)(emb_retriever, emb_retriever, rocchio5, evaluator, embed_model, doc_corpus=corpus)),
    ]

    for label, pipeline in variants:
        print(f"\n[bold blue]{label}[/bold blue]")
        run = {qid: pipeline.run_query(qid, query, k=10) for qid, query in queries.items()}
        print(run)
        metrics = pipeline.evaluator.evaluate(run, qrels)
        metric_names = list(next(iter(metrics.values())).keys()) if metrics else []
        macro = {}
        for metric in metric_names:
            scores = [qres.get(metric, 0.0) for qres in metrics.values()]
            macro[metric] = statistics.mean(scores) if scores else 0.0
            print(f"{metric}: {macro[metric]:.4f}")