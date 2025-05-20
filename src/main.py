from pathlib import Path
import statistics
from rich import print

from .utils.config import Config
from .utils.dependencies import build_pipeline, RocchioTrueFeedback


if __name__ == "__main__":
    cfg = Config.load(Path("config.yaml"))

    # Build main pipeline and get data
    pipe, queries, qrels = build_pipeline(cfg)
    print(f"qrels: {qrels}")

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

    # Build feedback variants with correct k for top-k relevant
    rocchio3 = RocchioTrueFeedback(qrels, cfg.alpha, cfg.beta, cfg.gamma, k=3)
    rocchio5 = RocchioTrueFeedback(qrels, cfg.alpha, cfg.beta, cfg.gamma, k=5)

    # Assemble configurations for ablation
    variants = [
        ("BM25 retrieval only",      type(pipe)(bm25, bm25,        None,     evaluator, embed_model, doc_corpus=corpus)),
        ("E5 dense only",            type(pipe)(emb_retriever, emb_retriever, None,     evaluator, embed_model, doc_corpus=corpus)),
        ("E5 + Rocchio (k=3)",       type(pipe)(emb_retriever, emb_retriever, rocchio3, evaluator, embed_model, doc_corpus=corpus)),
        ("E5 + Rocchio (k=5)",       type(pipe)(emb_retriever, emb_retriever, rocchio5, evaluator, embed_model, doc_corpus=corpus)),
    ]

    for label, pipeline in variants:
        print(f"\n[bold blue]{label}[/bold blue]")
        print(f"Processing queries for {label}...")

        run = {}
        for qid, query in queries.items():
            print(f"\nQuery {qid}: {query}")
            results = pipeline.run_query(qid, query, k=10)
            print(f"Top 10 results for query {qid}:")
            for docid, score in results.items():
                print(f"  {docid}: {score:.4f}")
            run[qid] = results

        print(f"\nFull run results for {label}:")
        print(run)

        print(f"\nEvaluating {label}...")

        # Print first 5 qrels
        print("Sample of qrels (up to 5):")
        qrels_items = list(qrels.items())[:5]
        for qid, relevance in qrels_items:
            print(f"Query {qid}: {relevance}")

        metrics = pipeline.evaluator.evaluate(run, qrels)
        print(f"Raw metrics: {metrics}")

        metric_names = list(next(iter(metrics.values())).keys()) if metrics else []
        print(f"Metric names found: {metric_names}")

        macro = {}
        for metric in metric_names:
            print(f"\nCalculating macro average for {metric}...")
            scores = [qres.get(metric, 0.0) for qres in metrics.values()]
            print(f"Individual scores: {scores}")
            macro[metric] = statistics.mean(scores) if scores else 0.0
            print(f"{metric}: {macro[metric]:.4f}")

        print(f"\nFinal macro metrics for {label}:")
        for metric, score in macro.items():
            print(f"{metric}: {score:.4f}")