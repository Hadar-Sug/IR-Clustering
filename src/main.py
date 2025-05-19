from pathlib import Path
from rich import print
from .utils.config import Config
from .utils.dependencies import build_pipeline

if __name__ == "__main__":
    # Load configuration
    cfg = Config.load(Path("config.yaml"))

    # Build the pipeline, queries, and qrels
    pipeline, queries, qrels = build_pipeline(cfg)

    # Limit to 2 queries for testing purposes
    queries = dict(list(queries.items())[:2])

    # Limit qrels to relevant data for 10 documents only
    limited_qrels = {qid: {doc_id: qrels[qid][doc_id] for doc_id in list(qrels[qid])[:100]} for qid in queries if qid in qrels}

    # Run evaluation on the limited queries and qrels
    metrics = pipeline.evaluate(queries, limited_qrels, k=5)

    # Print evaluation results
    print("[bold green]Evaluation Results[/bold green]")
    for qid, m in metrics.items():
        output = [f"{qid}:"]
        for metric in cfg.metrics:
            if metric in m:
                output.append(f"{metric}={m[metric]}")
        print(" ".join(output))
