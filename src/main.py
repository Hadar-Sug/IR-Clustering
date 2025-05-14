from pathlib import Path
from rich import print
from .utils.config import Config
from .utils.dependencies import build_pipeline


if __name__ == "__main__":
    cfg = Config.load(Path("/Users/hadar.sugarman/Documents/School/R-Clustering/config.yaml"))

    pipeline, queries, qrels = build_pipeline(cfg)

    metrics = pipeline.evaluate(queries, qrels, k=10)

    print("[bold green]Evaluation Results[/bold green]")
    for qid, m in metrics.items():
        print(qid, m["ndcg_cut_10"])
