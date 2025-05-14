from pathlib import Path
from rich import print
from utils.config import Config
from utils.dependencies import build_pipeline

if __name__ == "__main__":
    cfg = Config.load(Path("config.yaml"))

    # now also unpack the qrels dict
    pipeline, queries, qrels = build_pipeline(cfg)

    # pass qrels into evaluate()
    metrics = pipeline.evaluate(queries, qrels, k=100)

    print("[bold green]Evaluation Results[/bold green]")
    for qid, m in metrics.items():
        print(qid, m["ndcg_cut_10"])
