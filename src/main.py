from pathlib import Path
from rich import print
from utils.config import Config
from utils.dependencies import build_pipeline

if __name__ == "__main__":
    cfg = Config.load(Path("config.yaml"))
    pipeline, queries = build_pipeline(cfg)
    metrics = pipeline.evaluate(queries, k=100)
    print("[bold green]Evaluation Results[/bold green]")
    for qid, m in metrics.items():
        print(qid, m["ndcg_cut_10"])