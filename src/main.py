from pathlib import Path
import csv
from typing import Dict

from .utils.config import Config
from .utils.dependencies import build_pipeline
from .cross_validation import cv_rm3, cv_embedding
from .retriever.RM3 import PyTerrierRM3Retriever
from .evaluator.trec_eval import TrecEvaluator


def save_csv(path: Path, rows: list[Dict[str, float]]):
    if not rows:
        return
    header = list(rows[0].keys())
    with open(path, 'w', newline='', encoding='utf8') as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def evaluate_test(cfg: Config, rm3_params: Dict[str, float], rocchio_params: Dict[str, float]):
    pipe, queries, qrels = build_pipeline(cfg, **rocchio_params)
    queries = {qid: q for qid, q in queries.items() if qid in qrels}
    run = {}
    for qid, query in queries.items():
        run[qid] = pipe.run_query(qid, query, k=100)
    metrics = pipe.evaluator.evaluate(run, qrels)
    metric_names = list(next(iter(metrics.values())).keys())
    agg = {m: sum(v[m] for v in metrics.values()) / len(metrics) for m in metric_names}

    rm3 = PyTerrierRM3Retriever(index_path=cfg.pt_index_path, **rm3_params)
    rm3_run = {}
    for qid, query in queries.items():
        hits = rm3.search(query, k=100)
        rm3_run[qid] = {h.doc_id: h.score for h in hits}
    evalr = TrecEvaluator(cfg.metrics)
    rm3_metrics = evalr.evaluate(rm3_run, qrels)
    agg_rm3 = {m: sum(v[m] for v in rm3_metrics.values()) / len(rm3_metrics) for m in metric_names}
    return agg, agg_rm3


if __name__ == "__main__":
    cfg = Config.load(Path("config.yaml"))

    best_rm3, rm3_rows = cv_rm3(cfg, Path(cfg.rm3_results_path))
    best_rocchio, rocchio_rows = cv_embedding(cfg, Path(cfg.rocchio_results_path))

    save_csv(Path(cfg.rm3_results_path), rm3_rows)
    save_csv(Path(cfg.rocchio_results_path), rocchio_rows)

    test_dense, test_rm3 = evaluate_test(cfg, best_rm3, best_rocchio)

    save_test = Path(cfg.test_results_path)
    header = ["metric", "dense", "rm3"]
    with open(save_test, "w", newline="", encoding="utf8") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        for m in test_dense:
            writer.writerow({"metric": m, "dense": test_dense[m], "rm3": test_rm3[m]})

    print("Best RM3:", best_rm3)
    print("Best Rocchio:", best_rocchio)
    print("Test Dense Metrics", test_dense)
    print("Test RM3 Metrics", test_rm3)
