import csv
from pathlib import Path
from typing import Dict, List

import pandas as pd

from .utils.config import Config
from .utils.dependencies import build_pipeline
from .retriever.RM3 import PyTerrierRM3Retriever
from .feedback.rocchio import RocchioTrueFeedback
from .pipeline.pipeline import Pipeline
from .evaluator.trec_eval import TrecEvaluator


def _aggregate(metrics: Dict[str, Dict[str, float]]) -> Dict[str, float]:
    metric_names = list(next(iter(metrics.values())).keys())
    return {m: sum(v[m] for v in metrics.values()) / len(metrics) for m in metric_names}


def _load_best_rm3(path: Path) -> Dict[str, float]:
    best: Dict[str, float] | None = None
    with path.open(newline="", encoding="utf8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row["fb_docs"] = int(row.get("fb_docs", 0))
            row["fb_terms"] = int(row.get("fb_terms", 0))
            row["fb_lambda"] = float(row.get("fb_lambda", 0))
            ndcg = float(row.get("ndcg_cut_10", 0))
            if best is None or ndcg > best.get("ndcg_cut_10", -1):
                best = {
                    "fb_docs": row["fb_docs"],
                    "fb_terms": row["fb_terms"],
                    "fb_lambda": row["fb_lambda"],
                    "ndcg_cut_10": ndcg,
                }
    if best is None:
        raise RuntimeError(f"No rows in {path}")
    return {k: best[k] for k in ("fb_docs", "fb_terms", "fb_lambda")}


def _load_baselines(path: Path) -> List[Dict[str, float]]:
    if not path.exists():
        return []
    df = pd.read_csv(path)
    baselines = []
    for _, row in df.iterrows():
        if row.get("label") in {"BM25 retrieval only", "E5 dense only"}:
            baselines.append(row.to_dict())
    return baselines


def evaluate_pipeline(pipe: Pipeline, queries: Dict[str, str], qrels: Dict[str, Dict[str, int]]) -> Dict[str, float]:
    run = {}
    for qid, query in queries.items():
        run[qid] = pipe.run_query(qid, query, k=100)
    metrics = pipe.evaluator.evaluate(run, qrels)
    return _aggregate(metrics)


def run(cfg_path: str, out_csv: str, out_tex: str) -> None:
    cfg = Config.load(Path(cfg_path))
    best_rm3 = _load_best_rm3(Path(cfg.rm3_results_path))
    pipe, queries, qrels = build_pipeline(cfg, alpha=0.5, beta=1.25, rocchio_k=cfg.rocchio_k[0])
    queries = {qid: q for qid, q in queries.items() if qid in qrels}

    bm25 = pipe.first_stage
    emb = pipe.emb_retriever
    evaluator = pipe.evaluator
    model = pipe.embed_model
    corpus = pipe.doc_corpus

    results: List[Dict[str, float]] = []

    baseline_path = Path(cfg_path).with_name(f"{Path(cfg_path).stem.split('_')[0]}_dl_config_results.csv")
    results.extend(_load_baselines(baseline_path))

    if not any(r.get("label") == "BM25 retrieval only" for r in results):
        bm25_pipe = Pipeline(bm25, bm25, None, evaluator, model, corpus)
        bm25_metrics = evaluate_pipeline(bm25_pipe, queries, qrels)
        bm25_metrics["label"] = "BM25 retrieval only"
        results.append(bm25_metrics)

    if not any(r.get("label") == "E5 dense only" for r in results):
        dense_pipe = Pipeline(emb, emb, None, evaluator, model, corpus)
        dense_metrics = evaluate_pipeline(dense_pipe, queries, qrels)
        dense_metrics["label"] = "E5 dense only"
        results.append(dense_metrics)

    for k in cfg.rocchio_k:
        fb = RocchioTrueFeedback(qrels, k, alpha=0.5, beta=1.25)
        rocchio_pipe = Pipeline(bm25, emb, fb, evaluator, model, corpus)
        m = evaluate_pipeline(rocchio_pipe, queries, qrels)
        m["label"] = f"E5 + Rocchio (k={k})"
        results.append(m)

    rm3 = PyTerrierRM3Retriever(index_path=cfg.pt_index_path, **best_rm3)
    rm3_run = {}
    for qid, query in queries.items():
        hits = rm3.search(query, k=100)
        rm3_run[qid] = {h.doc_id: h.score for h in hits}
    evalr = TrecEvaluator(cfg.metrics)
    rm3_metrics = _aggregate(evalr.evaluate(rm3_run, qrels))
    rm3_metrics["label"] = "RM3"
    results.append(rm3_metrics)

    df = pd.DataFrame(results)
    cols = ["label"] + [c for c in df.columns if c != "label"]
    df = df[cols]
    df.to_csv(out_csv, index=False)
    df.to_latex(out_tex, index=False, float_format="%.4f")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run final test evaluation")
    parser.add_argument("config", help="Path to YAML config")
    parser.add_argument("--csv", default="final_results.csv", help="CSV output path")
    parser.add_argument("--tex", default="final_results.tex", help="LaTeX output path")
    args = parser.parse_args()
    run(args.config, args.csv, args.tex)
