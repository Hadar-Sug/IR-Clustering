import itertools
import statistics
from typing import Dict, List, Tuple
import csv
from pathlib import Path

import logging

# --- new: configure logging to both file and console ---
logging.basicConfig(
    filename="cv_debug.log",
    level=logging.DEBUG,
    format="%(asctime)s %(levelname)s:%(message)s",
)
logger = logging.getLogger(__name__)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)
console_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s:%(message)s"))
logger.addHandler(console_handler)
# --- end new ---

from sklearn.model_selection import KFold

from .utils.config import Config
from .utils.dependencies import build_pipeline
from .retriever.RM3 import PyTerrierRM3Retriever
from .evaluator.trec_eval import TrecEvaluator


def _aggregate(metrics: Dict[str, Dict[str, float]], metric_names: List[str]) -> Dict[str, float]:
    agg = {}
    for m in metric_names:
        scores = [res.get(m, 0.0) for res in metrics.values()]
        agg[m] = statistics.mean(scores) if scores else 0.0
    return agg


def cv_rm3(cfg: Config, results_path: Path) -> Tuple[Dict[str, float], List[Dict[str, float]]]:
    logger.debug(f"Starting cv_rm3 with cv_folds={cfg.cv_folds}, results_path={results_path}")
    """Grid search RM3 parameters using k-fold CV on the dev set.
    Resumes from ``results_path`` if it already contains evaluated rows."""
    if not (cfg.dev_queries_path and cfg.dev_top100_path and cfg.dev_qrels_path):
        raise ValueError("Dev set paths must be provided in config")

    # load dev data via build_pipeline but we only need queries/qrels
    dev_cfg = cfg.model_copy(update={
        'queries_path': cfg.dev_queries_path,
        'top100_path': cfg.dev_top100_path,
        'qrels_path': cfg.dev_qrels_path,
    })
    _, queries, qrels = build_pipeline(dev_cfg)
    qids = list(queries.keys())
    kf = KFold(n_splits=cfg.cv_folds, shuffle=True, random_state=42)

    param_grid = list(itertools.product(cfg.rm3_fb_docs, cfg.rm3_fb_terms, cfg.rm3_fb_lambda))
    results: List[Dict[str, float]] = []
    done_params = set()
    if results_path.exists():
        logger.debug(f"Resuming from existing results at {results_path}")
        with open(results_path, newline='', encoding='utf8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                res = {k: float(v) for k, v in row.items()}
                res['fb_docs'] = int(res['fb_docs'])
                res['fb_terms'] = int(res['fb_terms'])
                res['fb_lambda'] = float(res['fb_lambda'])
                results.append(res)
                done_params.add((res['fb_docs'], res['fb_terms'], res['fb_lambda']))

    best_score = float('-inf')
    best_params = {}
    for r in results:
        if r[cfg.metrics[0]] > best_score:
            best_score = r[cfg.metrics[0]]
            best_params = {
                'fb_docs': int(r['fb_docs']),
                'fb_terms': int(r['fb_terms']),
                'fb_lambda': float(r['fb_lambda'])
            }

    logger.debug(f"Initial best_score={best_score}, best_params={best_params}")
    for fb_docs, fb_terms, fb_lambda in param_grid:
        logger.debug(
            f"Evaluating params fb_docs={fb_docs}, fb_terms={fb_terms}, fb_lambda={fb_lambda}"
        )
        if (fb_docs, fb_terms, fb_lambda) in done_params:
            logger.debug("Params already done, skipping")
            continue
        fold_metrics = []
        try:
            for fold_idx, (train_idx, test_idx) in enumerate(kf.split(qids), start=1):
                logger.debug(
                    f"Fold {fold_idx}/{cfg.cv_folds}, test size={len(test_idx)}"
                )
                fold_qids = [qids[i] for i in test_idx]
                fold_qrels = {qid: qrels[qid] for qid in fold_qids}
                rm3 = PyTerrierRM3Retriever(
                    fb_terms=fb_terms,
                    fb_docs=fb_docs,
                    fb_lambda=fb_lambda,
                    index_path=cfg.pt_index_path,
                )
                run = {}
                for qid in fold_qids:
                    hits = rm3.search(queries[qid], k=100)
                    run[qid] = {h.doc_id: h.score for h in hits}
                evalr = TrecEvaluator(cfg.metrics)
                metrics = evalr.evaluate(run, fold_qrels)
                metric_names = list(next(iter(metrics.values())).keys())
                fold_metrics.append(_aggregate(metrics, metric_names))
        except Exception:
            logger.exception(
                "Exception during evaluation for fb_docs=%s fb_terms=%s fb_lambda=%s",
                fb_docs,
                fb_terms,
                fb_lambda,
            )
            continue
        avg = {m: statistics.mean([fm[m] for fm in fold_metrics]) for m in fold_metrics[0]}
        avg.update({'fb_docs': fb_docs, 'fb_terms': fb_terms, 'fb_lambda': fb_lambda})
        logger.debug(f"Averaged metrics: {avg}")

        file_exists = results_path.exists()
        with open(results_path, 'a', newline='', encoding='utf8') as f:
            writer = csv.DictWriter(f, fieldnames=avg.keys())
            if not file_exists:
                writer.writeheader()
            writer.writerow(avg)
        results.append(avg)
        done_params.add((fb_docs, fb_terms, fb_lambda))
        logger.debug(f"Appended results for fb_docs={fb_docs}, fb_terms={fb_terms}, fb_lambda={fb_lambda}")
        if avg[cfg.metrics[0]] > best_score:
            best_score = avg[cfg.metrics[0]]
            best_params = {'fb_docs': fb_docs, 'fb_terms': fb_terms, 'fb_lambda': fb_lambda}
            logger.debug(f"New best_score={best_score}, best_params={best_params}")

    return best_params, results


def cv_embedding(cfg: Config, results_path: Path) -> Tuple[Dict[str, float], List[Dict[str, float]]]:
    logger.debug(f"Starting cv_embedding with cv_folds={cfg.cv_folds}, results_path={results_path}")
    """Grid search Rocchio parameters for the embedding pipeline.
    Resumes from ``results_path`` if available."""
    if not (cfg.dev_queries_path and cfg.dev_top100_path and cfg.dev_qrels_path):
        raise ValueError("Dev set paths must be provided in config")

    dev_cfg = cfg.model_copy(update={
        'queries_path': cfg.dev_queries_path,
        'top100_path': cfg.dev_top100_path,
        'qrels_path': cfg.dev_qrels_path,
    })
    pipe, queries, qrels = build_pipeline(dev_cfg)

    qids = list(queries.keys())
    kf = KFold(n_splits=cfg.cv_folds, shuffle=True, random_state=42)

    param_grid = list(
        itertools.product(cfg.rocchio_alpha, cfg.rocchio_beta, cfg.rocchio_k)
    )
    results: List[Dict[str, float]] = []
    done_params = set()
    if results_path.exists():
        logger.debug(f"Resuming from existing results at {results_path}")
        with open(results_path, newline='', encoding='utf8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                res = {k: float(v) for k, v in row.items()}
                res['alpha'] = float(res['alpha'])
                res['beta'] = float(res['beta'])
                res['rocchio_k'] = int(res['rocchio_k'])
                results.append(res)
                done_params.add((res['alpha'], res['beta'], res['rocchio_k']))

    best_score = float('-inf')
    best_params = {}
    for r in results:
        if r[cfg.metrics[0]] > best_score:
            best_score = r[cfg.metrics[0]]
            best_params = {
                'alpha': r['alpha'],
                'beta': r['beta'],
                'rocchio_k': int(r['rocchio_k']),
            }

    logger.debug(f"Initial best_score={best_score}, best_params={best_params}")
    for alpha, beta, k in param_grid:
        logger.debug(
            f"Evaluating params alpha={alpha}, beta={beta}, rocchio_k={k}"
        )
        if (alpha, beta, k) in done_params:
            logger.debug("Params already done, skipping")
            continue
        fold_metrics = []
        try:
            for fold_idx, (train_idx, test_idx) in enumerate(kf.split(qids), start=1):
                logger.debug(
                    f"Fold {fold_idx}/{cfg.cv_folds}, test size={len(test_idx)}"
                )
                fold_qids = [qids[i] for i in test_idx]
                fold_qrels = {qid: qrels[qid] for qid in fold_qids}
                pipeline, _, _ = build_pipeline(
                    dev_cfg, alpha=alpha, beta=beta, rocchio_k=k
                )
                run = {}
                for qid in fold_qids:
                    run[qid] = pipeline.run_query(qid, queries[qid], k=100)
                metrics = pipeline.evaluator.evaluate(run, fold_qrels)
                metric_names = list(next(iter(metrics.values())).keys())
                fold_metrics.append(_aggregate(metrics, metric_names))
        except Exception:
            logger.exception(
                "Exception during evaluation for alpha=%s beta=%s rocchio_k=%s",
                alpha,
                beta,
                k,
            )
            continue
        avg = {m: statistics.mean([fm[m] for fm in fold_metrics]) for m in fold_metrics[0]}
        avg.update({'alpha': alpha, 'beta': beta, 'rocchio_k': k})
        logger.debug(f"Averaged metrics: {avg}")

        file_exists = results_path.exists()
        with open(results_path, 'a', newline='', encoding='utf8') as f:
            writer = csv.DictWriter(f, fieldnames=avg.keys())
            if not file_exists:
                writer.writeheader()
            writer.writerow(avg)
        done_params.add((alpha, beta, k))
        logger.debug(f"Appended results for alpha={alpha}, beta={beta}, rocchio_k={k}")
        if avg[cfg.metrics[0]] > best_score:
            best_score = avg[cfg.metrics[0]]
            best_params = {'alpha': alpha, 'beta': beta, 'rocchio_k': k}
            logger.debug(f"New best_score={best_score}, best_params={best_params}")

    return best_params, results
