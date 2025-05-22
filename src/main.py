from pathlib import Path
import statistics
from rich import print
import csv

from .utils.config import Config
from .utils.dependencies import build_pipeline, RocchioTrueFeedback

if __name__ == "__main__":
    # List of config files to iterate over
    config_files = [
        "2019_dl_config_a_0_5_b_1.yml",
    ]

    for config_file in config_files:
        # Load current config
        cfg = Config.load(Path(config_file))

        # Build main pipeline and get data
        pipe, queries, qrels = build_pipeline(cfg)

        # Filter queries to only those with matching qrels
        queries = {qid: query for qid, query in queries.items() if qid in qrels}
        qrels = {qid: qrels[qid] for qid in queries}

        print(f"[bold yellow]\n=== Results for {config_file} ===[/bold yellow]")
        print(f"Filtered to {len(queries)} queries with matching qrels")

        # Extract ready-built resources/releases from the pipeline for variants
        bm25 = pipe.first_stage
        emb_retriever = pipe.emb_retriever
        embed_model = pipe.embed_model
        feedback = pipe.feedback
        evaluator = pipe.evaluator
        corpus = pipe.doc_corpus

        # Build feedback variants with correct k for top-k relevant
        rocchio3 = RocchioTrueFeedback(qrels, 3, cfg.alpha, cfg.beta)
        rocchio5 = RocchioTrueFeedback(qrels, 5, cfg.alpha, cfg.beta)

        # Assemble configurations for ablation
        variants = [
            ("BM25 retrieval only",      type(pipe)(bm25, bm25,        None,     evaluator, embed_model, doc_corpus=corpus)),
            ("E5 dense only",            type(pipe)(emb_retriever, emb_retriever, None,     evaluator, embed_model, doc_corpus=corpus)),
            ("E5 + Rocchio (k=3)",       type(pipe)(emb_retriever, emb_retriever, rocchio3, evaluator, embed_model, doc_corpus=corpus)),
            ("E5 + Rocchio (k=5)",       type(pipe)(emb_retriever, emb_retriever, rocchio5, evaluator, embed_model, doc_corpus=corpus)),
        ]

        final_results = []

        for label, pipeline in variants:
            run = {}
            for qid, query in queries.items():
                results = pipeline.run_query(qid, query, k=10)
                run[qid] = results

            metrics = pipeline.evaluator.evaluate(run, qrels)
            print(metrics)
            metric_names = list(next(iter(metrics.values())).keys()) if metrics else []

            macro = {}
            for metric in metric_names:
                scores = [qres.get(metric, 0.0) for qres in metrics.values()]
                macro[metric] = statistics.mean(scores) if scores else 0.0

            macro["label"] = label
            final_results.append(macro)

        # Print table
        if final_results:
            print("\n=== Results Table ===")
            header = ["label"] + [m for m in final_results[0] if m != "label"]
            print("\t".join(header))
            for row in final_results:
                print("\t".join(f"{row[m]:.4f}" if m != "label" else str(row[m]) for m in header))

            # Save results to a file based on the config file name
            csv_name = f"{Path(config_file).stem}_results.csv"
            with open(csv_name, "w", newline="", encoding="utf8") as f:
                writer = csv.DictWriter(f, fieldnames=header)
                writer.writeheader()
                for row in final_results:
                    writer.writerow({k: row[k] for k in header})