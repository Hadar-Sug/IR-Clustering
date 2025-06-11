import ir_datasets
from rich.progress import track
from typing import Dict, List, Tuple
import gzip
import os
import json

class DataLoader:
    def __init__(
            self,
            queries_tsv: str,
            top100_gz:    str,
            qrels_txt:    str,
            docs_trec_gz: str,
            docs_json_path: str,     # ⬅️ Add path argument
    ):
        if not (os.path.exists(docs_json_path) or os.path.exists(docs_trec_gz)):
            raise FileNotFoundError(f"Document file not found: {docs_trec_gz}")

        # 1) Queries
        self.queries: Dict[str, str] = {}
        if queries_tsv.endswith('.gz'):
            open_fn = lambda path: gzip.open(path, "rt", encoding="utf8")
        else:
        
            open_fn = lambda path: open(path, "r", encoding="utf8")
        with open_fn(queries_tsv) as f:
            for line in f:
                qid, text = line.rstrip("\n").split("\t", 1)
                self.queries[qid] = text


        # 2) Candidate run
        self.pairs: List[Tuple[str, str, float]] = []
        self.doc_ids = set()
        with gzip.open(top100_gz, "rt", encoding="utf8") as f:
            line_count = 0
            for line in f:
                line_count += 1
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                qid, _, docid, _, score = parts[:5]
                self.pairs.append((qid, docid, float(score)))
                self.doc_ids.add(docid)

        # 3) Qrels - formatted as { qid: { docid: relevance_int, ... }, ... }
        self.qrels: Dict[str, Dict[str, int]] = {}
        if qrels_txt.endswith(".gz"):
            qrels_open = lambda p: gzip.open(p, "rt", encoding="utf8")
        else:
            qrels_open = lambda p: open(p, "r", encoding="utf8")
        with qrels_open(qrels_txt) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 4:
                    continue
                qid, _, docid, rating = parts
                if qid not in self.qrels:
                    self.qrels[qid] = {}
                self.qrels[qid][docid] = int(rating)

        # 4) Load only the test-set documents from your local full dump (.tsv after unzipping)
        self.docs: Dict[str, str] = {}
        self.json_path = docs_json_path

        if os.path.exists(self.json_path):
            # Load docs dict directly from json if available
            with open(self.json_path, "r", encoding="utf8") as jf:
                try:
                    self.docs = json.load(jf)
                except json.JSONDecodeError:
                    self.docs = {}
        if not self.docs:
            # Normal loading from gz file if json missing or invalid
            with gzip.open(docs_trec_gz, "rt", encoding="utf8") as f:
                for line in f:
                    parts = line.rstrip("\n").split("\t")
                    if len(parts) < 2:
                        continue  # skip malformed lines
                    docid = parts[0]
                    text = parts[-1]
                    if docid in self.doc_ids:
                        self.docs[docid] = text
            # dump to json for future use
            with open(self.json_path, "w", encoding="utf8") as jf:
                json.dump(self.docs, jf)
        else:
            # Ensure all required docs are present; load missing ones from gz if available
            missing_ids = self.doc_ids - self.docs.keys()
            if missing_ids and os.path.exists(docs_trec_gz):
                with gzip.open(docs_trec_gz, "rt", encoding="utf8") as f:
                    for line in f:
                        parts = line.rstrip("\n").split("\t")
                        if len(parts) < 2:
                            continue
                        docid = parts[0]
                        text = parts[-1]
                        if docid in missing_ids:
                            self.docs[docid] = text
                            missing_ids.remove(docid)
                            if not missing_ids:
                                break
                with open(self.json_path, "w", encoding="utf8") as jf:
                    json.dump(self.docs, jf)

        # Sanity check (existing)
        missing = self.doc_ids - self.docs.keys()
        if missing:
            sample = ", ".join(sorted(missing)[:5])
            raise RuntimeError(
                f"Missing {len(missing)} docs in {docs_trec_gz}: {sample}"
            )

    def load_queries(self) -> Dict[str, str]:
        return self.queries

    def load_qrels(self) -> Dict[str, Dict[str, int]]:
        return self.qrels

    def load_corpus(self) -> Dict[str, str]:
        return self.docs
    
    def load_pairs(self) -> List[Tuple[str, str, float]]:
        # List of (query_id, doc_id, score)
        return [(p[0], p[1], p[2]) for p in self.pairs]
