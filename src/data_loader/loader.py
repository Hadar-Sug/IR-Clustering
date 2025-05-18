import ir_datasets
from rich.progress import track
from typing import Dict, List, Tuple
import gzip

class DataLoader:
    def __init__(
            self,
            queries_tsv: str,
            top100_gz:    str,
            qrels_txt:    str,
            docs_trec_gz: str,
    ):
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
                if line_count==1:
                    print(parts)
                if len(parts) < 5:
                    continue
                qid, _, docid, _, score = parts[:5]
                self.pairs.append((qid, docid, float(score)))
                self.doc_ids.add(docid)

        # 3) Qrels
        self.qrels: Dict[str, Dict[str,int]] = {}
        with open(qrels_txt, "r", encoding="utf8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 4:
                    continue
                qid, _, docid, rating = parts
                self.qrels.setdefault(qid, {})[docid] = int(rating)

        # 4) Load only the test-set documents from your local full dump
        self.docs: Dict[str, str] = {}
        with gzip.open(docs_trec_gz, "rt", encoding="utf8") as f:
            for line in f:
                docid, text = line.rstrip("\n").split("\t", 1)
                if docid in self.doc_ids:
                    self.docs[docid] = text
                    # stop once we have fetched all test docs
                    if len(self.docs) == len(self.doc_ids):
                        break

        # Sanity check
        missing = self.doc_ids - self.docs.keys()
        if missing:
            raise RuntimeError(f"Missing {len(missing)} docs in {docs_trec_gz}: {missing}")

    def load_queries(self) -> Dict[str, str]:
        return self.queries

    def load_qrels(self) -> Dict[str, Dict[str, int]]:
        return self.qrels

    def load_corpus(self) -> Dict[str, str]:
        return self.docs
    
    def load_pairs(self) -> List[Tuple[str, str, float]]:
        # List of (query_id, doc_id, score)
        return [(p[0], p[1], p[2]) for p in self.pairs]