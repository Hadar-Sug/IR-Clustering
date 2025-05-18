import ir_datasets
from rich.progress import track
from typing import Dict, List, Tuple
import gzip

class DataLoader:
    def __init__(self, dataset: str, docs_gz_path: str):
        # 1) Load queries, scoreddocs & qrels as before
        self.ds = ir_datasets.load(dataset)

        self._queries = {
            q.query_id: q.text
            for q in track(self.ds.queries_iter(), description="Loading queries")
        }

        self._pairs = list(
            track(self.ds.scoreddocs_iter(), description="Loading doc-score pairs")
        )
        self._doc_ids = {p.doc_id for p in self._pairs}

        # 2) Load qrels
        self._qrels: Dict[str, Dict[str,int]] = {}
        for qrel in track(self.ds.qrels_iter(), description="Loading qrels"):
            self._qrels.setdefault(qrel.query_id, {})[qrel.doc_id] = qrel.relevance

        # 3) Load *only* the test docs from your local .gz
        self._docs: Dict[str,str] = {}
        with gzip.open(docs_gz_path, "rt", encoding="utf8") as fh:
            for line in track(fh, description="Loading test docs"):
                # each line is: doc_id<TAB>text
                parts = line.rstrip("\n").split("\t", 1)
                if len(parts) != 2:
                    continue
                doc_id, text = parts
                if doc_id in self._doc_ids:
                    self._docs[doc_id] = text

        # sanity check
        missing = self._doc_ids - self._docs.keys()
        if missing:
            raise RuntimeError(f"Missing docs: {len(missing)} ids not found in {docs_gz_path}")

    def load_queries(self) -> Dict[str, str]:
        return self._queries

    def load_qrels(self) -> Dict[str, Dict[str, int]]:
        return self._qrels

    def load_corpus(self) -> Dict[str, str]:
        return self._docs

    def load_pairs(self) -> List[Tuple[str, str, float]]:
        # List of (query_id, doc_id, score)
        return [(p.query_id, p.doc_id, p.score) for p in self._pairs]