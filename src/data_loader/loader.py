import ir_datasets
from rich.progress import track
from typing import Dict, List, Tuple

class DataLoader:
    def __init__(self, dataset: str):
        # Load the MSMARCO Document TREC-DL 2019 collection (Test set)
        self.ds = ir_datasets.load(dataset)

        # Extract queries, pairs, qrels, and doc_ids up front for reuse
        self._queries = {q.query_id: q.text for q in track(self.ds.queries_iter(), description="Loading queries")}
        self._pairs = list(track(self.ds.scoreddocs_iter(), description="Loading doc-score pairs"))
        self._doc_ids = {p.doc_id for p in self._pairs}
        self._docs = {
            doc.doc_id: doc.text
            for doc in track(self.ds.docs_iter(), description="Loading docs")
            if doc.doc_id in self._doc_ids
        }
        self._qrels = {}
        for qrel in track(self.ds.qrels_iter(), description="Loading qrels"):
            self._qrels.setdefault(qrel.query_id, {})[qrel.doc_id] = qrel.relevance

    def load_queries(self) -> Dict[str, str]:
        return self._queries

    def load_qrels(self) -> Dict[str, Dict[str, int]]:
        return self._qrels

    def load_corpus(self) -> Dict[str, str]:
        return self._docs

    def load_pairs(self) -> List[Tuple[str, str, float]]:
        # List of (query_id, doc_id, score)
        return [(p.query_id, p.doc_id, p.score) for p in self._pairs]