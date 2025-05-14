from typing import Dict, List
import ir_datasets
from rich.progress import track

class DataLoader:
    def __init__(self, dataset_name: str):
        self.ds = ir_datasets.load(dataset_name)

    def load_corpus(self) -> Dict[str, str]:
        return {d.doc_id: d.text for d in track(self.ds.docs_iter(), description="Loading docs")}

    def load_queries(self) -> Dict[str, str]:
        return {q.query_id: q.text for q in self.ds.queries_iter()}

    def load_qrels(self) -> Dict[str, Dict[str, int]]:
        qrels: Dict[str, Dict[str, int]] = {}
        for qrel in self.ds.qrels_iter():
            qrels.setdefault(qrel.query_id, {})[qrel.doc_id] = qrel.relevance
        return qrels