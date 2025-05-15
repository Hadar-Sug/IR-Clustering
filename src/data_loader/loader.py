from typing import Dict
from datasets import load_dataset
from rich.progress import track

class DataLoader:
    def __init__(self, queries_name: str, docs_name: str, qrels_name: str):
        # Do not use defaults; require explicit parameters
        self.queries = load_dataset(queries_name, 'queries')
        self.qrels = load_dataset(qrels_name, 'qrels')
        self.docs = load_dataset(docs_name, 'docs', split='train')

    def load_queries(self) -> Dict[str, str]:
        return {q['query_id']: q['text'] for q in track(self.queries, description="Loading queries")}

    def load_qrels(self) -> Dict[str, Dict[str, int]]:
        qrels: Dict[str, Dict[str, int]] = {}
        for q in track(self.qrels, description="Loading qrels"):
            qrels.setdefault(q['query_id'], {})[q['doc_id']] = int(q['relevance'])
        return qrels

    def load_corpus(self) -> Dict[str, str]:
        return {d['doc_id']: d['body'] for d in track(self.docs, description="Loading docs")}