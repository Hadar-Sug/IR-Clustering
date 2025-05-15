from typing import Dict
from datasets import load_dataset
from rich.progress import track

class DataLoader:
    def __init__(self, queries_name: str, docs_name: str, qrels_name: str):
        # Do not use defaults; require explicit parameters
        self.queries = load_dataset(queries_name, 'queries')
        self.qrels = load_dataset(qrels_name, 'qrels')
        self.docs = load_dataset(docs_name, 'docs')

    def load_queries(self) -> Dict[str, str]:
        # 'self.queries' is a DatasetDict, get the actual Dataset (split) (usually 'train')
        dataset = self.queries['train'] if 'train' in self.queries else list(self.queries.values())[0]
        return {q['query_id']: q['text'] for q in track(dataset, description="Loading queries")}

    def load_qrels(self) -> Dict[str, Dict[str, int]]:
        dataset = self.qrels['train'] if 'train' in self.qrels else list(self.qrels.values())[0]
        qrels: Dict[str, Dict[str, int]] = {}
        for q in track(dataset, description="Loading qrels"):
            qrels.setdefault(q['query_id'], {})[q['doc_id']] = int(q['relevance'])
        return qrels

    def load_corpus(self) -> Dict[str, str]:
        dataset = self.docs['train'] if 'train' in self.docs else list(self.docs.values())[0]
        return {d['doc_id']: d['body'] for d in track(dataset, description="Loading docs")}