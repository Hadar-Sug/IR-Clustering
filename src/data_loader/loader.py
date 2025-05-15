from typing import Dict
from datasets import load_dataset, Dataset, DatasetDict
from rich.progress import track

class DataLoader:
    def __init__(self, queries_name: str, docs_name: str, qrels_name: str):
        self.queries = load_dataset(queries_name, 'queries')
        self.qrels   = load_dataset(qrels_name, 'qrels')
        self.docs    = load_dataset(docs_name, 'corpus')

    def load_corpus(self) -> Dict[str, str]:
        # Accept both DatasetDict and Dataset
        dataset = self.docs
        if isinstance(dataset, DatasetDict):
            # Prefer common splits, fallback to first one
            for split_name in ['corpus', 'train', 'default', 'docs']:
                if split_name in dataset:
                    dataset = dataset[split_name]
                    break
            else:
                # fallback to first available split
                dataset = next(iter(dataset.values()))
        # Now dataset must be a Dataset object
        # Detect if 'text' or 'body' is the correct field (handle both cases)
        sample = dataset[0]
        text_key = 'body' if 'body' in sample else 'text'
        return {d['doc_id']: d[text_key] for d in track(dataset, description="Loading docs")}

    def load_queries(self) -> Dict[str, str]:
        dataset = self.queries
        if isinstance(dataset, DatasetDict):
            dataset = next(iter(dataset.values()))
        return {q['query_id']: q['text'] for q in track(dataset, description="Loading queries")}

    def load_qrels(self) -> Dict[str, Dict[str, int]]:
        dataset = self.qrels
        if isinstance(dataset, DatasetDict):
            dataset = next(iter(dataset.values()))
        qrels: Dict[str, Dict[str, int]] = {}
        for q in track(dataset, description="Loading qrels"):
            qrels.setdefault(q['query_id'], {})[q['doc_id']] = int(q['relevance'])
        return qrels