from typing import Dict, List
from datasets import load_dataset  # Hugging Face Datasets
from rich.progress import track

class DataLoader:
    def __init__(self, dataset_name: str = "irds/msmarco-document-v2_trec-dl-2019_judged"):
        # Loads the 'irds/msmarco-document-v2_trec-dl-2019_judged' Hugging Face dataset by default
        self.dataset = load_dataset(dataset_name)

    def load_corpus(self) -> Dict[str, str]:
        # The corpus is typically in "corpus" split for irds datasets, with "doc_id" and "text"
        ds_split = self.dataset["corpus"] if "corpus" in self.dataset else self.dataset["train"]
        return {doc["doc_id"]: doc["text"] for doc in track(ds_split, description="Loading docs")}

    def load_queries(self) -> Dict[str, str]:
        # Use the "queries" split if available, otherwise "dev" or "test"
        ds_split = None
        for name in ["queries", "dev", "test", "validation"]:
            if name in self.dataset:
                ds_split = self.dataset[name]
                break
        if ds_split is None:
            raise ValueError("No queries split found in dataset.")
        return {query["query_id"]: query["text"] for query in ds_split}

    def load_qrels(self) -> Dict[str, Dict[str, int]]:
        # Qrels are in "qrels" split, fields: 'query_id', 'doc_id', 'relevance'
        if "qrels" not in self.dataset:
            raise ValueError("No qrels split found in dataset.")
        qrels: Dict[str, Dict[str, int]] = {}
        for qrel in self.dataset["qrels"]:
            qrels.setdefault(qrel["query_id"], {})[qrel["doc_id"]] = int(qrel["relevance"])
        return qrels