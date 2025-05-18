import ir_datasets
from rich.progress import track
from typing import Dict, List, Tuple
import gzip

class DataLoader:
    """
    Loader for TREC DL-2019 test set from three local files:
      • msmarco-test2019-queries.tsv   (TSV of qid<TAB>query)
      • msmarco-doctest2019-top100.gz   (GZIP of space-sep runs: qid Q0 docid rank score runname)
      • 2019qrels-docs.txt              (Plain text: qid Q0 docid rating)
    """
    def __init__(
            self,
            queries_tsv: str,
            top100_gz:    str,
            qrels_txt:    str,
            test_docs_gz: str,
    ):
        # 1) Queries (TSV, possibly gzipped)
        self._queries: Dict[str, str] = {}
        if queries_tsv.endswith('.gz'):
            open_fn = lambda path: gzip.open(path, "rt", encoding="utf8")
        else:
            open_fn = lambda path: open(path, "r", encoding="utf8")
        with open_fn(queries_tsv) as f:
            for line in f:
                qid, text = line.rstrip("\n").split("\t", 1)
                self._queries[qid] = text

        # 2) Candidate run (GZIP)
        self._pairs: List[Tuple[str, str, float]] = []
        self.doc_ids = set()
        with gzip.open(top100_gz, "rt", encoding="utf8") as f:
            for line in f:
                # expected: qid Q0 docid rank score runname
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                qid, _, docid, _, score = parts[:5]
                self._pairs.append((qid, docid, float(score)))
                self.doc_ids.add(docid)

        # 3) Qrels (plain text)
        self._qrels: Dict[str, Dict[str,int]] = {}
        with open(qrels_txt, "r", encoding="utf8") as f:
            for line in f:
                # expected: qid Q0 docid rating
                parts = line.strip().split()
                if len(parts) != 4:
                    continue
                qid, _, docid, rating = parts
                self._qrels.setdefault(qid, {})[docid] = int(rating)
        # 4) Load only the test docs from the *mini-corpus*
        self._docs = {}
        with gzip.open(test_docs_gz, "rt", encoding="utf8") as fh:
            for line in fh:
                docid, text = line.rstrip("\n").split("\t", 1)
                self._docs[docid] = text

        # Sanity checks
        if len(self._queries) != 200:
            raise RuntimeError(f"Expected 200 queries, found {len(self._queries)}")
        if len(self._pairs) != 200 * 100:
            raise RuntimeError(f"Expected 20 000 run pairs, found {len(self._pairs)}")

    def load_queries(self) -> Dict[str, str]:
        return self._queries

    def load_qrels(self) -> Dict[str, Dict[str, int]]:
        return self._qrels

    def load_corpus(self) -> Dict[str, str]:
        return self._docs

    def load_pairs(self) -> List[Tuple[str, str, float]]:
        # List of (query_id, doc_id, score)
        return [(p.query_id, p.doc_id, p.score) for p in self._pairs]