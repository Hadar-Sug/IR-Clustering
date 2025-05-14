from src.retriever.bm25 import BM25Retriever   # <-- your implementation
from src.schema import DocScore    
from conftest import toy_corpus# <-- dataclass


def test_bm25_basic_ranking(toy_corpus):
    retriever = BM25Retriever(toy_corpus)      # ctor tokenises & builds BM25Okapi
    hits = retriever.search("cat", k=3)

    # sanity: correct length & ordering
    assert len(hits) == 3
    assert hits[0].doc_id == "d1"             
    assert hits[1].doc_id == "d3"
    assert hits[0].score >= hits[1].score >= hits[2].score

def test_bm25_empty_query_returns_empty(toy_corpus):
    r = BM25Retriever(toy_corpus)
    hits = r.search("", k=5)
    assert hits == []                          # design choice: empty list

def test_bm25_k_cutoff(toy_corpus):
    r = BM25Retriever(toy_corpus)
    hits = r.search("dog", k=1)
    assert len(hits) == 1
