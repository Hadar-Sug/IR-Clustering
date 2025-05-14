# tests/test_pipeline.py

import numpy as np
import pytest

from src.schema import DocScore
from src.pipeline.pipeline import Pipeline
from src.domain.interfaces import Retriever, FeedbackService, Evaluator

# ——— Dummy Components ————————————————————————————————————————— #

class DummyFirstRetriever(Retriever):
    def __init__(self, hits):
        self._hits = hits

    def search(self, query: str, k: int):
        # ignore query/k, return predefined hits
        return self._hits

    def doc_text(self, doc_id: str) -> str:
        raise NotImplementedError("Should not be called on first_stage")

    def search_by_vector(self, q_vec: np.ndarray, k: int):
        raise NotImplementedError("Should not be called on first_stage")


class DummyEmbedRetriever(Retriever):
    def __init__(self, corpus: dict[str, str], results: list[DocScore]):
        # raw corpus for doc_text
        self._corpus = corpus
        # predetermined vector‐based search results
        self._results = results

    def search(self, query: str, k: int):
        raise NotImplementedError("Not used in pipeline")

    def doc_text(self, doc_id: str) -> str:
        return self._corpus[doc_id]

    def search_by_vector(self, q_vec: np.ndarray, k: int) -> list[DocScore]:
        # ignore q_vec/k, always return the same list
        return self._results


class CaptureFeedback(FeedbackService):
    def __init__(self):
        self.calls = []

    def refine(self, qid: str, q_vec: np.ndarray, doc_vecs: dict[str, np.ndarray]) -> np.ndarray:
        # record inputs and return a dummy new vector
        self.calls.append((qid, q_vec.copy(), {d: v.copy() for d, v in doc_vecs.items()}))
        # return a fixed-length vector (value doesn't matter, emb_retriever ignores it)
        return np.array([999], dtype=np.float32)


class DummyEmbedModel:
    def __init__(self):
        self.calls = []

    def encode(self, text, convert_to_numpy=True, normalize_embeddings=True):
        # record the raw input and return its length as a 1D vector
        self.calls.append(text)
        length = len(text)
        return np.array([length], dtype=np.float32)


class DummyEvaluator(Evaluator):
    def __init__(self):
        self.calls = []

    def evaluate(self, run: dict[str, dict[str, float]], qrels: dict[str, dict[str, int]]) -> dict[str, float]:
        # record inputs and return a dummy metric
        self.calls.append((run, qrels))
        return {"dummy_metric": 42.0}


# ——— Fixtures ————————————————————————————————————————————————————— #

@pytest.fixture
def toy_hits():
    return [
        DocScore(doc_id="d1", score=1.23),
        DocScore(doc_id="d2", score=0.45)
    ]

@pytest.fixture
def toy_corpus():
    return {
        "d1": "text for doc1",
        "d2": "another text"
    }

@pytest.fixture
def rerank_results():
    return [
        DocScore(doc_id="d2", score=9.99),
        DocScore(doc_id="d1", score=5.55)
    ]


# ——— Tests ——————————————————————————————————————————————————

def test_run_query_flow(toy_hits, toy_corpus, rerank_results):
    first = DummyFirstRetriever(toy_hits)
    emb_ret  = DummyEmbedRetriever(toy_corpus, rerank_results)
    feedback = CaptureFeedback()
    model    = DummyEmbedModel()
    evaluator= DummyEvaluator()

    pipeline = Pipeline(first, emb_ret, feedback, evaluator, model)

    qid   = "q42"
    query = "hello world"
    out = pipeline.run_query(qid, query, k=10)

    # 1) Pipeline should return exactly the rerank_results as a dict
    expected = {d.doc_id: d.score for d in rerank_results}
    assert out == expected

    # 2) Feedback.refine should have been called once with:
    #    - same qid
    #    - q_vec = [len(query)]
    #    - doc_vecs = {doc_id: [len(text)]}
    assert len(feedback.calls) == 1
    called_qid, called_q_vec, called_doc_vecs = feedback.calls[0]

    assert called_qid == qid
    np.testing.assert_array_equal(called_q_vec, np.array([len(query)], dtype=np.float32))

    # check doc_vecs
    for doc_id, text in toy_corpus.items():
        expected_vec = np.array([len(text)], dtype=np.float32)
        np.testing.assert_array_equal(called_doc_vecs[doc_id], expected_vec)

    # 3) embed_model.encode should have been called once for the query
    assert model.calls[0] == query


def test_evaluate_aggregates_runs(toy_hits, toy_corpus, rerank_results):
    first = DummyFirstRetriever(toy_hits)
    emb_ret  = DummyEmbedRetriever(toy_corpus, rerank_results)
    feedback = CaptureFeedback()
    model    = DummyEmbedModel()
    evaluator= DummyEvaluator()

    pipeline = Pipeline(first, emb_ret, feedback, evaluator, model)

    # two queries
    queries = {"q1": "one", "q2": "two"}
    # sample qrels dict
    qrels = {
        "q1": {"d2": 1},
        "q2": {"d1": 0, "d2": 1}
    }

    metrics = pipeline.evaluate(queries, qrels, k=5)

    # 1) evaluate() returns exactly whatever DummyEvaluator returned
    assert metrics == {"dummy_metric": 42.0}

    # 2) DummyEvaluator.evaluate was called once with:
    assert len(evaluator.calls) == 1
    called_run, called_qrels = evaluator.calls[0]

    # run should have an entry per query, each mapping to rerank_results
    expected_run = {
        "q1": {d.doc_id: d.score for d in rerank_results},
        "q2": {d.doc_id: d.score for d in rerank_results},
    }
    assert called_run == expected_run
    assert called_qrels is qrels  # same object passed through
