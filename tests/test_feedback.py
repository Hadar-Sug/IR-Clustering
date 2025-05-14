import numpy as np
import pytest

from src.feedback.rocchio import RocchioTrueFeedback


@pytest.fixture
def simple_qrels():
    # q1: only doc1 relevant
    # q2: doc1 relevant, doc2 non-relevant
    # q3: empty / no entry
    return {
        "q1": {"d1": 1},
        "q2": {"d1": 1, "d2": 0},
        # "q3" not present
    }


@pytest.fixture
def doc_vecs():
    # two‐dimensional toy embeddings
    return {
        "d1": np.array([2.0, 0.0], dtype=np.float32),
        "d2": np.array([0.0, 3.0], dtype=np.float32),
    }


def test_no_relevant_returns_query(simple_qrels, doc_vecs):
    q_vec = np.array([1.0, 1.0], dtype=np.float32)
    fb = RocchioTrueFeedback(simple_qrels, alpha=1.0, beta=0.5, gamma=0.2)
    out = fb.refine("q3", q_vec, doc_vecs)  # q3 has no entry in qrels
    # Should fall back to original
    assert np.allclose(out, q_vec)


def test_only_positive_feedback(simple_qrels, doc_vecs):
    q_vec = np.array([1.0, 0.0], dtype=np.float32)
    fb = RocchioTrueFeedback(simple_qrels, alpha=1.0, beta=0.75, gamma=0.1)
    out = fb.refine("q1", q_vec, doc_vecs)
    # rel_centroid = d1 = [2,0]
    # non_centroid = 0 (no non-rel docs)
    # expected = alpha*q + beta*rel_centroid = [1 + .75*2, 0]
    expected = np.array([1 + 0.75 * 2.0, 0.0], dtype=np.float32)
    assert np.allclose(out, expected)


def test_positive_and_negative_feedback(simple_qrels, doc_vecs):
    q_vec = np.array([1.0, 0.0], dtype=np.float32)
    fb = RocchioTrueFeedback(simple_qrels, alpha=1.0, beta=0.75, gamma=0.15)
    out = fb.refine("q2", q_vec, doc_vecs)
    # rel_centroid = d1 = [2,0]
    # non_centroid = d2 = [0,3]
    # expected = [1 + .75*2 - .15*0, 0 + .75*0 - .15*3]
    expected = np.array([1 + 0.75*2.0, -0.15*3.0], dtype=np.float32)
    assert np.allclose(out, expected)


def test_output_shape_and_dtype(simple_qrels, doc_vecs):
    q_vec = np.zeros(4, dtype=np.float32)
    # extend doc_vecs to 4‐D for this test
    doc_vecs_4d = {
        "d1": np.array([1, 0, 2, 0], dtype=np.float32),
        "d2": np.array([0, 1, 0, 3], dtype=np.float32),
    }
    qrels_4d = {"q4": {"d1": 1, "d2": 0}}
    fb = RocchioTrueFeedback(qrels_4d, alpha=0.5, beta=0.5, gamma=0.5)
    out = fb.refine("q4", q_vec, doc_vecs_4d)
    # Should still be 4‐dimensional
    assert out.shape == (4,)
    assert out.dtype == np.float32
