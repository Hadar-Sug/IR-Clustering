import numpy as np
import pytest
from src.schema import DocScore
from src.retriever.embedding import EmbeddingRetriever


# 1) Dummy “SentenceTransformer” model that maps:
#    “cat”→axis-0 count, “dog”→axis-1 count, then L2-normalizes.
class DummyModel:
    def __init__(self, model_name: str):
        # ignore model_name
        pass

    def encode(
            self,
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False
    ):
        def make_vec(t: str):
            v = np.zeros(2, dtype=np.float32)
            v[0] = t.count("cat")
            v[1] = t.count("dog")
            if normalize_embeddings:
                norm = np.linalg.norm(v)
                if norm > 0:
                    v /= norm
            return v

        if isinstance(texts, str):
            return make_vec(texts)
        else:
            return np.stack([make_vec(t) for t in texts])


# 2) Dummy FAISS Index that uses exact inner-product on small arrays
class DummyIndex:
    def __init__(self, dim: int):
        self._embs = None

    def add(self, emb_array: np.ndarray):
        # simply store
        self._embs = emb_array

    def search(self, query_array: np.ndarray, k: int):
        # query_array shape (1, d); self._embs shape (N, d)
        # compute cosine (inner-product since vectors L2-normed)
        dots = (self._embs @ query_array[0]).astype(np.float32)
        # get top-k indices
        idxs = np.argsort(-dots)[:k]
        scores = dots[idxs]
        # FAISS returns arrays [ [scores...] ], [ [idxs...] ]
        return scores[np.newaxis, :], idxs[np.newaxis, :]


@pytest.fixture(autouse=True)
def patch_external(monkeypatch):
    # 1) Monkey-patch SentenceTransformer to DummyModel
    monkeypatch.setattr(
        'src.retriever.embedding.SentenceTransformer',
        DummyModel
    )
    # 2) Monkey-patch faiss.IndexFlatIP to DummyIndex
    monkeypatch.setattr(
        'src.retriever.embedding.faiss.IndexFlatIP',
        DummyIndex
    )


@pytest.fixture
def toy_corpus():
    return {
        "d1": "the cat sat on the mat",
        "d2": "dogs are lost in the fog",
        "d3": "cat and dog play together"
    }


def test_embedding_basic_ranking(toy_corpus):

    retriever = EmbeddingRetriever(model_name="dummy", corpus=toy_corpus)
    hits = retriever.search("cat", k=2)

    # DummyModel: "cat"→v = [1,0], so best match is any doc with "cat"
    # Both d1 and d3 contain "cat" once, but since our DummyIndex breaks ties
    # by zipping in order, we expect ["d1","d3"].
    assert isinstance(hits, list)
    assert len(hits) == 2
    assert [h.doc_id for h in hits] == ["d1", "d3"]
    # scores should be non-increasing
    assert all(hits[i].score >= hits[i+1].score for i in range(len(hits)-1))
    # ensure score matches inner product of normalized vectors
    # For "cat": each v_docs = [1,0]/1 == [1,0], so dot==1.0
    assert pytest.approx(hits[0].score, rel=1e-6) == 1.0


def test_embedding_oov_query(toy_corpus):

    retriever = EmbeddingRetriever(model_name="dummy", corpus=toy_corpus)
    hits = retriever.search("elephant", k=3)

    # DummyModel: "elephant"→v=[0,0], so all doc scores dot([0,0],doc_vec)=0
    assert len(hits) == 3
    assert all(np.isclose(h.score, 0.0) for h in hits)
    # ordering should be corpus order when all scores tie
    assert [h.doc_id for h in hits] == ["d1", "d2", "d3"]


def test_embedding_handles_k_greater_than_docs(toy_corpus):

    retriever = EmbeddingRetriever(model_name="dummy", corpus=toy_corpus)
    hits = retriever.search("cat", k=10)

    # should never return more than available docs
    assert len(hits) == 3

def test_doc_text_returns_original(toy_corpus):
    r = EmbeddingRetriever("dummy", toy_corpus)
    for doc_id, text in toy_corpus.items():
        assert r.doc_text(doc_id) == text


def test_search_by_vector_matches_search(toy_corpus):
    r = EmbeddingRetriever("dummy", toy_corpus)
    # embed the query
    q_vec = r.model.encode("cat", normalize_embeddings=True)
    # call both methods
    by_vec = r.search_by_vector(q_vec, k=2)
    by_txt = r.search("cat", k=2)

    # should get same doc_ids & scores
    assert [h.doc_id for h in by_vec] == [h.doc_id for h in by_txt]
    assert pytest.approx([h.score for h in by_vec]) == [h.score for h in by_txt]


def test_search_by_vector_respects_k_and_scores(toy_corpus):
    r = EmbeddingRetriever("dummy", toy_corpus)
    # vector exactly orthogonal yields 0 scores
    zero_vec = np.zeros(2, dtype=np.float32)
    hits = r.search_by_vector(zero_vec, k=3)
    # should be ordered by how corpus vectors dot zero_vec -> all zeros
    assert len(hits) == 3
    assert all(isinstance(h, DocScore) for h in hits)
    assert all(h.score == 0.0 for h in hits)

def test_embedding_return_type_and_scores(toy_corpus):
    from src.retriever.embedding import EmbeddingRetriever

    retriever = EmbeddingRetriever(model_name="dummy", corpus=toy_corpus)
    hits = retriever.search("dog", k=1)

    # Expect DocScore with proper attributes
    assert isinstance(hits[0], DocScore)
    assert hasattr(hits[0], "doc_id")
    assert hasattr(hits[0], "score")
    # "dog"→v=[0,1] normalized to [0,1], so best hit is d3 ([1 dog]) over d2 ([1 dog] too),
    # tie broken by doc order -> d1 has no dogs, so skip; d2 and d3 both have 1 dog,
    # but in corpus order d2 comes before d3.
    assert hits[0].doc_id in {"d2", "d3"}
    assert pytest.approx(hits[0].score, rel=1e-6) == 1.0
