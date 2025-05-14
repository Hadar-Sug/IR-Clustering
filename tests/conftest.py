import numpy as np
import pytest

# --- Tiny toy corpus ---------------------------------
@pytest.fixture
def toy_corpus():
    return {
        "d1": "the cat sat on the cat",
        "d2": "dogs are lost in the fog",
        "d3": "cat and dog play together"
    }

# --- Dummy embedder so we don't hit the network -------
class DummyEmbedder:
    """Maps term 'cat' -> axis-0, 'dog' -> axis-1, everything else -> 0."""
    def encode(self, texts, convert_to_numpy=True, normalize_embeddings=True):
        def _vec(t):
            v = np.zeros(2, dtype=np.float32)
            v[0] = t.count("cat")
            v[1] = t.count("dog")
            return v
        if isinstance(texts, str):
            out = _vec(texts)
        else:
            out = np.stack([_vec(t) for t in texts])
        # unit-length normalisation if requested
        if normalize_embeddings:
            n = np.linalg.norm(out, axis=-1, keepdims=True)
            n[n == 0] = 1
            out = out / n
        return out

@pytest.fixture
def dummy_embedder():
    return DummyEmbedder()
