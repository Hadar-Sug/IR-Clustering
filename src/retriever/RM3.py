# Heavy optional dependencies -------------------------------------------------
try:  # pragma: no cover - optional dependency
    import pandas as pd
except Exception:  # pragma: no cover - pandas may be absent
    pd = None  # type: ignore[misc]

try:  # pragma: no cover - optional dependency
    import pyterrier as pt
except Exception:  # pragma: no cover - pyterrier may be absent
    pt = None  # type: ignore[misc]
from ..schema import DocScore
from ..domain.interfaces import Retriever
import re
import string


def _sanitize_query(query: str) -> str:
    """Remove all punctuation so Terrier’s parser won’t choke."""
    # replace any punctuation char with space
    sanitized = re.sub(f"[{re.escape(string.punctuation)}]", " ", query)
    # collapse multiple spaces and trim
    return " ".join(sanitized.split())


class PyTerrierRM3Retriever(Retriever):
    def __init__(
        self,
        fb_terms: int = 3,
        fb_docs: int = 2,
        fb_lambda: float = 0.5,
        index_path: str = "pyterrier_index_19",
    ):
        if pt is None or pd is None:
            raise ImportError(
                "pyterrier and pandas are required for PyTerrierRM3Retriever"
            )

        # 0) Make sure PyTerrier is up and running
        if not pt.started():
            pt.init()

        index_dir = index_path

        # 1) Load the index (IndexFactory.of returns a Terrier Index object)
        try:
            terrier_index = pt.IndexFactory.of(index_dir)
            stats = terrier_index.getCollectionStatistics()
            if stats is None:
                raise ValueError("Index has no collection statistics")
            print(f"Loaded index: {index_dir}")
        except Exception as e:
            raise RuntimeError(f"Could not load Terrier index: {e}") from e

        # 2) Create an IndexRef from the same path (this is what BatchRetrieve & RM3 need)
        idx_ref = pt.IndexRef.of(index_dir)

        # 3) First‐stage BM25 for feedback
        bm25_initial = pt.BatchRetrieve(idx_ref, wmodel="BM25")

        # 4) RM3 rewriter (takes the IndexRef, not a retriever)
        rm3 = pt.rewrite.RM3(
            idx_ref,
            fb_terms=int(fb_terms),
            fb_docs=int(fb_docs),
            fb_lambda=float(fb_lambda),
        )

        # 5) Final BM25 on the expanded query
        bm25_final = pt.BatchRetrieve(idx_ref, wmodel="BM25")

        # 6) Chain them: retrieve → rewrite → retrieve
        self.pipeline = bm25_initial >> rm3 >> bm25_final

    def search(self, query: str, k: int = 100) -> list[DocScore]:
        safe_query = _sanitize_query(query)
        # skip if query became empty after sanitization (avoids Terrier NPE)
        if not safe_query:
            return []
        qdf = pd.DataFrame([{"qid": "Q0", "query": safe_query}])
        res = self.pipeline.transform(qdf)
        # pipeline may yield scores as strings; ensure numeric for nlargest
        res["score"] = pd.to_numeric(res["score"], errors="coerce")
        res = res.dropna(subset=["score"])
        topk = res.nlargest(k, "score")
        return [
            DocScore(row["docno"], float(row["score"]))
            for _, row in topk.iterrows()
        ]