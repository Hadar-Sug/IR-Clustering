import pandas as pd
import pyterrier as pt
from ..schema import DocScore
from ..domain.interfaces import Retriever


def _sanitize_query(query: str) -> str:
    """Remove characters that Terrier's parser dislikes."""
    # Terrier's query parser trips over apostrophes, even if escaped.
    # The simplest fix is to remove them entirely.
    return query.replace("'", "")


class PyTerrierRM3Retriever(Retriever):
    def __init__(self, fb_terms=3, fb_docs=2):
        # 0) Make sure PyTerrier is up and running
        if not pt.started():
            pt.init()

        hardcoded_index_path = (
            "/home/doadmin/Documents/ML/hadar/IR-Clustering/pyterrier_index_19"
        )

        # 1) Load the index (IndexFactory.of returns a Terrier Index object)
        try:
            terrier_index = pt.IndexFactory.of(hardcoded_index_path)
            stats = terrier_index.getCollectionStatistics()
            if stats is None:
                raise ValueError("Index has no collection statistics")
            print(f"Loaded index: {hardcoded_index_path}")
        except Exception as e:
            raise RuntimeError(f"Could not load Terrier index: {e}") from e

        # 2) Create an IndexRef from the same path (this is what BatchRetrieve & RM3 need)
        idx_ref = pt.IndexRef.of(hardcoded_index_path)

        # 3) First‐stage BM25 for feedback
        bm25_initial = pt.BatchRetrieve(idx_ref, wmodel="BM25")

        # 4) RM3 rewriter (takes the IndexRef, not a retriever)
        rm3 = pt.rewrite.RM3(
            idx_ref,
            fb_terms=int(fb_terms),
            fb_docs=int(fb_docs),
        )

        # 5) Final BM25 on the expanded query
        bm25_final = pt.BatchRetrieve(idx_ref, wmodel="BM25")

        # 6) Chain them: retrieve → rewrite → retrieve
        self.pipeline = bm25_initial >> rm3 >> bm25_final

    def search(self, query: str, k: int = 100) -> list[DocScore]:
        safe_query = _sanitize_query(query)
        qdf = pd.DataFrame([{"qid": "Q0", "query": safe_query}])
        res = self.pipeline.transform(qdf)
        topk = res.nlargest(k, "score")
        return [
            DocScore(row["docno"], float(row["score"]))
            for _, row in topk.iterrows()
        ]
