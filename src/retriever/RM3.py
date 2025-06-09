import pandas as pd
import pyterrier as pt
from ..schema import DocScore
from ..domain.interfaces import Retriever

class PyTerrierRM3Retriever(Retriever):
    def __init__(self, fb_terms=3, fb_docs=2):
        # Hard-coded index path
        hardcoded_index_path = (
            "/home/doadmin/Documents/ML/hadar/IR-Clustering/pyterrier_index_19"
        )

        # Load the on-disk Terrier index
        try:
            idx_ref = pt.IndexRef.of(hardcoded_index_path)
            # Basic sanity check
            stats = pt.IndexFactory.of(idx_ref).getCollectionStatistics()
            if stats is None:
                raise ValueError("Index loaded but has no collection statistics")
            print(f"Successfully loaded index from {hardcoded_index_path}")
            print(f"Index stats: {stats.toString()}")
        except Exception as e:
            raise RuntimeError(f"Error loading Terrier index: {e}") from e

        # 1) First-stage BM25 retriever for pseudo-relevance
        bm25_initial = pt.BatchRetrieve(idx_ref, wmodel="BM25")

        # 2) RM3 query rewriter
        rm3 = pt.rewrite.RM3(
            idx_ref,
            fb_terms=int(fb_terms),
            fb_docs=int(fb_docs),
        )

        # 3) Final BM25 retriever on expanded query
        bm25_final = pt.BatchRetrieve(idx_ref, wmodel="BM25")

        # Compose pipeline: retrieve → expand → retrieve
        self.pipeline = bm25_initial >> rm3 >> bm25_final

    def search(self, query: str, k: int = 100) -> list[DocScore]:
        # Prepare the query DataFrame
        qdf = pd.DataFrame([{"qid": "Q0", "query": query}])

        # Run the full pipeline
        res = self.pipeline.transform(qdf)

        # Take the top-k by score
        topk = res.nlargest(k, "score")

        # Convert to your DocScore domain objects
        return [
            DocScore(row["docno"], float(row["score"]))
            for _, row in topk.iterrows()
        ]
