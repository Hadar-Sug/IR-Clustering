# python
import pandas as pd
import pyterrier as pt
from ..schema import DocScore
from ..domain.interfaces import Retriever

pt.init()

class PyTerrierRM3Retriever(Retriever):
    def __init__(self, corpus: dict[str,str], fb_terms=10, fb_docs=10, ret=1000):
        # build a DF for PyTerrier
        df = (
            pd.DataFrame.from_dict(corpus, orient='index', columns=['text'])
              .reset_index()
              .rename(columns={'index':'docno'})
        )
        # create an in-memory index
        self.index = pt.IndexDataFrame(df, 'docno', 'text')
        # first‐stage BM25
        bm25 = pt.BatchRetrieve(self.index, wmodel='BM25')
        # RM3 expansion
        self.pipeline = pt.rewrite.RM3(bm25, fb_terms=fb_terms, fb_docs=fb_docs, ret=ret)

    def search(self, query: str, k: int = 100) -> list[DocScore]:
        # prepare a one‐row query DF
        qdf = pd.DataFrame([{'query': query, 'qid': 'Q0'}])
        # run RM3 + retrieve
        res = self.pipeline.transform(qdf)
        # take top-k results
        scores = res.nlargest(k, 'score')
        return [DocScore(row['docno'], float(row['score'])) for _, row in scores.iterrows()]