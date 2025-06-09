# python
import pandas as pd
import pyterrier as pt
from ..schema import DocScore
from ..domain.interfaces import Retriever

pt.init()

class PyTerrierRM3Retriever(Retriever):
    def __init__(self, fb_terms=3, fb_docs=2): # Removed corpus parameter and ret parameter
        # HARDCODED PATH for testing - replace with a parameter later
        # Ensure this path points to a directory containing a valid PyTerrier index
        # (e.g., a directory with data.properties, lexicon.lex, etc.)
        hardcoded_index_path = "/home/doadmin/Documents/ML/hadar/IR-Clustering/pyterrier_index_19" # Replace with your actual index path

        # Load the index from the specified path
        try:
            self.index = pt.IndexFactory.of(hardcoded_index_path)
            if self.index is None or self.index.getCollectionStatistics() is None: # Basic check
                raise ValueError(f"Failed to load index or index is empty at {hardcoded_index_path}")
            print(f"Successfully loaded index from {hardcoded_index_path}")
            print(f"Index statistics: {self.index.getCollectionStatistics().toString()}")
        except Exception as e:
            print(f"Error loading index from {hardcoded_index_path}: {e}")
            raise

        # first‐stage BM25
        bm25 = pt.BatchRetrieve(self.index, wmodel='BM25')
        # RM3 expansion: pass plain Python ints into the constructor
        rm3 = pt.rewrite.RM3(
            self.index,
            fb_terms=int(fb_terms),
            fb_docs=int(fb_docs),
        )
        # no need for manual fb_terms/fb_docs assignment
        self.pipeline = bm25 >> rm3

    def search(self, query: str, k: int = 100) -> list[DocScore]:
        # prepare a one‐row query DF
        qdf = pd.DataFrame([{'query': query, 'qid': 'Q0'}])
        # run RM3 + retrieve
        res = self.pipeline.transform(qdf)
        # take top-k results
        scores = res.nlargest(k, 'score')
        return [DocScore(row['docno'], float(row['score'])) for _, row in scores.iterrows()]