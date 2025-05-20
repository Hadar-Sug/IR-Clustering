from pathlib import Path
from pydantic import BaseModel, Field
import yaml

class Config(BaseModel):
    # Paths to each of the three inputs for TREC DL-2019
    queries_path: str = Field(
        default="msmarco-test2019-queries.tsv.gz",
        description="Gzipped TSV of query-id<TAB>query text"
    )
    top100_path: str = Field(
        default="msmarco-doctest2019-top100.gz",
        description="Gzipped run file: qid Q0 docid rank score runname"
    )
    qrels_path: str = Field(
        default="2019qrels-docs.txt",
        description="Plain text qrels: qid Q0 docid rating"
    )
    test_docs_path: str = Field(
        default="msmarco-docs.tsv.gz",
    )
    
    index_path: str = Field(
        default="faiss.index",
        description="Path to save/load FAISS index"
    )

    # Model & evaluation parameters
    model_name: str = Field(default="intfloat/e5-small")
    metrics: list[str] = Field(default_factory=lambda: ["ndcg_cut_10", "recip_rank", "map"])
    alpha: float = Field(default=1.0)
    beta: float = Field(default=0.75)
    gamma: float = Field(default=0.15)

    @classmethod
    def load(cls, path: Path) -> "Config":
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
        data = yaml.safe_load(path.read_text()) or {}
        if not isinstance(data, dict):
            raise TypeError(
                "Top-level YAML item must be a mapping "
                f"(got {type(data).__name__!s})."
            )
        return cls(**data)