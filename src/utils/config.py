from pathlib import Path
from pydantic import BaseModel, Field
import yaml
from typing import List, Dict, Optional

class Config(BaseModel):
    queries_path: str = Field(
        default="msmarco-test2019-queries.tsv.gz",
        description="Gzipped TSV of query-id<TAB>query text"
    )
    top100_path: str = Field(
        default="msmarco-doctest2019-top100.gz",
        description="Gzipped run file: qid Q0 docid rank score"
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
    pt_index_path: str = Field(
        default="pyterrier_index_19",
        description="Directory of the PyTerrier index"
    )

    # ➡️ New field for JSON output file
    docs_json_path: str = Field(
        default="msmarco-docs-output.json",
        description="Path to store/load the docs-derived JSON file"
    )

    model_name: str = Field(default="intfloat/e5-small")
    metrics: list[str] = Field(default_factory=lambda: ["ndcg_cut_10", "ndcg_cut_100", "recip_rank", "map"])
    alpha: float = Field(default=1.0)
    beta: float = Field(default=0.75)
    gamma: float = Field(default=0.15)

    # ----- Cross validation params -----
    cv_folds: int = Field(default=5, description="Number of folds for CV")

    # dev set paths
    dev_queries_path: Optional[str] = None
    dev_top100_path: Optional[str] = None
    dev_qrels_path: Optional[str] = None

    # RM3 search grid
    rm3_fb_docs: List[int] = Field(
        default_factory=lambda: [5, 10, 20, 50, 100]
    )
    rm3_fb_terms: List[int] = Field(
        default_factory=lambda: [10, 20, 50, 100]
    )
    rm3_fb_lambda: List[float] = Field(
        default_factory=lambda: [0.3, 0.5, 0.7, 0.8, 0.9]
    )

    # Embedding feedback grid
    rocchio_alpha: List[float] = Field(
        default_factory=lambda: [0.5, 0.8, 1.0, 1.2]
    )
    rocchio_beta: List[float] = Field(
        default_factory=lambda: [0.5, 0.75, 1.0, 1.25]
    )
    rocchio_k: List[int] = Field(
        default_factory=lambda: [5, 10, 20, 50, 100]
    )

    # output files for cross validation and evaluation
    rm3_results_path: str = Field(default="rm3_cv_results.csv")
    rocchio_results_path: str = Field(default="rocchio_cv_results.csv")
    test_results_path: str = Field(default="test_results.csv")

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
        # Automatically set json_output_path if not present, based on docs path
        if "json_output_path" not in data and "test_docs_path" in data:
            docs_path = Path(data["test_docs_path"])
            data["json_output_path"] = str(docs_path.with_suffix('.json'))
        return cls(**data)
