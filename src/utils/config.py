from pathlib import Path
from pydantic import BaseModel, Field
import yaml

class Config(BaseModel):
    queries_file: str = Field(default="irds/msmarco-document-v2_trec-dl-2019")
    qrels_file: str = Field(default="irds/msmarco-document-v2_trec-dl-2019")
    docs_file: str = Field(default="irds/msmarco-document-v2")
    model_name: str = "intfloat/e5-small"
    metrics: list[str] = Field(default_factory=lambda: ["ndcg_cut_10"])
    alpha: float = 1.0
    beta: float = 0.75
    gamma: float = 0.15

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