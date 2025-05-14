from pathlib import Path
from pydantic import BaseModel, Field
import yaml

class Config(BaseModel):
    dataset: str = Field(default="trec-dl-2019")
    model_name: str = "intfloat/e5-small"
    metrics: list[str] = ["ndcg_cut_10"]
alpha: float = 1.0
beta: float = 0.75
gamma: float = 0.15

@classmethod
def load(cls, path: Path):
    return cls(**yaml.safe_load(path.read_text()))