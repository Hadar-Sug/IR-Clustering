from dataclasses import dataclass


@dataclass
class DocScore:
    doc_id: str
    score: float
