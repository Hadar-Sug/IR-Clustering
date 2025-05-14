from typing import Dict
import pytrec_eval
from ..domain.interfaces import Evaluator

class TrecEvaluator(Evaluator):
    def __init__(self, metrics):
        self.metrics = set(metrics)

    def evaluate(self, run: Dict[str, Dict[str, float]], qrels: Dict[str, Dict[str, int]]):
        evaluator = pytrec_eval.RelevanceEvaluator(qrels, self.metrics)
        return evaluator.evaluate(run)