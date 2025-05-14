import unittest
from src.feedback.rocchio import RocchioTrueFeedback
import numpy as np

class test_feedback(unittest.TestCase):
    def test_refine_no_feedback(self):
        fb = RocchioTrueFeedback(qrels={}, alpha=1.0, beta=0.75, gamma=0.15)
        q = np.random.rand(384)
        q2 = fb.refine("dummy", q, {})
        assert np.allclose(q, q2)  # unchanged when no qrels
    

if __name__ == '__main__':
    unittest.main()
