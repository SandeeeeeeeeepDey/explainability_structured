import unittest
from src.inference import make_inference

class TestInference(unittest.TestCase):
    def test_inference(self):
        result = make_inference('65ab69d5c4a23e899548a7d1')
        self.assertIn('predicted_prob', result)
        self.assertIn('prediction_label', result)

if __name__ == "__main__":
    unittest.main()
