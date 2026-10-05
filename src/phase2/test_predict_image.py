import sys
import unittest
import tempfile
import shutil
from pathlib import Path

# Ensure src/phase2 is in sys.path when running from workspace root
_phase2_dir = str(Path(__file__).resolve().parent)
if _phase2_dir not in sys.path:
    sys.path.insert(0, _phase2_dir)

import torch
import numpy as np
from PIL import Image

try:
    from convnext_baseline import ConvNeXtTinyBaseline, CLASS_MAP, ID_TO_CLASS
    from predict_image import load_model, predict_single_image, format_prediction_output
except ImportError:
    from src.phase2.convnext_baseline import ConvNeXtTinyBaseline, CLASS_MAP, ID_TO_CLASS
    from src.phase2.predict_image import load_model, predict_single_image, format_prediction_output


class TestPredictImage(unittest.TestCase):

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.ckpt_path = Path(self.test_dir) / "best_convnext_tiny.pt"
        self.image_path = Path(self.test_dir) / "test_peach.png"

        # Create dummy checkpoint
        model = ConvNeXtTinyBaseline(pretrained=False, num_classes=2)
        torch.save({
            "epoch": 1,
            "model_state_dict": model.state_dict(),
            "val_loss": 0.1234,
            "val_accuracy": 0.95,
        }, self.ckpt_path)

        # Create dummy test image
        img = Image.new("RGB", (300, 300), color=(240, 120, 80))
        img.save(self.image_path)

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_load_model(self):
        """Test model loads from checkpoint successfully in eval mode."""
        model = load_model(self.ckpt_path, device=torch.device("cpu"))
        self.assertIsInstance(model, ConvNeXtTinyBaseline)
        self.assertFalse(model.training)

    def test_single_image_prediction(self):
        """Test single-image prediction returns valid probabilities and class."""
        model = load_model(self.ckpt_path, device=torch.device("cpu"))
        result = predict_single_image(model, self.image_path, device=torch.device("cpu"))

        self.assertIn("predicted_class_id", result)
        self.assertIn("predicted_class_name", result)
        self.assertIn("fresh_probability", result)
        self.assertIn("rotten_probability", result)

        # Check prediction is either 0 or 1
        self.assertIn(result["predicted_class_id"], [0, 1])
        self.assertIn(result["predicted_class_name"], ["Fresh_Peach", "Rotten_Peach"])

        # Check probabilities sum approximately to 1
        prob_sum = result["fresh_probability"] + result["rotten_probability"]
        self.assertAlmostEqual(prob_sum, 1.0, places=5)

        # Check individual probabilities are in [0, 1]
        self.assertTrue(0.0 <= result["fresh_probability"] <= 1.0)
        self.assertTrue(0.0 <= result["rotten_probability"] <= 1.0)

        # Check feature dimension is 768
        self.assertEqual(result["feature_dim"], 768)

    def test_format_prediction_output(self):
        """Test formatting matches expected text output."""
        mock_result = {
            "image_path": str(self.image_path),
            "predicted_class_id": 0,
            "predicted_class_name": "Fresh_Peach",
            "fresh_probability": 0.9876,
            "rotten_probability": 0.0124,
            "logits": [3.5, -1.2],
            "feature_dim": 768,
        }
        output_str = format_prediction_output(mock_result)
        self.assertIn("Prediction: Fresh_Peach", output_str)
        self.assertIn("Predicted class: 0", output_str)
        self.assertIn("Fresh probability:  98.7600%", output_str)
        self.assertIn("Rotten probability: 1.2400%", output_str)


if __name__ == "__main__":
    unittest.main()
