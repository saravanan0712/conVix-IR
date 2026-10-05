"""
Unit tests for ConViX-IR Phase 2 Interactive UI Module (app.py).
Verifies that:
1. The inference function accepts an image.
2. The output contains a valid class prediction (FRESH or ROTTEN).
3. Fresh probability is between 0 and 1.
4. Rotten probability is between 0 and 1.
5. Fresh probability + Rotten probability ~= 1.
6. The class mapping remains Fresh_Peach = 0, Rotten_Peach = 1.
"""

import sys
import unittest
import tempfile
import shutil
from pathlib import Path

# Ensure src/phase2 is in sys.path
_phase2_dir = str(Path(__file__).resolve().parent)
if _phase2_dir not in sys.path:
    sys.path.insert(0, _phase2_dir)

import torch
import numpy as np
from PIL import Image

try:
    from convnext_baseline import ConvNeXtTinyBaseline, CLASS_MAP, ID_TO_CLASS
    from app import predict_freshness, predict_for_gradio
except ImportError:
    from src.phase2.convnext_baseline import ConvNeXtTinyBaseline, CLASS_MAP, ID_TO_CLASS
    from src.phase2.app import predict_freshness, predict_for_gradio


class TestApp(unittest.TestCase):

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.ckpt_path = Path(self.test_dir) / "best_convnext_tiny.pt"
        self.image_path = Path(self.test_dir) / "sample_peach.jpg"

        # Create dummy checkpoint
        self.model = ConvNeXtTinyBaseline(pretrained=False, num_classes=2)
        torch.save({
            "epoch": 1,
            "model_state_dict": self.model.state_dict(),
            "val_loss": 0.15,
            "val_accuracy": 0.96,
        }, self.ckpt_path)

        # Create synthetic test image
        self.pil_image = Image.new("RGB", (256, 256), color=(255, 140, 90))
        self.pil_image.save(self.image_path)

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_01_predict_freshness_with_pil_image(self):
        """Test 1: Inference function accepts a PIL image and returns valid prediction."""
        res = predict_freshness(
            image_input=self.pil_image,
            checkpoint_path=self.ckpt_path,
            model=self.model,
            device=torch.device("cpu"),
        )
        self.assertIn(res["prediction_display"], ["FRESH", "ROTTEN"])
        self.assertIn(res["class_name"], ["Fresh_Peach", "Rotten_Peach"])
        self.assertIn(res["class_id"], [0, 1])

    def test_02_probabilities_in_valid_range(self):
        """Test 2 & 3: Fresh and Rotten probabilities are bounded in [0, 1]."""
        res = predict_freshness(
            image_input=self.pil_image,
            model=self.model,
            device=torch.device("cpu"),
        )
        self.assertTrue(0.0 <= res["fresh_probability"] <= 1.0)
        self.assertTrue(0.0 <= res["rotten_probability"] <= 1.0)

    def test_03_probabilities_sum_to_one(self):
        """Test 4: Fresh probability + Rotten probability ~= 1."""
        res = predict_freshness(
            image_input=self.pil_image,
            model=self.model,
            device=torch.device("cpu"),
        )
        prob_sum = res["fresh_probability"] + res["rotten_probability"]
        self.assertAlmostEqual(prob_sum, 1.0, places=5)

    def test_04_class_mapping_preserved(self):
        """Test 5: Class mapping remains Fresh_Peach = 0, Rotten_Peach = 1."""
        self.assertEqual(CLASS_MAP["Fresh_Peach"], 0)
        self.assertEqual(CLASS_MAP["Rotten_Peach"], 1)
        self.assertEqual(ID_TO_CLASS[0], "Fresh_Peach")
        self.assertEqual(ID_TO_CLASS[1], "Rotten_Peach")

    def test_05_predict_for_gradio_wrapper(self):
        """Test 6: Gradio callback wrapper returns 3 formatted strings."""
        pred_label, fresh_str, rotten_str = predict_for_gradio(
            self.pil_image,
            checkpoint_path=str(self.ckpt_path),
        )
        self.assertIn(pred_label, ["FRESH", "ROTTEN"])
        self.assertTrue(fresh_str.endswith("%"))
        self.assertTrue(rotten_str.endswith("%"))


if __name__ == "__main__":
    unittest.main()
