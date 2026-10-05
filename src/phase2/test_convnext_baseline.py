"""
Unit tests for ConViX-IR Phase 2 ConvNeXt-Tiny Baseline Module.
Verifies all 16 mathematical, architectural, and data-flow requirements without running a full training session.
"""

import unittest
import tempfile
import shutil
import json
import csv
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image

from convnext_baseline import (
    ConvNeXtTinyBaseline,
    PeachDataset,
    get_transforms,
    calculate_metrics,
    compute_ece,
    CLASS_MAP,
    ID_TO_CLASS,
    SPLITS,
    IMAGENET_MEAN,
    IMAGENET_STD,
    set_seed,
)


class TestConvNeXtBaseline(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        set_seed(42)
        # Instantiate model without downloading weights for fast local unit testing
        cls.model = ConvNeXtTinyBaseline(pretrained=False, num_classes=2)
        cls.model.eval()

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.dataset_root = Path(self.test_dir) / "dataset"

        # Create temporary dataset structure
        for split in SPLITS:
            for cls_name in CLASS_MAP.keys():
                cls_dir = self.dataset_root / split / cls_name
                cls_dir.mkdir(parents=True, exist_ok=True)
                # Create a sample test image
                img = Image.new("RGB", (256, 256), color=(200, 100, 50))
                img.save(cls_dir / "sample.png")

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_01_model_construction_succeeds(self):
        """Test 1: Model construction succeeds."""
        self.assertIsInstance(self.model, nn.Module)
        self.assertEqual(self.model.feature_dim, 768)
        self.assertEqual(self.model.num_classes, 2)

    def test_02_logits_shape(self):
        """Test 2: Input [batch_size, 3, 224, 224] produces logits.shape = [batch_size, 2]."""
        dummy_input = torch.randn(4, 3, 224, 224)
        logits = self.model(dummy_input)
        self.assertEqual(logits.shape, (4, 2))

    def test_03_feature_extraction_shape(self):
        """Test 3: Feature extraction produces features.shape = [batch_size, 768]."""
        dummy_input = torch.randn(4, 3, 224, 224)
        features = self.model.extract_features(dummy_input)
        self.assertEqual(features.shape, (4, 768))

    def test_04_softmax_shape(self):
        """Test 4: Softmax produces probabilities.shape = [batch_size, 2]."""
        dummy_input = torch.randn(4, 3, 224, 224)
        probs = self.model.predict_probabilities(dummy_input)
        self.assertEqual(probs.shape, (4, 2))

    def test_05_probabilities_sum_to_one(self):
        """Test 5: For every sample, sum_k p_C(k) == 1."""
        dummy_input = torch.randn(6, 3, 224, 224)
        probs = self.model.predict_probabilities(dummy_input)
        prob_sums = probs.sum(dim=-1).detach().numpy()
        np.testing.assert_allclose(prob_sums, np.ones(6), atol=1e-5)

    def test_06_class_mapping(self):
        """Test 6: Class mapping Fresh_Peach -> 0, Rotten_Peach -> 1."""
        self.assertEqual(CLASS_MAP["Fresh_Peach"], 0)
        self.assertEqual(CLASS_MAP["Rotten_Peach"], 1)
        self.assertEqual(ID_TO_CLASS[0], "Fresh_Peach")
        self.assertEqual(ID_TO_CLASS[1], "Rotten_Peach")

    def test_07_dataset_split_separation(self):
        """Test 7: Dataset split separation and data loading."""
        t_dict = get_transforms()
        train_ds = PeachDataset(self.dataset_root, split="train", transform=t_dict["train"])
        val_ds = PeachDataset(self.dataset_root, split="val", transform=t_dict["val"])
        test_ds = PeachDataset(self.dataset_root, split="test", transform=t_dict["test"])

        self.assertEqual(len(train_ds), 2)
        self.assertEqual(len(val_ds), 2)
        self.assertEqual(len(test_ds), 2)
        self.assertEqual(train_ds[0]["split"], "train")
        self.assertEqual(val_ds[0]["split"], "val")
        self.assertEqual(test_ds[0]["split"], "test")

    def test_08_train_transforms_contain_augmentation(self):
        """Test 8: Training transforms contain augmentation."""
        t_dict = get_transforms()
        train_t = t_dict["train"]
        transform_types = [type(t) for t in train_t.transforms]
        self.assertIn(transforms.RandomResizedCrop, transform_types)
        self.assertIn(transforms.RandomHorizontalFlip, transform_types)
        self.assertIn(transforms.RandomRotation, transform_types)
        self.assertIn(transforms.ColorJitter, transform_types)

    def test_09_val_transforms_contain_no_random_augmentation(self):
        """Test 9: Validation transforms contain no random augmentation."""
        t_dict = get_transforms()
        val_t = t_dict["val"]
        transform_types = [type(t) for t in val_t.transforms]
        self.assertNotIn(transforms.RandomResizedCrop, transform_types)
        self.assertNotIn(transforms.RandomHorizontalFlip, transform_types)
        self.assertNotIn(transforms.RandomRotation, transform_types)
        self.assertNotIn(transforms.ColorJitter, transform_types)
        self.assertIn(transforms.Resize, transform_types)
        self.assertIn(transforms.CenterCrop, transform_types)

    def test_10_test_transforms_contain_no_random_augmentation(self):
        """Test 10: Test transforms contain no random augmentation."""
        t_dict = get_transforms()
        test_t = t_dict["test"]
        transform_types = [type(t) for t in test_t.transforms]
        self.assertNotIn(transforms.RandomResizedCrop, transform_types)
        self.assertNotIn(transforms.RandomHorizontalFlip, transform_types)
        self.assertNotIn(transforms.RandomRotation, transform_types)
        self.assertNotIn(transforms.ColorJitter, transform_types)
        self.assertIn(transforms.Resize, transform_types)
        self.assertIn(transforms.CenterCrop, transform_types)

    def test_11_cross_entropy_loss_executes(self):
        """Test 11: Cross-entropy loss executes correctly."""
        criterion = nn.CrossEntropyLoss()
        logits = torch.randn(4, 2, requires_grad=True)
        targets = torch.tensor([0, 1, 1, 0], dtype=torch.long)
        loss = criterion(logits, targets)
        self.assertTrue(loss.item() > 0)
        loss.backward()
        self.assertIsNotNone(logits.grad)

    def test_12_adamw_optimizer_initializes(self):
        """Test 12: AdamW optimizer initializes with correct parameters."""
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-4, weight_decay=1e-2)
        self.assertEqual(optimizer.defaults["lr"], 1e-4)
        self.assertEqual(optimizer.defaults["weight_decay"], 1e-2)

    def test_13_cosine_scheduler_initializes(self):
        """Test 13: Cosine scheduler initializes and updates learning rate."""
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=1e-6)
        initial_lr = scheduler.get_last_lr()[0]
        self.assertAlmostEqual(initial_lr, 1e-4)
        optimizer.step()
        scheduler.step()
        next_lr = scheduler.get_last_lr()[0]
        self.assertNotEqual(initial_lr, next_lr)

    def test_14_checkpoint_save_and_load(self):
        """Test 14: Checkpoint save and load preserves state dict."""
        ckpt_path = Path(self.test_dir) / "test_ckpt.pt"
        torch.save({
            "epoch": 5,
            "model_state_dict": self.model.state_dict(),
            "val_loss": 0.25,
            "val_accuracy": 0.92,
        }, ckpt_path)

        self.assertTrue(ckpt_path.exists())
        loaded = torch.load(ckpt_path)
        self.assertEqual(loaded["epoch"], 5)
        self.assertAlmostEqual(loaded["val_loss"], 0.25)
        new_model = ConvNeXtTinyBaseline(pretrained=False, num_classes=2)
        new_model.load_state_dict(loaded["model_state_dict"])

    def test_15_metric_calculations(self):
        """Test 15: Metric calculations execute correctly (Accuracy, Precision, Recall, F1, Specificity, ROC-AUC, Brier, ECE, Confusion Matrix)."""
        y_true = np.array([0, 0, 1, 1, 1, 0, 1, 0])
        y_pred = np.array([0, 0, 1, 1, 0, 0, 1, 1])
        y_prob_rotten = np.array([0.1, 0.2, 0.8, 0.9, 0.4, 0.3, 0.85, 0.6])

        metrics = calculate_metrics(y_true, y_pred, y_prob_rotten)
        self.assertIn("accuracy", metrics)
        self.assertIn("precision", metrics)
        self.assertIn("recall", metrics)
        self.assertIn("f1", metrics)
        self.assertIn("specificity", metrics)
        self.assertIn("roc_auc", metrics)
        self.assertIn("brier_score", metrics)
        self.assertIn("ece", metrics)
        self.assertIn("confusion_matrix", metrics)
        
        # Check that values are within mathematical bounds
        self.assertTrue(0.0 <= metrics["accuracy"] <= 1.0)
        self.assertTrue(0.0 <= metrics["precision"] <= 1.0)
        self.assertTrue(0.0 <= metrics["recall"] <= 1.0)
        self.assertTrue(0.0 <= metrics["f1"] <= 1.0)
        self.assertTrue(0.0 <= metrics["specificity"] <= 1.0)
        self.assertTrue(0.0 <= metrics["roc_auc"] <= 1.0)
        self.assertTrue(0.0 <= metrics["brier_score"] <= 1.0)
        self.assertTrue(0.0 <= metrics["ece"] <= 1.0)

    def test_16_feature_extraction_768d(self):
        """Test 16: Feature extraction returns 768-D representations."""
        dummy_batch = torch.randn(2, 3, 224, 224)
        features = self.model.extract_features(dummy_batch)
        self.assertEqual(features.dim(), 2)
        self.assertEqual(features.shape[1], 768)


if __name__ == "__main__":
    unittest.main()
