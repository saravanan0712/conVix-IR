"""
Unit tests and local verification for Phase 1 Dataset Audit module.
Uses synthetic in-memory/temporary images to verify:
1. Exact duplicate detection.
2. Near-duplicate clustering (dHash).
3. Cross-split leakage evaluation (PASS / REVIEW_REQUIRED / FAIL).
4. Corrupt image handling.
5. Report generation integrity.
"""

import unittest
import tempfile
import shutil
import json
import csv
from pathlib import Path
from PIL import Image, ImageDraw

from phase1_dataset_audit import (
    run_audit,
    compute_md5,
    compute_dhash,
    hamming_distance,
    determine_leakage_status,
)


class TestPhase1DatasetAudit(unittest.TestCase):

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.dataset_root = Path(self.test_dir) / "dataset"
        self.output_dir = Path(self.test_dir) / "phase1_results"

        # Create split and class folders
        for split in ["train", "val", "test"]:
            for cls in ["Fresh_Peach", "Rotten_Peach"]:
                (self.dataset_root / split / cls).mkdir(parents=True, exist_ok=True)

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def _create_synthetic_image(self, path: Path, color=(255, 100, 50), text="Peach"):
        """Create a reproducible synthetic test image."""
        img = Image.new("RGB", (64, 64), color=color)
        draw = ImageDraw.Draw(img)
        draw.text((10, 20), text, fill=(255, 255, 255))
        img.save(path)

    def test_dhash_and_hamming(self):
        img1_path = self.dataset_root / "train" / "Fresh_Peach" / "test1.png"
        img2_path = self.dataset_root / "train" / "Fresh_Peach" / "test2.png"
        
        self._create_synthetic_image(img1_path, color=(200, 50, 50), text="A")
        self._create_synthetic_image(img2_path, color=(200, 50, 50), text="A")

        hex1, int1 = compute_dhash(img1_path)
        hex2, int2 = compute_dhash(img2_path)

        self.assertEqual(len(hex1), 16)
        self.assertEqual(hamming_distance(int1, int2), 0)

    def test_leakage_status_rules(self):
        self.assertEqual(determine_leakage_status(cross_exact_count=0, cross_near_count=0), "PASS")
        self.assertEqual(determine_leakage_status(cross_exact_count=0, cross_near_count=2), "REVIEW_REQUIRED")
        self.assertEqual(determine_leakage_status(cross_exact_count=1, cross_near_count=0), "FAIL")
        self.assertEqual(determine_leakage_status(cross_exact_count=1, cross_near_count=5), "FAIL")

    def test_full_audit_workflow_with_leakage(self):
        # Image 1 in train Fresh
        img_train = self.dataset_root / "train" / "Fresh_Peach" / "peach_01.png"
        self._create_synthetic_image(img_train, color=(220, 120, 80), text="Fresh1")

        # Image 2 in test Fresh (Exact copy of peach_01 -> cross-split exact duplicate)
        img_test_exact = self.dataset_root / "test" / "Fresh_Peach" / "peach_01_copy.png"
        shutil.copyfile(img_train, img_test_exact)

        # Image 3 in val Rotten
        img_val = self.dataset_root / "val" / "Rotten_Peach" / "rotten_01.png"
        self._create_synthetic_image(img_val, color=(80, 50, 30), text="Rotten1")

        # Corrupt file
        corrupt_path = self.dataset_root / "train" / "Rotten_Peach" / "corrupt_img.jpg"
        with open(corrupt_path, "wb") as f:
            f.write(b"NOT_A_VALID_IMAGE_BYTES")

        # Run audit
        results = run_audit(str(self.dataset_root), str(self.output_dir))

        self.assertEqual(results["valid_count"], 3)
        self.assertEqual(results["corrupt_count"], 1)
        self.assertEqual(results["exact_groups"], 1)
        self.assertEqual(results["cross_exact"], 1)
        self.assertEqual(results["status"], "FAIL")

        # Verify output files exist
        inv_file = self.output_dir / "phase1_image_inventory.csv"
        dup_file = self.output_dir / "phase1_duplicate_report.csv"
        json_file = self.output_dir / "phase1_audit_report.json"
        txt_file = self.output_dir / "phase1_summary.txt"

        self.assertTrue(inv_file.exists())
        self.assertTrue(dup_file.exists())
        self.assertTrue(json_file.exists())
        self.assertTrue(txt_file.exists())

        # Inspect json content
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            self.assertEqual(data["leakage_status"], "FAIL")
            self.assertEqual(data["total_valid_images"], 3)
            self.assertEqual(data["total_corrupt_images"], 1)
            self.assertEqual(len(data["cross_split_exact_duplicates"]), 1)


if __name__ == "__main__":
    unittest.main()
