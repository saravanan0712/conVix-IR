# ConViX-IR: Phase 1 — Dataset Verification & Leakage Audit Guide

This guide provides instructions for executing the read-only **Phase 1 Dataset Verification and Leakage Audit** on the peach freshness dataset stored in Google Drive.

---

## 🎯 Purpose of Phase 1

Before training any baseline CNN or multi-modal Vision Transformer (ViT), the dataset must undergo a comprehensive data quality and leakage audit:
1. **Integrity Check:** Detect corrupted or unreadable images across all splits (`train`, `val`, `test`).
2. **Exact Duplicate Detection:** Identify identical files (MD5 matching) crossing split boundaries.
3. **Near-Duplicate Clustering:** Group visually near-identical images (perceptual dHash, 64-bit, Hamming distance $\le 4$) to assess potential physical session overlaps.
4. **Leakage Protocol Evaluation:** Generate an automated verdict (`PASS`, `REVIEW_REQUIRED`, or `FAIL`).

> **Note:** The audit is strictly **READ-ONLY**. It does not move, rename, delete, or modify any dataset images, nor does it create a new random split.

---

## 🚀 Execution in Google Colab

### 1. Mount Google Drive and Clone/Open Project

```python
from google.colab import drive
drive.mount('/content/drive')
```

### 2. Navigate to the Repository Root

```bash
cd /content/drive/MyDrive/conVix-IR  # Or your project directory
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Phase 1 Audit Script

```bash
python src/phase1/phase1_dataset_audit.py /content/drive/MyDrive/conVix-IR/dataset --output_dir phase1_results
```

---

## 📊 Generated Output Artifacts

The audit produces 4 output files in `phase1_results/`:

| File | Description |
| :--- | :--- |
| `phase1_image_inventory.csv` | Full inventory of every verified image (path, split, class, label, dimensions, file size, MD5, dHash). |
| `phase1_duplicate_report.csv` | Itemized report of exact duplicate groups and near-duplicate clusters with cross-split flags. |
| `phase1_audit_report.json` | Comprehensive machine-readable JSON containing full dataset metrics, distributions, and leakage findings. |
| `phase1_summary.txt` | Human-readable executive summary of findings and final status verdict. |

---

## ⚖️ Leakage Status Decision Rules

- **`FAIL`**: Exact duplicate files (identical MD5) exist across different splits (`Train ↔ Val`, `Train ↔ Test`, `Val ↔ Test`).
- **`REVIEW_REQUIRED`**: No exact cross-split duplicates found, but near-duplicate clusters cross split boundaries.
- **`PASS`**: Zero exact or near-duplicate cross-split groups detected.

*(Corrupted/unreadable files are cataloged separately and do not automatically trigger a pipeline failure).*
