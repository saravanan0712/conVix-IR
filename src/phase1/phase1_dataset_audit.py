"""
ConViX-IR: Phase 1 — Dataset Verification and Leakage Audit
===========================================================
Context-Verified Iterative Re-Inspection CNN–ViT Framework for
Calibrated Peach Freshness Assessment.

This module performs a strict, read-only data quality and integrity audit:
1. Recursive scan of train, val, and test splits.
2. Image integrity verification and dimension extraction via PIL.
3. MD5 exact duplicate detection across splits.
4. Perceptual dHash (64-bit, Hamming distance <= 4) near-duplicate clustering.
5. Cross-split data leakage detection (Train ↔ Val, Train ↔ Test, Val ↔ Test).
6. Generation of inventory CSV, duplicate report CSV, audit JSON, and summary TXT.

Label mapping:
    Fresh_Peach  -> 0
    Rotten_Peach -> 1 (Positive Class)
"""

import os
import sys
import argparse
import hashlib
import json
import csv
from pathlib import Path
from typing import List, Dict, Tuple, Set, Optional, Any
from collections import defaultdict

from PIL import Image

# Supported image file extensions
SUPPORTED_EXTENSIONS: Set[str] = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Standard class mapping
CLASS_MAP: Dict[str, int] = {
    "Fresh_Peach": 0,
    "Rotten_Peach": 1,
}

SPLITS: List[str] = ["train", "val", "test"]


class UnionFind:
    """Disjoint Set Union (Union-Find) for deterministic near-duplicate clustering."""

    def __init__(self, elements: List[str]):
        self.parent: Dict[str, str] = {elem: elem for elem in elements}
        self.rank: Dict[str, int] = {elem: 0 for elem in elements}

    def find(self, item: str) -> str:
        if self.parent[item] != item:
            self.parent[item] = self.find(self.parent[item])
        return self.parent[item]

    def union(self, item1: str, item2: str) -> None:
        root1 = self.find(item1)
        root2 = self.find(item2)
        if root1 == root2:
            return
        if self.rank[root1] < self.rank[root2]:
            self.parent[root1] = root2
        elif self.rank[root1] > self.rank[root2]:
            self.parent[root2] = root1
        else:
            self.parent[root2] = root1
            self.rank[root1] += 1


def compute_md5(file_path: Path, chunk_size: int = 65536) -> str:
    """Compute MD5 hash for exact file content comparison."""
    md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        while chunk := f.read(chunk_size):
            md5.update(chunk)
    return md5.hexdigest()


def compute_dhash(image_path: Path, hash_size: int = 8) -> Tuple[str, int]:
    """
    Compute 64-bit perceptual difference hash (dHash) using PIL.
    Returns:
        (hex_string, integer_hash)
    """
    with Image.open(image_path) as img:
        # Convert to grayscale and resize to (hash_size + 1, hash_size)
        resample_filter = getattr(Image, "Resampling", Image).BILINEAR
        gray_img = img.convert("L").resize((hash_size + 1, hash_size), resample=resample_filter)
        if hasattr(gray_img, "get_flattened_data"):
            pixels = list(gray_img.get_flattened_data())
        else:
            pixels = list(gray_img.getdata())
        
        diff_bits = []
        width = hash_size + 1
        for row in range(hash_size):
            row_start = row * width
            for col in range(hash_size):
                left_pixel = pixels[row_start + col]
                right_pixel = pixels[row_start + col + 1]
                diff_bits.append(1 if left_pixel > right_pixel else 0)
        
        # Pack 64 bits into integer
        hash_int = 0
        for bit in diff_bits:
            hash_int = (hash_int << 1) | bit
            
        hex_str = f"{hash_int:016x}"
        return hex_str, hash_int


def hamming_distance(h1: int, h2: int) -> int:
    """Compute Hamming distance between two 64-bit integers."""
    try:
        return (h1 ^ h2).bit_count()
    except AttributeError:
        return bin(h1 ^ h2).count("1")


def scan_and_verify_dataset(dataset_root: Path) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Recursively scan dataset splits and verify each image file with PIL.
    Returns (valid_records, corrupt_records).
    """
    valid_records: List[Dict[str, Any]] = []
    corrupt_records: List[Dict[str, Any]] = []

    # Sort to ensure absolute determinism
    for split in SPLITS:
        split_dir = dataset_root / split
        if not split_dir.exists():
            continue

        for class_name, label_id in sorted(CLASS_MAP.items()):
            class_dir = split_dir / class_name
            if not class_dir.exists():
                continue

            # Gather all candidate image files deterministically
            file_candidates = sorted([
                f for f in class_dir.rglob("*") 
                if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS
            ])

            for file_path in file_candidates:
                rel_path = file_path.relative_to(dataset_root).as_posix()
                file_size = file_path.stat().st_size

                try:
                    # Verify image integrity
                    with Image.open(file_path) as img:
                        img.verify()
                    
                    # Re-open to extract metadata (verify closes the image stream)
                    with Image.open(file_path) as img:
                        width, height = img.size
                    
                    # Compute hashes
                    md5_val = compute_md5(file_path)
                    dhash_hex, dhash_int = compute_dhash(file_path)

                    valid_records.append({
                        "path": rel_path,
                        "abs_path": str(file_path.resolve()),
                        "split": split,
                        "class": class_name,
                        "label": label_id,
                        "width": width,
                        "height": height,
                        "file_size_bytes": file_size,
                        "md5": md5_val,
                        "dhash": dhash_hex,
                        "dhash_int": dhash_int,
                    })

                except Exception as exc:
                    corrupt_records.append({
                        "path": rel_path,
                        "abs_path": str(file_path.resolve()),
                        "split": split,
                        "class": class_name,
                        "label": label_id,
                        "error": str(exc),
                        "file_size_bytes": file_size,
                    })

    return valid_records, corrupt_records


def detect_exact_duplicates(valid_records: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Detect exact file duplicates via MD5.
    Returns (duplicate_groups, cross_split_duplicates).
    """
    md5_to_records = defaultdict(list)
    for rec in valid_records:
        md5_to_records[rec["md5"]].append(rec)

    duplicate_groups = []
    cross_split_duplicates = []
    cluster_idx = 0

    for md5_val, records in sorted(md5_to_records.items()):
        if len(records) > 1:
            cluster_idx += 1
            splits_in_group = sorted(list(set(r["split"] for r in records)))
            is_cross_split = len(splits_in_group) > 1
            
            group_info = {
                "cluster_id": f"EXACT_{cluster_idx:04d}",
                "md5": md5_val,
                "count": len(records),
                "splits": splits_in_group,
                "cross_split": is_cross_split,
                "items": records,
            }
            duplicate_groups.append(group_info)
            if is_cross_split:
                cross_split_duplicates.append(group_info)

    return duplicate_groups, cross_split_duplicates


def detect_near_duplicates(valid_records: List[Dict[str, Any]], max_hamming_dist: int = 4) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Cluster near-duplicates using perceptual dHash and Hamming distance <= max_hamming_dist.
    Returns (near_duplicate_clusters, cross_split_clusters).
    """
    paths = [r["path"] for r in valid_records]
    record_by_path = {r["path"]: r for r in valid_records}
    uf = UnionFind(paths)

    n = len(valid_records)
    # Pairwise comparison (deterministic order)
    for i in range(n):
        r1 = valid_records[i]
        h1 = r1["dhash_int"]
        p1 = r1["path"]
        for j in range(i + 1, n):
            r2 = valid_records[j]
            h2 = r2["dhash_int"]
            p2 = r2["path"]
            dist = hamming_distance(h1, h2)
            if dist <= max_hamming_dist:
                uf.union(p1, p2)

    # Group records by root
    clusters_map = defaultdict(list)
    for p in paths:
        root = uf.find(p)
        clusters_map[root].append(record_by_path[p])

    near_clusters = []
    cross_split_clusters = []
    cluster_idx = 0

    for root_path, group in sorted(clusters_map.items()):
        if len(group) > 1:
            cluster_idx += 1
            splits = sorted(list(set(r["split"] for r in group)))
            classes = sorted(list(set(r["class"] for r in group)))
            is_cross_split = len(splits) > 1

            cluster_info = {
                "cluster_id": f"NEAR_{cluster_idx:04d}",
                "count": len(group),
                "splits": splits,
                "classes": classes,
                "cross_split": is_cross_split,
                "items": group,
            }
            near_clusters.append(cluster_info)
            if is_cross_split:
                cross_split_clusters.append(cluster_info)

    return near_clusters, cross_split_clusters


def compute_statistics(valid_records: List[Dict[str, Any]], corrupt_records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Calculate aggregate counts, distributions, and image resolution statistics."""
    split_counts = defaultdict(int)
    class_counts = defaultdict(int)
    class_counts_by_split = {s: defaultdict(int) for s in SPLITS}

    widths = []
    heights = []

    for r in valid_records:
        split_counts[r["split"]] += 1
        class_counts[r["class"]] += 1
        class_counts_by_split[r["split"]][r["class"]] += 1
        widths.append(r["width"])
        heights.append(r["height"])

    dim_stats = {}
    if widths and heights:
        dim_stats = {
            "min_width": min(widths),
            "max_width": max(widths),
            "min_height": min(heights),
            "max_height": max(heights),
            "min_resolution": f"{min(widths)}x{min(heights)}",
            "max_resolution": f"{max(widths)}x{max(heights)}",
        }

    return {
        "total_valid_images": len(valid_records),
        "total_corrupt_images": len(corrupt_records),
        "split_counts": dict(split_counts),
        "class_counts": dict(class_counts),
        "class_counts_by_split": {s: dict(c) for s, c in class_counts_by_split.items()},
        "image_dimension_statistics": dim_stats,
    }


def determine_leakage_status(cross_exact_count: int, cross_near_count: int) -> str:
    """
    Evaluate protocol leakage rules:
    - FAIL: Exact duplicate across different splits.
    - REVIEW_REQUIRED: Near-duplicates cross splits without exact match.
    - PASS: No exact or near-duplicate cross-split groups.
    """
    if cross_exact_count > 0:
        return "FAIL"
    elif cross_near_count > 0:
        return "REVIEW_REQUIRED"
    else:
        return "PASS"


def generate_reports(
    dataset_root: Path,
    output_dir: Path,
    valid_records: List[Dict[str, Any]],
    corrupt_records: List[Dict[str, Any]],
    exact_groups: List[Dict[str, Any]],
    cross_exact: List[Dict[str, Any]],
    near_clusters: List[Dict[str, Any]],
    cross_near: List[Dict[str, Any]],
    stats: Dict[str, Any],
    leakage_status: str,
) -> None:
    """Write all 4 required Phase 1 audit artifact files."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. phase1_image_inventory.csv
    inventory_path = output_dir / "phase1_image_inventory.csv"
    inv_fields = ["path", "split", "class", "label", "width", "height", "file_size_bytes", "md5", "dhash"]
    with open(inventory_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=inv_fields)
        writer.writeheader()
        for r in valid_records:
            writer.writerow({k: r[k] for k in inv_fields})

    # 2. phase1_duplicate_report.csv
    dup_report_path = output_dir / "phase1_duplicate_report.csv"
    dup_fields = ["duplicate_type", "cluster_id", "path", "split", "class", "md5", "dhash", "cross_split"]
    with open(dup_report_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=dup_fields)
        writer.writeheader()
        
        # Exact duplicates
        for grp in exact_groups:
            for item in grp["items"]:
                writer.writerow({
                    "duplicate_type": "EXACT_MD5",
                    "cluster_id": grp["cluster_id"],
                    "path": item["path"],
                    "split": item["split"],
                    "class": item["class"],
                    "md5": item["md5"],
                    "dhash": item["dhash"],
                    "cross_split": grp["cross_split"],
                })

        # Near duplicates
        for clus in near_clusters:
            for item in clus["items"]:
                writer.writerow({
                    "duplicate_type": "NEAR_DHASH_HAMMING_4",
                    "cluster_id": clus["cluster_id"],
                    "path": item["path"],
                    "split": item["split"],
                    "class": item["class"],
                    "md5": item["md5"],
                    "dhash": item["dhash"],
                    "cross_split": clus["cross_split"],
                })

    # 3. phase1_audit_report.json
    audit_json_path = output_dir / "phase1_audit_report.json"
    audit_data = {
        "project": "ConViX-IR",
        "phase": "Phase 1 — Dataset Verification and Leakage Audit",
        "dataset_root": str(dataset_root),
        "total_valid_images": stats["total_valid_images"],
        "total_corrupt_images": stats["total_corrupt_images"],
        "corrupt_files": corrupt_records,
        "class_counts": stats["class_counts"],
        "split_counts": stats["split_counts"],
        "class_counts_by_split": stats["class_counts_by_split"],
        "image_dimension_statistics": stats["image_dimension_statistics"],
        "exact_duplicate_statistics": {
            "total_duplicate_groups": len(exact_groups),
            "total_duplicate_files": sum(g["count"] for g in exact_groups),
            "cross_split_exact_groups_count": len(cross_exact),
        },
        "near_duplicate_statistics": {
            "dhash_size": 8,
            "hamming_distance_threshold": 4,
            "total_near_duplicate_clusters": len(near_clusters),
            "total_near_duplicate_files": sum(c["count"] for c in near_clusters),
            "cross_split_near_clusters_count": len(cross_near),
        },
        "cross_split_exact_duplicates": [
            {
                "cluster_id": g["cluster_id"],
                "md5": g["md5"],
                "splits": g["splits"],
                "paths": [it["path"] for it in g["items"]],
            } for g in cross_exact
        ],
        "cross_split_near_duplicates": [
            {
                "cluster_id": c["cluster_id"],
                "splits": c["splits"],
                "classes": c["classes"],
                "paths": [it["path"] for it in c["items"]],
            } for c in cross_near
        ],
        "group_information": {
            "true_metadata_available": False,
            "note": "Near-duplicate clusters are used as a proxy for possible shared physical peach/capture session because true peach/session identifiers are unavailable."
        },
        "leakage_status": leakage_status,
    }

    with open(audit_json_path, "w", encoding="utf-8") as f:
        json.dump(audit_data, f, indent=2)

    # 4. phase1_summary.txt
    summary_path = output_dir / "phase1_summary.txt"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("================================================================================\n")
        f.write("ConViX-IR: PHASE 1 DATASET AUDIT & LEAKAGE VERIFICATION SUMMARY\n")
        f.write("================================================================================\n\n")
        f.write(f"Dataset Root: {dataset_root}\n")
        f.write(f"Audit Status: {leakage_status}\n\n")
        
        f.write("--- DATASET TOTALS ---\n")
        f.write(f"Total Valid Images:    {stats['total_valid_images']}\n")
        f.write(f"Total Corrupt Images:  {stats['total_corrupt_images']}\n")
        for split, count in sorted(stats["split_counts"].items()):
            f.write(f"  - Split '{split}': {count}\n")
        f.write("\n--- CLASS DISTRIBUTION ---\n")
        for cls, count in sorted(stats["class_counts"].items()):
            f.write(f"  - Class '{cls}' (label {CLASS_MAP.get(cls, '?')}): {count}\n")
        
        f.write("\n--- SPLIT x CLASS BREAKDOWN ---\n")
        for split in SPLITS:
            cls_breakdown = stats["class_counts_by_split"].get(split, {})
            fresh = cls_breakdown.get("Fresh_Peach", 0)
            rotten = cls_breakdown.get("Rotten_Peach", 0)
            total = fresh + rotten
            f.write(f"  - {split.upper():<5}: Total={total:<4} | Fresh_Peach={fresh:<4} | Rotten_Peach={rotten:<4}\n")

        dim = stats.get("image_dimension_statistics", {})
        if dim:
            f.write("\n--- RESOLUTION RANGE ---\n")
            f.write(f"  Min Resolution: {dim.get('min_resolution')}\n")
            f.write(f"  Max Resolution: {dim.get('max_resolution')}\n")

        f.write("\n--- DUPLICATE & LEAKAGE ANALYSIS ---\n")
        f.write(f"Exact Duplicate Groups (MD5):                 {len(exact_groups)}\n")
        f.write(f"Cross-Split Exact Duplicates:                 {len(cross_exact)}\n")
        f.write(f"Near-Duplicate Clusters (dHash Hamming <= 4): {len(near_clusters)}\n")
        f.write(f"Cross-Split Near-Duplicate Clusters:          {len(cross_near)}\n")

        f.write("\n--- GROUP & METADATA PROXY NOTE ---\n")
        f.write("Near-duplicate clusters are used as a proxy for possible shared physical peach/capture\n")
        f.write("session because true peach/session identifiers are unavailable.\n")
        f.write("The existing train/val/test split was audited in place (READ-ONLY).\n\n")

        f.write("--- FINAL VERDICT ---\n")
        if leakage_status == "PASS":
            f.write("PASS: No exact or near-duplicate cross-split groups detected.\n")
        elif leakage_status == "REVIEW_REQUIRED":
            f.write("REVIEW_REQUIRED: No exact duplicates crossed splits, but near-duplicate\n")
            f.write("clusters cross split boundaries. Inspect phase1_duplicate_report.csv for details.\n")
        else:
            f.write("FAIL: Exact duplicate files cross train/val/test split boundaries.\n")
        f.write("================================================================================\n")


def run_audit(dataset_root_path: str, output_dir_path: str = "phase1_results") -> Dict[str, Any]:
    """Execute complete Phase 1 dataset audit protocol."""
    dataset_root = Path(dataset_root_path).resolve()
    output_dir = Path(output_dir_path).resolve()

    print(f"[*] Starting ConViX-IR Phase 1 Dataset Audit")
    print(f"[*] Dataset Root: {dataset_root}")
    print(f"[*] Output Directory: {output_dir}")

    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root directory does not exist: {dataset_root}")

    # 1. Scan and verify
    print("[*] Scanning images across train, val, and test splits...")
    valid_records, corrupt_records = scan_and_verify_dataset(dataset_root)
    print(f"    -> Found {len(valid_records)} valid images and {len(corrupt_records)} corrupt images.")

    # 2. Compute statistics
    stats = compute_statistics(valid_records, corrupt_records)

    # 3. Exact duplicates
    print("[*] Checking for exact duplicates (MD5)...")
    exact_groups, cross_exact = detect_exact_duplicates(valid_records)
    print(f"    -> Found {len(exact_groups)} exact duplicate groups ({len(cross_exact)} crossing splits).")

    # 4. Near duplicates
    print("[*] Computing perceptual dHash (64-bit) & clustering near-duplicates (Hamming <= 4)...")
    near_clusters, cross_near = detect_near_duplicates(valid_records, max_hamming_dist=4)
    print(f"    -> Found {len(near_clusters)} near-duplicate clusters ({len(cross_near)} crossing splits).")

    # 5. Determine status
    leakage_status = determine_leakage_status(len(cross_exact), len(cross_near))
    print(f"[*] Final Leakage Evaluation Status: {leakage_status}")

    # 6. Generate reports
    print(f"[*] Generating audit reports in {output_dir}...")
    generate_reports(
        dataset_root=dataset_root,
        output_dir=output_dir,
        valid_records=valid_records,
        corrupt_records=corrupt_records,
        exact_groups=exact_groups,
        cross_exact=cross_exact,
        near_clusters=near_clusters,
        cross_near=cross_near,
        stats=stats,
        leakage_status=leakage_status,
    )
    print("[+] Phase 1 audit completed successfully.")
    
    return {
        "status": leakage_status,
        "valid_count": len(valid_records),
        "corrupt_count": len(corrupt_records),
        "exact_groups": len(exact_groups),
        "cross_exact": len(cross_exact),
        "near_clusters": len(near_clusters),
        "cross_near": len(cross_near),
    }


def main():
    parser = argparse.ArgumentParser(
        description="ConViX-IR Phase 1: Read-only Dataset Verification and Leakage Audit"
    )
    parser.add_argument(
        "dataset_root",
        nargs="?",
        default="/content/drive/MyDrive/conVix-IR/dataset",
        help="Path to the dataset directory (default: /content/drive/MyDrive/conVix-IR/dataset)",
    )
    parser.add_argument(
        "--output_dir",
        "-o",
        default="phase1_results",
        help="Directory to save audit output files (default: phase1_results)",
    )
    args = parser.parse_args()

    run_audit(dataset_root_path=args.dataset_root, output_dir_path=args.output_dir)


if __name__ == "__main__":
    main()
