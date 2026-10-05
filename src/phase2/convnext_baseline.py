"""
ConViX-IR: Phase 2 — ConvNeXt-Tiny Baseline
============================================
Context-Verified Iterative Re-Inspection CNN–ViT Framework for
Calibrated Peach Freshness Assessment.

Core Architecture:
    x in R^(3 x 224 x 224)
        ↓
    ConvNeXt-Tiny Backbone
        ↓
    Global Average Pooling (GAP)
        ↓
    E_C in R^768 (768-D Feature Representation)
        ↓
    Linear Classifier (W_C in R^(2 x 768), b_C in R^2)
        ↓
    z_C in R^2 (Class Logits)
        ↓
    p_C = softmax(z_C) in R^2 (Class Probabilities)
        ↓
    y_hat = argmax(p_C)

Classes:
    0 = Fresh_Peach
    1 = Rotten_Peach (Positive Class)
"""

import os
import sys
import argparse
import random
import json
import csv
import math
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

try:
    import torchvision.models as models
    from torchvision.models import ConvNeXt_Tiny_Weights
    TORCHVISION_AVAILABLE = True
except ImportError:
    TORCHVISION_AVAILABLE = False

try:
    import timm
    TIMM_AVAILABLE = True
except ImportError:
    TIMM_AVAILABLE = False


# Class and Split Constants
CLASS_MAP: Dict[str, int] = {
    "Fresh_Peach": 0,
    "Rotten_Peach": 1,
}
ID_TO_CLASS: Dict[int, str] = {v: k for k, v in CLASS_MAP.items()}
SPLITS: List[str] = ["train", "val", "test"]

# ImageNet normalization statistics
IMAGENET_MEAN: List[float] = [0.485, 0.456, 0.406]
IMAGENET_STD: List[float] = [0.229, 0.224, 0.225]
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def set_seed(seed: int = 42) -> None:
    """Set random seeds for deterministic reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_transforms() -> Dict[str, transforms.Compose]:
    """
    Build data transformation pipelines.
    Train: Data augmentation (RandomResizedCrop, RandomHorizontalFlip, Rotation, ColorJitter)
    Val/Test: Deterministic preprocessing (Resize, CenterCrop, Normalization)
    """
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    val_test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    return {
        "train": train_transform,
        "val": val_test_transform,
        "test": val_test_transform,
    }


class PeachDataset(Dataset):
    """
    Read-only Dataset loader for ConViX-IR peach freshness data.
    Preserves strict existing train / val / test folder splits.
    """

    def __init__(self, dataset_root: Union[str, Path], split: str, transform: Optional[transforms.Compose] = None):
        self.dataset_root = Path(dataset_root).resolve()
        self.split = split
        self.transform = transform
        self.samples: List[Tuple[Path, int, str]] = []

        split_dir = self.dataset_root / split
        if not split_dir.exists():
            raise FileNotFoundError(f"Split directory does not exist: {split_dir}")

        for class_name, label_id in sorted(CLASS_MAP.items()):
            class_dir = split_dir / class_name
            if not class_dir.exists():
                continue
            
            image_files = sorted([
                f for f in class_dir.rglob("*")
                if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS
            ])
            for img_path in image_files:
                self.samples.append((img_path, label_id, class_name))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        img_path, label, class_name = self.samples[idx]
        with Image.open(img_path) as img:
            img = img.convert("RGB")
            if self.transform is not None:
                tensor = self.transform(img)
            else:
                tensor = transforms.ToTensor()(img)

        return {
            "image": tensor,
            "label": torch.tensor(label, dtype=torch.long),
            "path": str(img_path),
            "class_name": class_name,
            "split": self.split,
        }


class ConvNeXtTinyBaseline(nn.Module):
    """
    ConvNeXt-Tiny Baseline Architecture for ConViX-IR Phase 2.
    
    Mathematical Formulation:
        E_C = GAP(F_C(x)) in R^768
        z_C = W_C E_C + b_C in R^2
        p_C = softmax(z_C)
    """

    def __init__(self, pretrained: bool = True, num_classes: int = 2, drop_rate: float = 0.0):
        super().__init__()
        self.num_classes = num_classes
        self.feature_dim = 768

        # Load ConvNeXt-Tiny backbone
        if TIMM_AVAILABLE:
            self.backbone = timm.create_model(
                "convnext_tiny",
                pretrained=pretrained,
                num_classes=0,  # return pooled 768-D representation
                drop_rate=drop_rate,
            )
        elif TORCHVISION_AVAILABLE:
            weights = ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
            tv_model = models.convnext_tiny(weights=weights)
            # Remove existing classifier head, retain features and avgpool
            self.backbone = nn.Sequential(
                tv_model.features,
                tv_model.avgpool,
                nn.Flatten(1),
            )
        else:
            raise ImportError("Neither 'timm' nor 'torchvision' is available for ConvNeXt-Tiny.")

        # Standard linear classifier: W_C in R^(2 x 768), b_C in R^2
        self.classifier = nn.Linear(self.feature_dim, num_classes)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract 768-dimensional global representation E_C.
        Input:  [B, 3, 224, 224]
        Output: [B, 768]
        """
        features = self.backbone(x)
        if features.dim() > 2:
            features = torch.flatten(features, 1)
        return features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass producing class logits z_C.
        Input:  [B, 3, 224, 224]
        Output: [B, 2]
        """
        features = self.extract_features(x)
        logits = self.classifier(features)
        return logits

    def predict_probabilities(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute class probabilities p_C = softmax(z_C).
        Input:  [B, 3, 224, 224]
        Output: [B, 2]
        """
        logits = self.forward(x)
        return F.softmax(logits, dim=-1)


def compute_ece(probs: np.ndarray, labels: np.ndarray, num_bins: int = 10) -> float:
    """
    Calculate Expected Calibration Error (ECE) for binary Rotten_Peach classification.
    """
    bin_boundaries = np.linspace(0, 1, num_bins + 1)
    ece = 0.0
    total_samples = len(labels)

    # For binary prediction, use the predicted class confidence and binary accuracy
    pred_labels = (probs >= 0.5).astype(int)
    confidences = np.where(pred_labels == 1, probs, 1.0 - probs)
    accuracies = (pred_labels == labels).astype(float)

    for i in range(num_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper) if i > 0 else (confidences >= bin_lower) & (confidences <= bin_upper)
        bin_size = np.sum(in_bin)

        if bin_size > 0:
            bin_acc = np.mean(accuracies[in_bin])
            bin_conf = np.mean(confidences[in_bin])
            ece += (bin_size / total_samples) * abs(bin_acc - bin_conf)

    return float(ece)


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob_rotten: np.ndarray) -> Dict[str, Any]:
    """
    Compute full suite of Phase 2 evaluation metrics:
    Accuracy, Precision, Recall, F1, Specificity, ROC-AUC, Brier Score, ECE, Confusion Matrix.
    """
    # Confusion matrix components
    # Fresh_Peach = 0, Rotten_Peach = 1
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))

    total = tp + tn + fp + fn
    accuracy = (tp + tn) / total if total > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    # ROC-AUC calculation
    try:
        from sklearn.metrics import roc_auc_score
        roc_auc = float(roc_auc_score(y_true, y_prob_rotten))
    except Exception:
        # Fallback manual trapezoidal approximation if scikit-learn is unavailable
        roc_auc = 0.0

    # Brier Score: (1/N) * sum((p_i - y_i)^2)
    brier_score = float(np.mean((y_prob_rotten - y_true) ** 2))

    # Expected Calibration Error
    ece = compute_ece(y_prob_rotten, y_true, num_bins=10)

    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "specificity": float(specificity),
        "roc_auc": float(roc_auc),
        "brier_score": float(brier_score),
        "ece": float(ece),
        "confusion_matrix": {
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "tp": tp,
        },
    }


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    scaler: Optional[torch.amp.GradScaler] = None,
) -> Tuple[float, float]:
    """Train for one epoch across D_train."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    use_cuda = device.type == "cuda"

    for batch in dataloader:
        images = batch["image"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()

        if use_cuda and scaler is not None:
            with torch.amp.autocast(device_type="cuda"):
                logits = model(images)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

        running_loss += loss.item() * images.size(0)
        preds = torch.argmax(logits, dim=1)
        correct += (preds == labels).sum().item()
        total += images.size(0)

    epoch_loss = running_loss / total if total > 0 else 0.0
    epoch_acc = correct / total if total > 0 else 0.0
    return epoch_loss, epoch_acc


def evaluate_dataset(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float, List[Dict[str, Any]]]:
    """
    Evaluate model on validation or test dataset.
    Returns: (loss, accuracy, item_predictions)
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    predictions: List[Dict[str, Any]] = []

    with torch.no_grad():
        for batch in dataloader:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)
            paths = batch["path"]

            logits = model(images)
            loss = criterion(logits, labels)

            running_loss += loss.item() * images.size(0)
            probs = F.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)

            correct += (preds == labels).sum().item()
            total += images.size(0)

            for i in range(images.size(0)):
                predictions.append({
                    "image_path": paths[i],
                    "true_label": int(labels[i].item()),
                    "predicted_label": int(preds[i].item()),
                    "fresh_probability": float(probs[i][0].item()),
                    "rotten_probability": float(probs[i][1].item()),
                    "logit_fresh": float(logits[i][0].item()),
                    "logit_rotten": float(logits[i][1].item()),
                })

    eval_loss = running_loss / total if total > 0 else 0.0
    eval_acc = correct / total if total > 0 else 0.0
    return eval_loss, eval_acc, predictions


def export_features(
    model: nn.Module,
    dataloaders: Dict[str, DataLoader],
    output_dir: Path,
    device: torch.device,
) -> None:
    """
    Extract and save 768-D ConvNeXt feature representations E_C for all splits.
    """
    features_dir = output_dir / "features"
    features_dir.mkdir(parents=True, exist_ok=True)
    model.eval()

    print(f"[*] Exporting 768-D ConvNeXt-Tiny features E_C to {features_dir}...")

    with torch.no_grad():
        for split, loader in dataloaders.items():
            all_features: List[np.ndarray] = []
            all_paths: List[str] = []
            all_labels: List[int] = []

            for batch in loader:
                images = batch["image"].to(device)
                feats = model.extract_features(images)
                all_features.append(feats.cpu().numpy())
                all_paths.extend(batch["path"])
                all_labels.extend(batch["label"].tolist())

            if all_features:
                cat_feats = np.concatenate(all_features, axis=0)  # [N, 768]
                np.save(features_dir / f"{split}_features_768d.npy", cat_feats)
                with open(features_dir / f"{split}_feature_meta.json", "w", encoding="utf-8") as f:
                    json.dump({"paths": all_paths, "labels": all_labels, "feature_shape": list(cat_feats.shape)}, f, indent=2)
                print(f"    -> Split '{split}': Extracted features shape {cat_feats.shape}")


def run_experiment(
    dataset_root: str,
    output_dir: str = "phase2_results",
    epochs: int = 20,
    batch_size: int = 16,
    lr: float = 1e-4,
    weight_decay: float = 1e-2,
    seed: int = 42,
    save_features: bool = False,
    device_name: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Run complete Phase 2 training and evaluation pipeline.
    """
    set_seed(seed)
    output_path = Path(output_dir).resolve()
    output_path.mkdir(parents=True, exist_ok=True)

    if device_name:
        device = torch.device(device_name)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("================================================================================")
    print("ConViX-IR: PHASE 2 ConvNeXt-Tiny BASELINE EXPERIMENT")
    print("================================================================================")
    print(f"Dataset Root: {dataset_root}")
    print(f"Output Directory: {output_path}")
    print(f"Compute Device: {device}")
    print(f"Hyperparameters: Epochs={epochs}, BatchSize={batch_size}, LR={lr}, WeightDecay={weight_decay}, Seed={seed}")

    # 1. Transforms and Data Loaders
    transforms_dict = get_transforms()
    datasets = {
        split: PeachDataset(dataset_root=dataset_root, split=split, transform=transforms_dict[split])
        for split in SPLITS
    }

    print(f"[*] Dataset Counts: Train={len(datasets['train'])}, Val={len(datasets['val'])}, Test={len(datasets['test'])}")

    dataloaders = {
        split: DataLoader(
            datasets[split],
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=0,
            pin_memory=(device.type == "cuda"),
        )
        for split in SPLITS
    }

    # 2. Model, Optimizer, Scheduler, Loss
    print("[*] Initializing ConvNeXt-Tiny baseline...")
    model = ConvNeXtTinyBaseline(pretrained=True, num_classes=2).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None

    # 3. Training Loop with Best-Validation-Loss Checkpointing
    best_val_loss = float("inf")
    best_checkpoint_path = output_path / "best_convnext_tiny.pt"
    history: List[Dict[str, Any]] = []

    print("[*] Starting model training across epochs...")
    for epoch in range(1, epochs + 1):
        current_lr = scheduler.get_last_lr()[0]
        train_loss, train_acc = train_one_epoch(
            model=model,
            dataloader=dataloaders["train"],
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            scaler=scaler,
        )

        val_loss, val_acc, _ = evaluate_dataset(
            model=model,
            dataloader=dataloaders["val"],
            criterion=criterion,
            device=device,
        )

        scheduler.step()

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_acc,
            "val_loss": val_loss,
            "val_accuracy": val_acc,
            "learning_rate": current_lr,
        })

        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_loss": val_loss,
                "val_accuracy": val_acc,
                "seed": seed,
            }, best_checkpoint_path)

        print(f"Epoch [{epoch:02d}/{epochs:02d}] "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}% | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc*100:.2f}% | "
              f"LR: {current_lr:.2e} {'[BEST VAL CHECKPOINT SAVED]' if is_best else ''}")

    # 4. Save Training History CSV
    history_csv_path = output_path / "training_history.csv"
    with open(history_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "train_accuracy", "val_loss", "val_accuracy", "learning_rate"])
        writer.writeheader()
        writer.writerows(history)

    # 5. Load Best Checkpoint for Final Test Evaluation
    print(f"[*] Loading best validation checkpoint from {best_checkpoint_path} for final test evaluation...")
    checkpoint = torch.load(best_checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    test_loss, test_acc, test_predictions = evaluate_dataset(
        model=model,
        dataloader=dataloaders["test"],
        criterion=criterion,
        device=device,
    )

    # 6. Compute Comprehensive Test Metrics
    y_true = np.array([p["true_label"] for p in test_predictions])
    y_pred = np.array([p["predicted_label"] for p in test_predictions])
    y_prob_rotten = np.array([p["rotten_probability"] for p in test_predictions])

    metrics = calculate_metrics(y_true, y_pred, y_prob_rotten)
    metrics["test_loss"] = float(test_loss)
    metrics["best_epoch"] = int(checkpoint["epoch"])
    metrics["best_val_loss"] = float(checkpoint["val_loss"])

    # 7. Write Output Files
    # a. test_predictions.csv
    test_pred_path = output_path / "test_predictions.csv"
    with open(test_pred_path, "w", newline="", encoding="utf-8") as f:
        fields = ["image_path", "true_label", "predicted_label", "fresh_probability", "rotten_probability", "logit_fresh", "logit_rotten"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(test_predictions)

    # b. test_metrics.json
    metrics_json_path = output_path / "test_metrics.json"
    with open(metrics_json_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    # c. confusion_matrix.csv
    cm_path = output_path / "confusion_matrix.csv"
    cm = metrics["confusion_matrix"]
    with open(cm_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["", "Pred_Fresh_0", "Pred_Rotten_1"])
        writer.writerow(["True_Fresh_0", cm["tn"], cm["fp"]])
        writer.writerow(["True_Rotten_1", cm["fn"], cm["tp"]])

    # d. phase2_summary.txt
    summary_path = output_path / "phase2_summary.txt"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("================================================================================\n")
        f.write("ConViX-IR: PHASE 2 ConvNeXt-Tiny BASELINE SUMMARY\n")
        f.write("================================================================================\n\n")
        f.write(f"Model: ConvNeXt-Tiny (Pretrained)\n")
        f.write(f"Seed: {seed}\n")
        f.write(f"Best Epoch: {checkpoint['epoch']} (Best Val Loss: {checkpoint['val_loss']:.4f})\n\n")
        f.write("--- TEST EVALUATION METRICS ---\n")
        f.write(f"Accuracy:        {metrics['accuracy']*100:.2f}%\n")
        f.write(f"Precision:       {metrics['precision']*100:.2f}%\n")
        f.write(f"Recall:          {metrics['recall']*100:.2f}%\n")
        f.write(f"F1-Score:        {metrics['f1']*100:.2f}%\n")
        f.write(f"Specificity:     {metrics['specificity']*100:.2f}%\n")
        f.write(f"ROC-AUC:         {metrics['roc_auc']:.4f}\n")
        f.write(f"Brier Score:     {metrics['brier_score']:.4f}\n")
        f.write(f"ECE:             {metrics['ece']:.4f}\n\n")
        f.write("--- CONFUSION MATRIX ---\n")
        f.write(f"True Fresh (0):  TN={cm['tn']:<4} FP={cm['fp']:<4}\n")
        f.write(f"True Rotten (1): FN={cm['fn']:<4} TP={cm['tp']:<4}\n\n")
        f.write("--- RESEARCH INTEGRITY DISCLAIMER ---\n")
        f.write("Phase 2 is an independent ConvNeXt-Tiny CNN baseline. It does not implement\n")
        f.write("CNN-ViT fusion, context verification, disagreement measurement, dynamic fusion,\n")
        f.write("or iterative re-inspection.\n")
        f.write("================================================================================\n")

    # 8. Optional Feature Export
    if save_features:
        export_features(model, dataloaders, output_path, device)

    print("\n[+] Phase 2 ConvNeXt-Tiny baseline experiment completed successfully.")
    print(f"[+] Results saved to: {output_path}")
    return metrics


def main():
    parser = argparse.ArgumentParser(description="ConViX-IR Phase 2: ConvNeXt-Tiny Baseline Training and Evaluation")
    parser.add_argument("dataset_root", nargs="?", default="/content/drive/MyDrive/conVix-IR/dataset", help="Path to peach dataset root")
    parser.add_argument("--output_dir", "-o", default="phase2_results", help="Directory to save Phase 2 artifacts")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate for AdamW")
    parser.add_argument("--weight_decay", type=float, default=1e-2, help="Weight decay for AdamW")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--save_features", action="store_true", help="Extract and save 768-D ConvNeXt features E_C")

    args = parser.parse_args()

    run_experiment(
        dataset_root=args.dataset_root,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        seed=args.seed,
        save_features=args.save_features,
    )


if __name__ == "__main__":
    main()
