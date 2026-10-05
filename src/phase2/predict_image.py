"""
ConViX-IR: Phase 2 — Single Image Inference Interface
======================================================
Context-Verified Iterative Re-Inspection CNN–ViT Framework for
Calibrated Peach Freshness Assessment.

Performs inference on a single peach image using the trained Phase 2
ConvNeXt-Tiny CNN baseline model.

Flow:
    Input Image
        ↓
    RGB Preprocessing (224x224, ImageNet Normalization)
        ↓
    Trained ConvNeXt-Tiny Backbone
        ↓
    768-D Feature Vector (E_C)
        ↓
    Linear Classifier
        ↓
    Logits (z_C)
        ↓
    Softmax Probabilities (p_C)
        ↓
    Predicted Class: 0 (Fresh_Peach) or 1 (Rotten_Peach)
"""

import sys
import argparse
from pathlib import Path
from typing import Dict, Any, Optional, Union

import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image

try:
    from .convnext_baseline import (
        ConvNeXtTinyBaseline,
        CLASS_MAP,
        ID_TO_CLASS,
        IMAGENET_MEAN,
        IMAGENET_STD,
    )
except ImportError:
    from convnext_baseline import (
        ConvNeXtTinyBaseline,
        CLASS_MAP,
        ID_TO_CLASS,
        IMAGENET_MEAN,
        IMAGENET_STD,
    )


def get_inference_transform() -> transforms.Compose:
    """
    Exact deterministic preprocessing used during validation and test evaluation.
    """
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def load_model(
    checkpoint_path: Union[str, Path],
    device: torch.device,
) -> ConvNeXtTinyBaseline:
    """
    Load the trained ConvNeXt-Tiny model weights from checkpoint.
    """
    ckpt_path = Path(checkpoint_path).resolve()
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {ckpt_path}")

    # Initialize model structure without downloading pretrained weights
    model = ConvNeXtTinyBaseline(pretrained=False, num_classes=2)
    checkpoint = torch.load(ckpt_path, map_location=device)

    # Handle checkpoint dictionary format or raw state dict
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    elif isinstance(checkpoint, dict):
        model.load_state_dict(checkpoint)
    else:
        raise ValueError(f"Unrecognized checkpoint format in: {ckpt_path}")

    model.to(device)
    model.eval()
    return model


def predict_single_image(
    model: ConvNeXtTinyBaseline,
    image_path: Union[str, Path],
    device: torch.device,
    transform: Optional[transforms.Compose] = None,
) -> Dict[str, Any]:
    """
    Run inference on a single image and return class probabilities and prediction.
    """
    img_path = Path(image_path).resolve()
    if not img_path.exists():
        raise FileNotFoundError(f"Image file not found: {img_path}")

    if transform is None:
        transform = get_inference_transform()

    with Image.open(img_path) as img:
        img_rgb = img.convert("RGB")
        tensor = transform(img_rgb)

    # Add batch dimension: [1, 3, 224, 224]
    input_batch = tensor.unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        features = model.extract_features(input_batch)  # [1, 768]
        logits = model.classifier(features)             # [1, 2]
        probabilities = F.softmax(logits, dim=-1)[0]    # [2]
        predicted_idx = int(torch.argmax(probabilities).item())

    fresh_prob = float(probabilities[0].item())
    rotten_prob = float(probabilities[1].item())

    return {
        "image_path": str(img_path),
        "predicted_class_id": predicted_idx,
        "predicted_class_name": ID_TO_CLASS[predicted_idx],
        "fresh_probability": fresh_prob,
        "rotten_probability": rotten_prob,
        "logits": [float(logits[0][0].item()), float(logits[0][1].item())],
        "feature_dim": int(features.shape[1]),
    }


def format_prediction_output(result: Dict[str, Any]) -> str:
    """Format inference results in a clean, human-readable layout."""
    lines = [
        "========================================",
        "ConViX-IR Phase 2 CNN Inference",
        "========================================",
        f"Image: {result['image_path']}",
        "",
        f"Prediction: {result['predicted_class_name']}",
        f"Predicted class: {result['predicted_class_id']}",
        "",
        f"Fresh probability:  {result['fresh_probability'] * 100:.4f}%",
        f"Rotten probability: {result['rotten_probability'] * 100:.4f}%",
        "========================================",
    ]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="ConViX-IR Phase 2: Single-Image Peach Freshness Inference"
    )
    parser.add_argument(
        "image_path",
        type=str,
        help="Path to the input peach image (.jpg, .png, etc.)",
    )
    parser.add_argument(
        "--checkpoint",
        "-c",
        type=str,
        default="phase2_results/best_convnext_tiny.pt",
        help="Path to trained ConvNeXt-Tiny checkpoint (default: phase2_results/best_convnext_tiny.pt)",
    )
    parser.add_argument(
        "--device",
        "-d",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Compute device: 'cuda' or 'cpu' (default: auto-detect)",
    )

    args = parser.parse_args()
    device = torch.device(args.device)

    try:
        model = load_model(args.checkpoint, device=device)
        result = predict_single_image(model, args.image_path, device=device)
        print(format_prediction_output(result))
    except Exception as exc:
        print(f"[ERROR] Inference failed: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
