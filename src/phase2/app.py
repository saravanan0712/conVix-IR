"""
ConViX-IR: Phase 2 — Interactive Image-Upload UI Application
============================================================
Context-Verified Iterative Re-Inspection CNN–ViT Framework for
Calibrated Peach Freshness Assessment.

Provides a user-facing interactive image-upload interface (via Gradio)
for peach freshness classification using the trained Phase 2 ConvNeXt-Tiny
CNN baseline.

Pipeline:
    Uploaded Image (from PC / UI)
        ↓
    RGB Preprocessing (224x224, ImageNet Normalization)
        ↓
    Trained ConvNeXt-Tiny Backbone
        ↓
    768-D Feature Vector (E_C)
        ↓
    Linear Classifier (W_C in R^(2 x 768), b_C in R^2)
        ↓
    Logits (z_C)
        ↓
    Softmax Class Probabilities (p_C)
        ↓
    Prediction Display: FRESH or ROTTEN
"""

import os
import sys
import argparse
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, Union

import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np

# Ensure phase2 directory is on sys.path for robust imports
_phase2_dir = str(Path(__file__).resolve().parent)
if _phase2_dir not in sys.path:
    sys.path.insert(0, _phase2_dir)

try:
    from convnext_baseline import (
        ConvNeXtTinyBaseline,
        CLASS_MAP,
        ID_TO_CLASS,
        IMAGENET_MEAN,
        IMAGENET_STD,
    )
    from predict_image import get_inference_transform, load_model
except ImportError:
    from src.phase2.convnext_baseline import (
        ConvNeXtTinyBaseline,
        CLASS_MAP,
        ID_TO_CLASS,
        IMAGENET_MEAN,
        IMAGENET_STD,
    )
    from src.phase2.predict_image import get_inference_transform, load_model


# Global cached model instance to avoid re-loading on every click
_CACHED_MODEL: Optional[ConvNeXtTinyBaseline] = None
_CACHED_CKPT_PATH: Optional[str] = None


def get_model(
    checkpoint_path: Union[str, Path] = "phase2_results/best_convnext_tiny.pt",
    device: Optional[torch.device] = None,
) -> ConvNeXtTinyBaseline:
    """Get or load cached ConvNeXt-Tiny model in eval mode."""
    global _CACHED_MODEL, _CACHED_CKPT_PATH

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt_str = str(Path(checkpoint_path).resolve())
    if _CACHED_MODEL is None or _CACHED_CKPT_PATH != ckpt_str:
        _CACHED_MODEL = load_model(checkpoint_path, device=device)
        _CACHED_CKPT_PATH = ckpt_str

    return _CACHED_MODEL


def predict_freshness(
    image_input: Union[Image.Image, np.ndarray, str, Path],
    checkpoint_path: Union[str, Path] = "phase2_results/best_convnext_tiny.pt",
    model: Optional[ConvNeXtTinyBaseline] = None,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """
    Core inference function for uploaded image.
    
    Args:
        image_input: PIL Image, NumPy array, or path to image file.
        checkpoint_path: Path to trained checkpoint.
        model: Optional pre-loaded model (useful for unit testing).
        device: Target compute device.

    Returns:
        Dict with predicted class name, probabilities, and formatted strings.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if model is None:
        model = get_model(checkpoint_path=checkpoint_path, device=device)

    # Convert input to PIL Image in RGB mode
    if isinstance(image_input, (str, Path)):
        img_path = Path(image_input)
        if not img_path.exists():
            raise FileNotFoundError(f"Image not found: {img_path}")
        with Image.open(img_path) as img:
            pil_img = img.convert("RGB")
    elif isinstance(image_input, np.ndarray):
        pil_img = Image.fromarray(image_input).convert("RGB")
    elif isinstance(image_input, Image.Image):
        pil_img = image_input.convert("RGB")
    else:
        raise ValueError(f"Unsupported image input type: {type(image_input)}")

    # Apply deterministic validation/test preprocessing
    transform = get_inference_transform()
    tensor = transform(pil_img)  # [3, 224, 224]
    input_batch = tensor.unsqueeze(0).to(device)  # [1, 3, 224, 224]

    model.eval()
    with torch.no_grad():
        features = model.extract_features(input_batch)  # [1, 768]
        logits = model.classifier(features)             # [1, 2]
        probabilities = F.softmax(logits, dim=-1)[0]    # [2]
        predicted_idx = int(torch.argmax(probabilities).item())

    fresh_prob = float(probabilities[0].item())
    rotten_prob = float(probabilities[1].item())

    # Map to user-facing labels
    # 0 = Fresh_Peach -> "FRESH", 1 = Rotten_Peach -> "ROTTEN"
    display_prediction = "FRESH" if predicted_idx == 0 else "ROTTEN"
    class_name = ID_TO_CLASS[predicted_idx]

    return {
        "prediction_display": display_prediction,
        "class_name": class_name,
        "class_id": predicted_idx,
        "fresh_probability": fresh_prob,
        "rotten_probability": rotten_prob,
        "fresh_probability_str": f"{fresh_prob * 100:.2f}%",
        "rotten_probability_str": f"{rotten_prob * 100:.2f}%",
        "disclaimer": "Model-estimated class probabilities, NOT physical freshness percentages.",
    }


def predict_for_gradio(
    image: Optional[Image.Image],
    checkpoint_path: str = "phase2_results/best_convnext_tiny.pt",
) -> Tuple[str, str, str]:
    """Gradio callback wrapper returning UI string tuple."""
    if image is None:
        return "Please upload an image.", "-", "-"

    try:
        res = predict_freshness(image, checkpoint_path=checkpoint_path)
        return (
            res["prediction_display"],
            res["fresh_probability_str"],
            res["rotten_probability_str"],
        )
    except Exception as exc:
        return f"Error: {exc}", "-", "-"


def create_app(checkpoint_path: str = "phase2_results/best_convnext_tiny.pt") -> Any:
    """Construct the Gradio interactive UI."""
    try:
        import gradio as gr
    except ImportError:
        raise ImportError(
            "Gradio is required to run the interactive UI. "
            "Please install it with: pip install gradio"
        )

    title = "ConViX-IR Peach Freshness Detection"
    description = (
        "Upload a peach image to classify it as Fresh or Rotten "
        "using the trained ConvNeXt-Tiny Phase 2 baseline."
    )

    with gr.Blocks(title=title) as app:
        gr.Markdown(f"# 🍑 {title}")
        gr.Markdown(description)

        with gr.Row():
            with gr.Column():
                image_input = gr.Image(
                    type="pil",
                    label="Upload Peach Image",
                    sources=["upload", "clipboard"],
                )
                predict_button = gr.Button("Predict Freshness", variant="primary", size="lg")

            with gr.Column():
                prediction_output = gr.Textbox(
                    label="Prediction",
                    placeholder="Prediction will appear here...",
                    interactive=False,
                )
                fresh_prob_output = gr.Textbox(
                    label="Fresh Probability",
                    placeholder="Fresh probability...",
                    interactive=False,
                )
                rotten_prob_output = gr.Textbox(
                    label="Rotten Probability",
                    placeholder="Rotten probability...",
                    interactive=False,
                )
                gr.Markdown(
                    "> **Note:** The probabilities shown are model-estimated class probabilities, "
                    "NOT physical freshness percentages."
                )

        # Wire click and auto-change events
        fn_callback = lambda img: predict_for_gradio(img, checkpoint_path=checkpoint_path)
        predict_button.click(
            fn=fn_callback,
            inputs=[image_input],
            outputs=[prediction_output, fresh_prob_output, rotten_prob_output],
        )

    return app


def main():
    parser = argparse.ArgumentParser(
        description="ConViX-IR Phase 2: Interactive Peach Freshness Classification UI"
    )
    parser.add_argument(
        "--checkpoint",
        "-c",
        type=str,
        default="phase2_results/best_convnext_tiny.pt",
        help="Path to trained ConvNeXt-Tiny checkpoint (default: phase2_results/best_convnext_tiny.pt)",
    )
    parser.add_argument(
        "--port",
        "-p",
        type=int,
        default=7860,
        help="Port to run the UI server on (default: 7860)",
    )
    parser.add_argument(
        "--share",
        action="store_true",
        help="Create a publicly shareable Gradio link (recommended for Colab)",
    )

    args = parser.parse_args()

    try:
        app = create_app(checkpoint_path=args.checkpoint)
        print("================================================================================")
        print("ConViX-IR Phase 2 Interactive UI Starting")
        print(f"Checkpoint: {args.checkpoint}")
        print("================================================================================")
        app.launch(server_port=args.port, share=args.share)
    except ImportError as err:
        print(f"\n[!] {err}", file=sys.stderr)
        print("[!] In Google Colab or local terminal, install Gradio with: pip install gradio\n", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
