"""
inference.py — Run detection on a single image and save annotated output
"""

import torch
from PIL import Image, ImageDraw
from torchvision import transforms as T

from model import load_model


@torch.no_grad()
def predict(image_path, weights_path, conf_thresh=0.5, num_classes=2, output_path="inference_result.jpg"):
    """
    Detect microplastics in a single image and save an annotated result.

    Args:
        image_path   (str)  : Path to the input image.
        weights_path (str)  : Path to trained .pth weights file.
        conf_thresh  (float): Minimum confidence score to display a box (0–1).
        num_classes  (int)  : Must match the value used during training.
        output_path  (str)  : Where to save the annotated result image.

    Returns:
        dict: {"count": int, "boxes": list, "scores": list}
    """
    model, device = load_model(weights_path, num_classes=num_classes)
    print(f"✅  Loaded weights: {weights_path}")

    # Load and preprocess image
    img_orig       = Image.open(image_path).convert("RGB")
    orig_w, orig_h = img_orig.size

    transform  = T.Compose([T.Resize((320, 320)), T.ToTensor()])
    img_tensor = transform(img_orig).unsqueeze(0).to(device)

    # Forward pass
    predictions = model(img_tensor)[0]
    boxes  = predictions["boxes"].cpu().numpy()
    scores = predictions["scores"].cpu().numpy()
    labels = predictions["labels"].cpu().numpy()

    # Rescale boxes to original image resolution
    sx = orig_w / 320
    sy = orig_h / 320
    boxes[:, [0, 2]] *= sx
    boxes[:, [1, 3]] *= sy

    # Apply confidence threshold
    keep = scores >= conf_thresh

    # Draw results
    draw = ImageDraw.Draw(img_orig)
    results_boxes  = []
    results_scores = []

    for box, score, label in zip(boxes[keep], scores[keep], labels[keep]):
        x1, y1, x2, y2 = box
        draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
        draw.text((x1, max(0, y1 - 12)), f"Plastic {score:.2f}", fill="red")
        results_boxes.append([float(x1), float(y1), float(x2), float(y2)])
        results_scores.append(float(score))

    count = len(results_boxes)
    img_orig.save(output_path)

    print(f"\n🔍  Detected {count} microplastic particle(s) (threshold={conf_thresh})")
    print(f"💾  Annotated image saved → {output_path}")

    return {"count": count, "boxes": results_boxes, "scores": results_scores}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run microplastics inference on a single image")
    parser.add_argument("--image",       required=True,           help="Path to input image")
    parser.add_argument("--weights",     default="best_model.pth",help="Path to trained weights (.pth)")
    parser.add_argument("--conf_thresh", type=float, default=0.5, help="Confidence threshold (0–1)")
    parser.add_argument("--output",      default="inference_result.jpg", help="Output image path")
    args = parser.parse_args()

    result = predict(
        image_path   = args.image,
        weights_path = args.weights,
        conf_thresh  = args.conf_thresh,
        output_path  = args.output,
    )
    print(f"\nScores: {[round(s, 3) for s in result['scores']]}")
