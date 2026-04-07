"""
export.py — Export trained model to ONNX for mobile / edge deployment
"""

import torch
from model import load_model


def export_onnx(weights_path, output_path="microplastics.onnx", num_classes=2):
    """
    Convert a trained .pth model to ONNX format.

    The exported model accepts a single 320×320 RGB image tensor
    and outputs raw SSD predictions (before NMS post-processing).
    For full post-processing on device, apply NMS with your target framework.

    Args:
        weights_path (str): Path to trained .pth weights.
        output_path  (str): Where to save the .onnx file.
        num_classes  (int): Must match training configuration.
    """
    model, device = load_model(weights_path, num_classes=num_classes)
    model.eval()

    dummy_input = torch.zeros(1, 3, 320, 320, device=device)

    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        opset_version  = 11,
        input_names    = ["image"],
        output_names   = ["boxes", "labels", "scores"],
        dynamic_axes   = {"image": {0: "batch_size"}},
    )

    print(f"✅  Exported ONNX model → {output_path}")
    print("    Input  : (1, 3, 320, 320)  float32")
    print("    Outputs: boxes, labels, scores")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Export microplastics model to ONNX")
    parser.add_argument("--weights", default="best_model.pth", help="Path to trained .pth file")
    parser.add_argument("--output",  default="microplastics.onnx", help="Output .onnx path")
    args = parser.parse_args()

    export_onnx(args.weights, args.output)
