"""
utils.py — Helper utilities: dataset extraction and training curve plotting
"""

import os
import json
import zipfile


def extract_dataset(zip_path, output_dir="dataset"):
    """
    Extract a Roboflow-style zip archive to a local folder.

    Expected zip structure:
        train/_annotations.csv
        train/*.jpg
        valid/_annotations.csv
        valid/*.jpg

    Args:
        zip_path   (str): Path to the .zip file.
        output_dir (str): Destination folder (created if it doesn't exist).
    """
    if os.path.exists(output_dir):
        print(f"⚠️  '{output_dir}' already exists — skipping extraction.")
        return
    print(f"📦  Extracting {zip_path} → {output_dir} ...")
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(output_dir)
    print("✅  Done.")


def plot_losses(history_path="training_history.json", output_path="loss_curve.png"):
    """
    Plot training and validation loss curves from a saved JSON history file.
    Saves a PNG image and optionally displays it.

    Args:
        history_path (str): Path to training_history.json produced by main.py.
        output_path  (str): Where to save the PNG plot.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("⚠️  matplotlib not installed. Run: pip install matplotlib")
        return

    with open(history_path, "r") as f:
        history = json.load(f)

    epochs      = range(1, len(history["train_loss"]) + 1)
    train_loss  = history["train_loss"]
    val_loss    = history["val_loss"]

    plt.figure(figsize=(9, 5))
    plt.plot(epochs, train_loss, label="Train Loss", linewidth=2, color="#2196F3")
    plt.plot(epochs, val_loss,   label="Val Loss",   linewidth=2, color="#F44336")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Microplastics Detector — Training History")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"📈  Loss curve saved → {output_path}")
    plt.show()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Utilities: extract dataset or plot loss curve")
    parser.add_argument("--extract_zip", type=str, default=None,  help="Zip file to extract")
    parser.add_argument("--data_dir",    type=str, default="dataset")
    parser.add_argument("--plot",        action="store_true",      help="Plot training_history.json")
    parser.add_argument("--history",     type=str, default="training_history.json")
    args = parser.parse_args()

    if args.extract_zip:
        extract_dataset(args.extract_zip, args.data_dir)

    if args.plot:
        plot_losses(args.history)
