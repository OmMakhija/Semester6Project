"""
main.py — Entry point: orchestrates dataset setup, training, and checkpointing
"""

import os
import time
import json
import argparse

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from dataset  import MicroplasticsDataset, get_transforms, collate_fn
from model    import build_model
from train    import train_one_epoch
from evaluate import evaluate
from utils    import extract_dataset


def run(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n🔧  Device : {device}")
    print(f"📁  Data   : {args.data_dir}")
    print(f"🔁  Epochs : {args.epochs}  |  Batch: {args.batch_size}  |  LR: {args.lr}\n")

    # ── 1. Datasets ──────────────────────────────────────────────────────
    train_ds = MicroplasticsDataset(
        img_dir    = os.path.join(args.data_dir, "train"),
        csv_path   = os.path.join(args.data_dir, "train", "_annotations.csv"),
        transforms = get_transforms(train=True),
    )
    valid_ds = MicroplasticsDataset(
        img_dir    = os.path.join(args.data_dir, "valid"),
        csv_path   = os.path.join(args.data_dir, "valid", "_annotations.csv"),
        transforms = get_transforms(train=False),
    )

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size,
        shuffle=True,  num_workers=2, collate_fn=collate_fn,
        pin_memory=(device.type == "cuda"), drop_last=True,  # drop_last avoids single-sample batches that break BatchNorm
    )
    valid_loader = DataLoader(
        valid_ds, batch_size=args.batch_size,
        shuffle=False, num_workers=2, collate_fn=collate_fn,
        pin_memory=(device.type == "cuda"),
    )

    print(f"📊  Train: {len(train_ds)} images | Valid: {len(valid_ds)} images")

    # ── 2. Model ─────────────────────────────────────────────────────────
    model = build_model(num_classes=2, pretrained=False).to(device)
    print("🧠  SSDLite320-MobileNetV3 loaded (pretrained on COCO)\n")

    # ── 3. Optimizer + Scheduler ─────────────────────────────────────────
    optimizer = optim.SGD(
        model.parameters(),
        lr=args.lr, momentum=0.9, weight_decay=4e-5,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # ── 4. Training Loop ─────────────────────────────────────────────────
    best_val_loss = float("inf")
    history       = {"train_loss": [], "val_loss": []}

    print(f"🚀  Training started...\n")
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        train_loss = train_one_epoch(model, optimizer, train_loader, device, epoch)
        val_loss   = evaluate(model, valid_loader, device)
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        elapsed = time.time() - t0
        print(f"  ⏱  {elapsed:.1f}s | LR: {scheduler.get_last_lr()[0]:.2e}")

        # Save best checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "best_model.pth")
            print(f"  ✅  best_model.pth saved (val_loss={best_val_loss:.4f})")
        print()

    # ── 5. Final Saves ───────────────────────────────────────────────────
    torch.save(model.state_dict(), "last_model.pth")
    with open("training_history.json", "w") as f:
        json.dump(history, f, indent=2)

    print(f"🎉  Done! Best val loss: {best_val_loss:.4f}")
    print("📁  Outputs: best_model.pth | last_model.pth | training_history.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Microplastics Detector")
    parser.add_argument("--data_dir",    type=str,   default="dataset", help="Root dataset folder")
    parser.add_argument("--epochs",      type=int,   default=30)
    parser.add_argument("--batch_size",  type=int,   default=8)
    parser.add_argument("--lr",          type=float, default=0.01)
    parser.add_argument("--extract_zip", type=str,   default=None,      help="Extract dataset zip first")
    args = parser.parse_args()

    if args.extract_zip:
        extract_dataset(args.extract_zip, args.data_dir)

    run(args)