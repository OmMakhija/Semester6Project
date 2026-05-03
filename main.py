"""
main.py — Entry point: orchestrates dataset setup, training, and checkpointing

Anti-overfitting/underfitting measures:
  - LR warmup (5 epochs) to avoid instability at the start
  - ReduceLROnPlateau: drops LR when val loss stalls (adaptive)
  - Early stopping (patience=15): halts when no improvement
  - Higher weight decay (1e-4) for regularization
  - Nesterov momentum for faster, smoother convergence
  - MPS support for Apple Silicon (M1/M2/M3/M4)
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


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


def run(args):
    device = get_device()
    print(f"\n🔧  Device      : {device}")
    print(f"📁  Data        : {args.data_dir}")
    print(f"🔁  Max Epochs  : {args.epochs}")
    print(f"📦  Batch size  : {args.batch_size}")
    print(f"📉  Base LR     : {args.lr}")
    print(f"⏸️   Early stop  : patience={args.patience}\n")

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
        pin_memory=False, drop_last=True,
    )
    valid_loader = DataLoader(
        valid_ds, batch_size=args.batch_size,
        shuffle=False, num_workers=2, collate_fn=collate_fn,
        pin_memory=False,
    )

    print(f"📊  Train: {len(train_ds)} images | Valid: {len(valid_ds)} images")

    # ── 2. Model ─────────────────────────────────────────────────────────
    model = build_model(num_classes=2, pretrained=False).to(device)
    print("🧠  SSDLite320-MobileNetV3 ready\n")

    # ── 3. Optimizer ─────────────────────────────────────────────────────
    optimizer = optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=0.9,
        weight_decay=1e-4,
        nesterov=True,
    )

    # ── 4. LR Warmup ─────────────────────────────────────────────────────
    warmup_epochs = 5
    def warmup_lambda(epoch):
        if epoch < warmup_epochs:
            return 0.1 + 0.9 * (epoch / warmup_epochs)
        return 1.0

    warmup_scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warmup_lambda)

    # ── 5. Plateau Scheduler ─────────────────────────────────────────────
    plateau_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=5,
        min_lr=1e-6,
    )

    # ── 6. Training Loop ─────────────────────────────────────────────────
    best_val_loss     = float("inf")
    epochs_no_improve = 0
    history           = {"train_loss": [], "val_loss": [], "lr": []}

    print(f"🚀  Training started...\n")
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        current_lr = optimizer.param_groups[0]["lr"]
        train_loss = train_one_epoch(model, optimizer, train_loader, device, epoch)
        val_loss   = evaluate(model, valid_loader, device)

        if epoch <= warmup_epochs:
            warmup_scheduler.step()
        else:
            plateau_scheduler.step(val_loss)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["lr"].append(current_lr)

        gap     = val_loss - train_loss
        elapsed = time.time() - t0
        print(f"  ⏱  {elapsed:.1f}s | LR: {current_lr:.2e} | Gap: {gap:.3f}")

        if gap > 3.0:
            print(f"  ⚠️  Large train/val gap ({gap:.2f}) — possible overfitting")

        if val_loss < best_val_loss:
            best_val_loss     = val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), "best_model.pth")
            print(f"  ✅  best_model.pth saved (val_loss={best_val_loss:.4f})")
        else:
            epochs_no_improve += 1
            print(f"  ⏳  No improvement for {epochs_no_improve}/{args.patience} epochs")

        print()

        if epochs_no_improve >= args.patience:
            print(f"🛑  Early stopping at epoch {epoch} — no improvement for {args.patience} epochs")
            break

    # ── 7. Final Saves ───────────────────────────────────────────────────
    torch.save(model.state_dict(), "last_model.pth")
    with open("training_history.json", "w") as f:
        json.dump(history, f, indent=2)

    print(f"\n🎉  Done! Best val loss: {best_val_loss:.4f}")
    print("📁  Outputs: best_model.pth | last_model.pth | training_history.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Contaminant Detector")
    parser.add_argument("--data_dir",    type=str,   default="dataset")
    parser.add_argument("--epochs",      type=int,   default=200,  help="Max epochs (early stopping may end sooner)")
    parser.add_argument("--batch_size",  type=int,   default=32,   help="32 for Apple Silicon MPS, 8 for CPU")
    parser.add_argument("--lr",          type=float, default=0.005, help="Base learning rate")
    parser.add_argument("--patience",    type=int,   default=15,   help="Early stopping patience")
    parser.add_argument("--extract_zip", type=str,   default=None)
    args = parser.parse_args()

    if args.extract_zip:
        extract_dataset(args.extract_zip, args.data_dir)

    run(args)