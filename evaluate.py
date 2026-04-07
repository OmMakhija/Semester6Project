"""
evaluate.py — Validation loss computation
"""

import torch


@torch.no_grad()
def evaluate(model, loader, device):
    """
    Compute validation loss over the entire validation set.

    Note: model.train() is intentionally used here so that SSD's
    internal loss computation (which requires train mode) still runs,
    but gradients are disabled via @torch.no_grad().

    Args:
        model  : Detection model.
        loader : Validation DataLoader.
        device : torch.device.

    Returns:
        float: Average validation loss.
    """
    model.train()   # SSD needs train mode to return a loss dict
    total_loss = 0.0
    n = 0

    for images, targets in loader:
        images  = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses    = sum(loss_dict.values())

        total_loss += losses.item()
        n += 1

    avg = total_loss / max(n, 1)
    print(f"  [Val  Loss ] {avg:.4f}")
    return avg
