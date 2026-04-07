"""
train.py — Training loop (one epoch forward + backward pass)
"""

import torch


def train_one_epoch(model, optimizer, loader, device, epoch):
    """
    Run one full training epoch.

    Args:
        model     : Detection model in train() mode.
        optimizer : torch optimizer.
        loader    : Training DataLoader.
        device    : torch.device.
        epoch     : Current epoch number (for logging).

    Returns:
        float: Average training loss for this epoch.
    """
    model.train()
    total_loss = 0.0
    n = 0

    for images, targets in loader:
        images  = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses    = sum(loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        total_loss += losses.item()
        n += 1

    avg = total_loss / max(n, 1)
    print(f"  [Epoch {epoch:03d}] Train Loss: {avg:.4f}")
    return avg
