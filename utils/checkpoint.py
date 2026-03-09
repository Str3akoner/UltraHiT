import os
from pathlib import Path
from typing import Dict

import torch

def save_checkpoint(state: Dict, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)


def unwrap_model(model):
    return model.module if hasattr(model, "module") else model


def load_resume_checkpoint(args, model, optimizer, scheduler, logger):
    start_epoch = args.start_epoch
    best_metric = -1.0

    if not args.resume:
        return start_epoch, best_metric

    if not os.path.isfile(args.resume):
        raise FileNotFoundError(f"Resume checkpoint not found: {args.resume}")

    logger.info(f"Loading resume checkpoint from: {args.resume}")
    checkpoint = torch.load(args.resume, map_location="cpu")

    model.load_state_dict(checkpoint["state_dict"], strict=False)

    if "optimizer" in checkpoint and optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer"])
    if "scheduler" in checkpoint and scheduler is not None:
        scheduler.load_state_dict(checkpoint["scheduler"])

    start_epoch = checkpoint.get("epoch", 0)
    best_metric = checkpoint.get("best_val_acc", checkpoint.get("best_f1", -1.0))

    logger.info(f"Resumed from epoch {start_epoch}, best_metric={best_metric:.4f}")
    return start_epoch, best_metric