import argparse
import json
import logging
import os
import random
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from torch.utils.tensorboard import SummaryWriter
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision.transforms import InterpolationMode
from utils import (
    AverageMeter,
    FocalLoss,
    build_transform,
    cleanup_distributed,
    create_logger,
    get_device,
    init_distributed_mode,
    is_main_process,
    load_resume_checkpoint,
    save_args,
    save_checkpoint,
    seed_everything,
    unwrap_model,
)

ACTION_KEYS = ["y", "x"]  # "y" : continue , "x" : stop
ACTION_TO_INDEX = {k: i for i, k in enumerate(ACTION_KEYS)}
INDEX_TO_ACTION = {i: k for k, i in ACTION_TO_INDEX.items()}

SUPPORTED_ARCHS = [
    name for name in [
        "resnet18",
        "resnet34",
        "resnet50",
        "resnet101",
        "convnext_tiny",
        "convnext_small",
        "convnext_base",
        "convnext_large",
    ]
    if hasattr(models, name)
]


def parse_args():
    parser = argparse.ArgumentParser(description="Train an image classifier")

    # Data
    parser.add_argument("--train-csv", required=True, type=str, help="Path to training CSV")
    parser.add_argument("--val-csv", required=True, type=str, help="Path to validation CSV")
    parser.add_argument("--stage", required=True, type=int, help="Stage id to train/evaluate")
    parser.add_argument("--input-size", nargs=2, type=int, default=[224, 224], help="Input image size as H W")
    parser.add_argument("--resize-width", default=1280, type=int, help="Width before crop")
    parser.add_argument("--resize-height", default=960, type=int, help="Height before crop")
    parser.add_argument("--crop-left", default=200, type=int, help="Crop left")
    parser.add_argument("--crop-right", default=824, type=int, help="Crop right")
    parser.add_argument("--crop-top", default=103, type=int, help="Crop top")
    parser.add_argument("--crop-bottom", default=666, type=int, help="Crop bottom")

    # Model
    parser.add_argument("--arch", default="resnet50", choices=SUPPORTED_ARCHS, help="Backbone architecture")
    parser.add_argument("--dropout", default=0.0, type=float, help="Dropout before heads")
    parser.add_argument("--num-actions", default=2, type=int, help="Number of classes")
    parser.add_argument("--pretrain", default="", type=str, help="Optional checkpoint path to initialize the model")
    parser.add_argument("--no-imagenet-pretrain", action="store_true", help="Disable ImageNet pretrained backbone")

    # Training
    parser.add_argument("--epochs", default=20, type=int, help="Total epochs")
    parser.add_argument("--batch-size", default=256, type=int, help="Global batch size")
    parser.add_argument("--workers", default=4, type=int, help="Number of workers per process")
    parser.add_argument("--lr", default=1e-4, type=float, help="Learning rate")
    parser.add_argument("--weight-decay", default=1e-3, type=float, help="Weight decay")
    parser.add_argument("--momentum", default=0.9, type=float, help="Momentum (for SGD)")
    parser.add_argument("--optimizer", default="adam", choices=["adam", "sgd"], help="Optimizer")
    parser.add_argument("--scheduler", default="cosine", choices=["cosine", "step"], help="LR scheduler")
    parser.add_argument("--step-size", default=20, type=int, help="StepLR step size")
    parser.add_argument("--step-gamma", default=0.1, type=float, help="StepLR gamma")
    parser.add_argument("--seed", default=42, type=int, help="Random seed")
    parser.add_argument("--print-freq", default=50, type=int, help="Logging frequency")
    parser.add_argument("--evaluate", action="store_true", help="Run evaluation only")

    # Loss
    parser.add_argument("--loss-type", default="ce", choices=["ce", "focal"], help="Classification loss")
    parser.add_argument("--label-smoothing", default=0.1, type=float, help="Label smoothing for CE")
    parser.add_argument("--aux-coef", default=0.0, type=float, help="Weight for auxiliary distance loss")
    parser.add_argument(
        "--reweight",
        default="none",
        choices=["none", "weighted", "sqrt-weighted", "class-balanced"],
        help="Class reweighting strategy",
    )
    parser.add_argument("--class-balanced-beta", default=0.99, type=float, help="Beta for class-balanced weights")

    # Augmentation / normalization
    parser.add_argument("--randomresizedcrop", action="store_true", help="Use RandomResizedCrop for training")
    parser.add_argument(
        "--randomresizedcrop-scale-min",
        default=0.08,
        type=float,
        help="Lower bound of RandomResizedCrop scale",
    )
    parser.add_argument("--randomhorizontalflip", action="store_true", help="Use RandomHorizontalFlip for training")
    parser.add_argument(
        "--interpolation",
        default="nearest",
        choices=["nearest", "bilinear"],
        help="Interpolation for final resize",
    )
    parser.add_argument(
        "--normalizer",
        default="imagenet",
        choices=["imagenet", "stat", "original_imagenet"],
        help="Normalization strategy",
    )

    # Logging
    parser.add_argument("--log-dir", default="logs", type=str, help="Root log directory")
    parser.add_argument("--exp-name", default="exp", type=str, help="Experiment name")
    parser.add_argument("--save-every", default=1, type=int, help="Save checkpoint every N epochs")

    return parser.parse_args()


def get_default_weights(arch: str, use_imagenet_pretrain: bool):
    if not use_imagenet_pretrain:
        return None
    try:
        return models.get_model_weights(arch).DEFAULT
    except Exception:
        return None


def check_required_columns(df: pd.DataFrame, required_columns: List[str], csv_path: str):
    missing = [c for c in required_columns if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in {csv_path}: {missing}")

def prepare_metadata(csv_path: str, stage: int, logger: logging.Logger) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    required_columns = ["path", "action_key", "stage", "supervise_distance"] + [f"d{i}" for i in range(1, 7)]
    check_required_columns(df, required_columns, csv_path)

    original_len = len(df)
    df = df[df["stage"] == stage].copy()
    df["action_key"] = df["action_key"].astype(str)

    valid_action_mask = df["action_key"].isin(ACTION_TO_INDEX.keys())
    invalid_action_count = (~valid_action_mask).sum()
    if invalid_action_count > 0:
        logger.warning(f"Dropping {invalid_action_count} rows with invalid action_key in {csv_path}")
        df = df[valid_action_mask]

    exists_mask = df["path"].map(lambda x: isinstance(x, str) and os.path.exists(x))
    missing_path_count = (~exists_mask).sum()
    if missing_path_count > 0:
        logger.warning(f"Dropping {missing_path_count} rows with missing image files in {csv_path}")
        df = df[exists_mask]

    df = df.reset_index(drop=True)

    logger.info(
        f"Loaded {csv_path}: original={original_len}, stage_filtered={len(df)} (stage={stage})"
    )

    if len(df) == 0:
        raise ValueError(f"No valid samples found in {csv_path} for stage={stage}")

    return df


class CsvImageDataset(Dataset):
    def __init__(
        self,
        metadata: pd.DataFrame,
        transform,
        resize_size: Tuple[int, int],
        crop_box: Tuple[int, int, int, int],
    ):
        self.metadata = metadata.reset_index(drop=True)
        self.transform = transform
        self.resize_size = resize_size
        self.crop_box = crop_box

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, index: int):
        row = self.metadata.iloc[index]

        image = Image.open(row["path"]).convert("RGB")
        image = image.resize(self.resize_size, resample=Image.Resampling.LANCZOS, reducing_gap=3)
        image = image.crop(self.crop_box)
        image = self.transform(image)

        target = torch.tensor(ACTION_TO_INDEX[str(row["action_key"])], dtype=torch.long)
        supervise_distance = torch.tensor(bool(row["supervise_distance"]), dtype=torch.bool)
        distances = torch.tensor([row[f"d{i}"] for i in range(1, 7)], dtype=torch.float32)

        return image, target, supervise_distance, distances


class IdentityModule(nn.Module):
    def forward(self, x):
        return x


class DiscreteHead(nn.Module):
    def __init__(self, in_features: int, out_features: int, dropout: float, arch: str):
        super().__init__()
        if "resnet" in arch:
            self.trunk = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(in_features, out_features),
            )
        elif "convnext" in arch:
            self.trunk = nn.Sequential(
                nn.Dropout(dropout),
                nn.Flatten(1),
                nn.LayerNorm(in_features, eps=1e-6),
                nn.Linear(in_features, out_features),
            )
        else:
            raise ValueError(f"Unsupported architecture: {arch}")

    def forward(self, x):
        return self.trunk(x)


def build_model(args, logger: logging.Logger):
    weights = get_default_weights(args.arch, not args.no_imagenet_pretrain)
    model = models.__dict__[args.arch](weights=weights)

    if "resnet" in args.arch:
        in_features = model.fc.in_features
        model.fc = nn.Identity()
    elif "convnext" in args.arch:
        in_features = model.classifier[2].in_features
        model.classifier = IdentityModule()
    else:
        raise ValueError(f"Unsupported architecture: {args.arch}")

    model.actors = DiscreteHead(in_features, args.num_actions, args.dropout, args.arch)
    model.distancers = DiscreteHead(in_features, 6, args.dropout, args.arch)

    if args.pretrain:
        if os.path.isfile(args.pretrain):
            logger.info(f"Loading checkpoint from: {args.pretrain}")
            checkpoint = torch.load(args.pretrain, map_location="cpu")
            state_dict = checkpoint["state_dict"] if isinstance(checkpoint, dict) and "state_dict" in checkpoint else checkpoint
            missing, unexpected = model.load_state_dict(state_dict, strict=False)
            logger.info(f"Checkpoint loaded. Missing keys: {len(missing)}, Unexpected keys: {len(unexpected)}")
        else:
            raise FileNotFoundError(f"Checkpoint not found: {args.pretrain}")

    logger.info(f"Created model: {args.arch}")
    return model


def compute_class_weights(train_df: pd.DataFrame, num_classes: int, strategy: str, beta: float) -> np.ndarray:
    counts = np.zeros(num_classes, dtype=np.float32)
    for i in range(num_classes):
        counts[i] = (train_df["action_key"] == INDEX_TO_ACTION[i]).sum()

    if np.any(counts == 0):
        raise ValueError(f"At least one class has zero samples: counts={counts.tolist()}")

    freqs = counts / counts.sum()

    if strategy == "none":
        weights = np.ones_like(counts, dtype=np.float32)
    elif strategy == "weighted":
        weights = (1.0 / freqs) / num_classes
    elif strategy == "sqrt-weighted":
        weights = np.sqrt((1.0 / freqs) / num_classes)
    elif strategy == "class-balanced":
        effective_num = 1.0 - np.power(beta, counts)
        weights = (1.0 - beta) / effective_num
        weights = weights / weights.mean()
    else:
        raise ValueError(f"Unsupported reweight strategy: {strategy}")

    return weights.astype(np.float32)


def build_criterion(args, class_weights: torch.Tensor, device: torch.device):
    if args.loss_type == "ce":
        return nn.CrossEntropyLoss(weight=class_weights, label_smoothing=args.label_smoothing).to(device)
    if args.loss_type == "focal":
        return FocalLoss(weight=class_weights).to(device)
    raise ValueError(f"Unsupported loss type: {args.loss_type}")


def build_optimizer(args, model: nn.Module):
    if args.optimizer == "adam":
        return torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    if args.optimizer == "sgd":
        return torch.optim.SGD(
            model.parameters(),
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
        )
    raise ValueError(f"Unsupported optimizer: {args.optimizer}")


def build_scheduler(args, optimizer):
    if args.scheduler == "cosine":
        return CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    if args.scheduler == "step":
        return StepLR(optimizer, step_size=args.step_size, gamma=args.step_gamma)
    raise ValueError(f"Unsupported scheduler: {args.scheduler}")


def top1_accuracy(logits: torch.Tensor, target: torch.Tensor) -> float:
    pred = logits.argmax(dim=1)
    correct = (pred == target).sum().item()
    return 100.0 * correct / target.size(0)


def masked_mse_loss(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = mask.bool()
    if mask.sum() == 0:
        return pred.new_zeros(())
    sample_loss = ((pred - target) ** 2).mean(dim=1)
    return sample_loss[mask].mean()


def log_epoch_summary(
    logger: logging.Logger,
    split: str,
    epoch: int,
    loss_meter: AverageMeter,
    acc_meter: AverageMeter,
    aux_meter: AverageMeter = None,
):
    msg = f"{split} Epoch [{epoch}] Loss {loss_meter.avg:.4f} Acc@1 {acc_meter.avg:.2f}"
    if aux_meter is not None:
        msg += f" Aux {aux_meter.avg:.4f}"
    logger.info(msg)


def train_one_epoch(
    train_loader: DataLoader,
    model: nn.Module,
    criterion: nn.Module,
    optimizer,
    device: torch.device,
    epoch: int,
    args,
    logger: logging.Logger,
    writer: SummaryWriter = None,
):
    model.train()

    batch_time = AverageMeter("batch_time")
    data_time = AverageMeter("data_time")
    loss_meter = AverageMeter("loss")
    aux_meter = AverageMeter("aux_loss")
    acc_meter = AverageMeter("acc1")

    end = time.time()

    for step, (images, target, supervise_distance, distances) in enumerate(train_loader):
        data_time.update(time.time() - end)

        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        supervise_distance = supervise_distance.to(device, non_blocking=True)
        distances = distances.to(device, non_blocking=True)

        features = model(images)
        model_ref = unwrap_model(model)
        logits = model_ref.actors(features)
        pred_distances = model_ref.distancers(features)

        cls_loss = criterion(logits, target)
        aux_loss = args.aux_coef * masked_mse_loss(pred_distances, distances, supervise_distance)
        total_loss = cls_loss + aux_loss

        optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        optimizer.step()

        acc1 = top1_accuracy(logits, target)

        loss_meter.update(total_loss.item(), images.size(0))
        aux_meter.update(aux_loss.item(), images.size(0))
        acc_meter.update(acc1, images.size(0))

        batch_time.update(time.time() - end)
        end = time.time()

        if is_main_process() and (step % args.print_freq == 0 or step == len(train_loader) - 1):
            logger.info(
                f"Train Epoch [{epoch}/{args.epochs}] "
                f"Step [{step + 1}/{len(train_loader)}] "
                f"Time {batch_time.val:.3f} ({batch_time.avg:.3f}) "
                f"Data {data_time.val:.3f} ({data_time.avg:.3f}) "
                f"Loss {loss_meter.val:.4f} ({loss_meter.avg:.4f}) "
                f"Aux {aux_meter.val:.4f} ({aux_meter.avg:.4f}) "
                f"Acc@1 {acc_meter.val:.2f} ({acc_meter.avg:.2f})"
            )

    loss_meter.synchronize_between_processes(device)
    aux_meter.synchronize_between_processes(device)
    acc_meter.synchronize_between_processes(device)

    if is_main_process():
        log_epoch_summary(logger, "Train", epoch, loss_meter, acc_meter, aux_meter)
        if writer is not None:
            writer.add_scalar("Loss", loss_meter.avg, epoch)
            writer.add_scalar("AuxLoss", aux_meter.avg, epoch)
            writer.add_scalar("Acc1", acc_meter.avg, epoch)

    return {
        "loss": loss_meter.avg,
        "aux_loss": aux_meter.avg,
        "acc1": acc_meter.avg,
    }


@torch.no_grad()
def validate(
    val_loader: DataLoader,
    model: nn.Module,
    device: torch.device,
    epoch: int,
    args,
    logger: logging.Logger,
    writer: SummaryWriter = None,
):
    model.eval()

    criterion = nn.CrossEntropyLoss().to(device)

    batch_time = AverageMeter("batch_time")
    loss_meter = AverageMeter("loss")
    acc_meter = AverageMeter("acc1")

    end = time.time()

    for step, (images, target, _, _) in enumerate(val_loader):
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        features = model(images)
        model_ref = unwrap_model(model)
        logits = model_ref.actors(features)

        loss = criterion(logits, target)
        acc1 = top1_accuracy(logits, target)

        loss_meter.update(loss.item(), images.size(0))
        acc_meter.update(acc1, images.size(0))

        batch_time.update(time.time() - end)
        end = time.time()

    loss_meter.synchronize_between_processes(device)
    acc_meter.synchronize_between_processes(device)

    if is_main_process():
        log_epoch_summary(logger, "Val", epoch, loss_meter, acc_meter)
        if writer is not None:
            writer.add_scalar("Loss", loss_meter.avg, epoch)
            writer.add_scalar("Acc1", acc_meter.avg, epoch)

    return {
        "loss": loss_meter.avg,
        "acc1": acc_meter.avg,
    }


def main():
    args = parse_args()
    init_distributed_mode(args)
    device = get_device(args)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    log_dir = Path(args.log_dir) / f"stage{args.stage}" / args.exp_name / timestamp
    logger = create_logger(log_dir, args.rank)

    if is_main_process():
        logger.info("Starting training")
        logger.info(f"Distributed: {args.distributed}")
        logger.info(f"World size: {args.world_size}")
        logger.info(f"Device: {device}")
    if is_main_process():
        save_args(args, log_dir)
    seed_everything(args.seed, args.rank)

    if args.batch_size % args.world_size != 0:
        raise ValueError(
            f"Global batch size ({args.batch_size}) must be divisible by world size ({args.world_size})"
        )
    per_process_batch_size = args.batch_size // args.world_size

    train_df = prepare_metadata(args.train_csv, args.stage, logger)
    val_df = prepare_metadata(args.val_csv, args.stage, logger)

    class_weights_np = compute_class_weights(
        train_df=train_df,
        num_classes=args.num_actions,
        strategy=args.reweight,
        beta=args.class_balanced_beta,
    )
    class_weights = torch.tensor(class_weights_np, dtype=torch.float32, device=device)

    if is_main_process():
        logger.info(f"Class weights: {class_weights_np.tolist()}")
        logger.info(
            f"Train class counts: "
            f"{[(k, int((train_df['action_key'] == k).sum())) for k in ACTION_KEYS]}"
        )

    train_transform = build_transform(args, is_train=True)
    val_transform = build_transform(args, is_train=False)

    resize_size = (args.resize_width, args.resize_height)
    crop_box = (args.crop_left, args.crop_top, args.crop_right, args.crop_bottom)

    train_dataset = CsvImageDataset(
        metadata=train_df,
        transform=train_transform,
        resize_size=resize_size,
        crop_box=crop_box,
    )
    val_dataset = CsvImageDataset(
        metadata=val_df,
        transform=val_transform,
        resize_size=resize_size,
        crop_box=crop_box,
    )

    train_sampler = DistributedSampler(train_dataset, shuffle=True) if args.distributed else None
    val_sampler = DistributedSampler(val_dataset, shuffle=False) if args.distributed else None

    train_loader = DataLoader(
        train_dataset,
        batch_size=per_process_batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=args.workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=per_process_batch_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=args.workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )

    model = build_model(args, logger).to(device)

    if args.distributed:
        model = DDP(
            model,
            device_ids=[args.local_rank] if torch.cuda.is_available() else None,
            output_device=args.local_rank if torch.cuda.is_available() else None,
            find_unused_parameters=False,
        )

    criterion = build_criterion(args, class_weights, device)
    optimizer = build_optimizer(args, model)
    scheduler = build_scheduler(args, optimizer)

    train_writer = SummaryWriter(log_dir / "train") if is_main_process() else None
    val_writer = SummaryWriter(log_dir / "val") if is_main_process() else None

    best_val_loss = float("inf")

    if args.evaluate:
        validate(val_loader, model, device, epoch=0, args=args, logger=logger, writer=val_writer)
        if train_writer is not None:
            train_writer.close()
        if val_writer is not None:
            val_writer.close()
        cleanup_distributed()
        return

    for epoch in range(1, args.epochs + 1):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        train_stats = train_one_epoch(
            train_loader=train_loader,
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            args=args,
            logger=logger,
            writer=train_writer,
        )

        val_stats = validate(
            val_loader=val_loader,
            model=model,
            device=device,
            epoch=epoch,
            args=args,
            logger=logger,
            writer=val_writer,
        )

        scheduler.step()

        if is_main_process():
            current_model = unwrap_model(model)
            checkpoint = {
                "epoch": epoch,
                "arch": args.arch,
                "state_dict": current_model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "args": vars(args),
                "train_stats": train_stats,
                "val_stats": val_stats,
            }

            if epoch % args.save_every == 0:
                save_checkpoint(checkpoint, log_dir / f"checkpoint_epoch_{epoch}.pth.tar")

            save_checkpoint(checkpoint, log_dir / "checkpoint_last.pth.tar")

            if val_stats["loss"] < best_val_loss:
                best_val_loss = val_stats["loss"]
                save_checkpoint(checkpoint, log_dir / "checkpoint_best.pth.tar")
                logger.info(f"Saved new best checkpoint at epoch {epoch} (val loss={best_val_loss:.4f})")

    if train_writer is not None:
        train_writer.close()
    if val_writer is not None:
        val_writer.close()

    cleanup_distributed()


if __name__ == "__main__":
    main()