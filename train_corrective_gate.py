import argparse
import logging
import os
import time
from pathlib import Path
from typing import Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.tensorboard import SummaryWriter
import torch.distributed as dist
from datasets import SeqDataset_Corrective_Gate
from models.seq_model import ClassificationModelSeq
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

MODEL_NAMES = sorted(
    name for name in models.__dict__
    if name.islower() and not name.startswith("__") and callable(models.__dict__[name])
)



def parse_args():
    parser = argparse.ArgumentParser(description="Train a sequential binary correction-flag classifier")

    parser.add_argument("--train-csv", required=True, type=str, help="Path to training CSV")
    parser.add_argument("--val-csv", required=True, type=str, help="Path to validation CSV")
    parser.add_argument("--stage", required=True, type=int, help="Stage id")

    parser.add_argument("-a", "--arch", default="resnet50", type=str, help="Backbone architecture")
    parser.add_argument("--log-dir", default="logs", type=str, help="Logging directory")
    parser.add_argument("--exp-name", default="exp", type=str, help="Experiment name")

    parser.add_argument("-j", "--workers", default=4, type=int, help="Number of data loading workers")
    parser.add_argument("--epochs", default=50, type=int, help="Number of total epochs to run")
    parser.add_argument("--start-epoch", default=0, type=int, help="Manual epoch number")
    parser.add_argument("-b", "--batch-size", default=16, type=int, help="Global batch size")
    parser.add_argument("--lr", default=5e-3, type=float, help="Initial learning rate")
    parser.add_argument("--weight-decay", default=1e-4, type=float, help="Weight decay")
    parser.add_argument("--momentum", default=0.9, type=float, help="Momentum for SGD")
    parser.add_argument("--optimizer", default="adam", choices=["adam", "sgd"], help="Optimizer")
    parser.add_argument("--scheduler", default="step", choices=["step", "cosine"], help="LR scheduler")
    parser.add_argument("--step-size", default=20, type=int, help="StepLR step size")
    parser.add_argument("--step-gamma", default=0.1, type=float, help="StepLR gamma")

    parser.add_argument("--resume", default="", type=str, help="Path to a checkpoint for resuming")
    parser.add_argument("--pretrain", default="", type=str, help="Path to a pretrained checkpoint")
    parser.add_argument("--disable-backbone-pretrain", action="store_true", help="Disable ImageNet backbone pretraining")
    parser.add_argument("-e", "--evaluate", action="store_true", help="Evaluate only")
    parser.add_argument("--seed", default=42, type=int, help="Random seed")

    parser.add_argument("--action-type", default="discrete", type=str, help="Action type")
    parser.add_argument("--reweight", default="none", choices=["none", "weighted", "sqrt-weighted", "class-balanced"])
    parser.add_argument("--class-balanced-beta", default=0.99, type=float, help="Beta for class-balanced weighting")
    parser.add_argument("--loss-type", default="ce", choices=["ce", "focal"], help="Classification loss")

    parser.add_argument("--gaussian", default=0.0, type=float, help="Gaussian noise std")
    parser.add_argument("--colorjitter", default=0, type=int, help="Enable ColorJitter if > 0")
    parser.add_argument("--randomaffine", default=0, type=int, help="Enable RandomAffine if > 0")
    parser.add_argument("--input-size", nargs=2, type=int, default=[224, 224], help="Input image size as H W")
    parser.add_argument("--seq-len", default=5, type=int, help="Sequence length")
    parser.add_argument("--past-k", default=4, type=int, help="Number of past actions")
    parser.add_argument("--dropout", default=0.0, type=float, help="Dropout ratio")
    parser.add_argument("--normalizer", default="imagenet", choices=["imagenet", "stat", "original_imagenet"])
    parser.add_argument("--interpolation", default="nearest", choices=["nearest", "bilinear"])

    parser.add_argument("--print-freq", default=100, type=int, help="Logging frequency")
    parser.add_argument("--val-every", default=1, type=int, help="Validate every N epochs")
    parser.add_argument("--save-every", default=1, type=int, help="Save checkpoint every N epochs")

    return parser.parse_args()


def compute_class_weights(counts: np.ndarray, strategy: str, beta: float = 0.99) -> np.ndarray:
    counts = np.asarray(counts, dtype=np.float32)

    if counts.ndim != 1 or len(counts) != 2:
        raise ValueError(f"Expected binary class counts, but got: {counts}")

    if counts.sum() == 0:
        raise ValueError("Empty class counts")

    if strategy == "none":
        return np.ones_like(counts, dtype=np.float32)

    if np.any(counts == 0):
        raise ValueError(f"Cannot compute class weights with zero-count classes: {counts.tolist()}")

    freqs = counts / counts.sum()

    if strategy == "weighted":
        weights = (1.0 / freqs) / len(counts)
    elif strategy == "sqrt-weighted":
        weights = np.sqrt((1.0 / freqs) / len(counts))
    elif strategy == "class-balanced":
        effective_num = 1.0 - np.power(beta, counts)
        weights = (1.0 - beta) / effective_num
        weights = weights / weights.mean()
    else:
        raise ValueError(f"Unsupported reweight strategy: {strategy}")

    return weights.astype(np.float32)


def build_model(args, logger: logging.Logger) -> nn.Module:
    model = ClassificationModelSeq(
        arch=args.arch,
        num_actions=2,
        dropout=args.dropout,
        action_type=args.action_type,
        pretrain_backbbone=not args.disable_backbone_pretrain,
        use_distancer=False,
    )

    logger.info(f"Created model: {args.arch}")

    if args.pretrain:
        if not os.path.isfile(args.pretrain):
            raise FileNotFoundError(f"Pretrained checkpoint not found: {args.pretrain}")
        logger.info(f"Loading pretrained checkpoint from: {args.pretrain}")
        checkpoint = torch.load(args.pretrain, map_location="cpu")
        state_dict = checkpoint["state_dict"] if isinstance(checkpoint, dict) and "state_dict" in checkpoint else checkpoint
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        logger.info(f"Pretrained load finished. Missing keys: {len(missing)}, Unexpected keys: {len(unexpected)}")

    return model


def build_optimizer(args, model: nn.Module):
    if args.pretrain:
        actor_params = list(model.actors.parameters())
        backbone_params = [v for k, v in model.named_parameters() if not k.startswith("actors.")]
        if args.optimizer == "sgd":
            return torch.optim.SGD(
                [
                    {"params": backbone_params, "lr": args.lr * 0.5},
                    {"params": actor_params, "lr": args.lr},
                ],
                momentum=args.momentum,
                weight_decay=args.weight_decay,
            )
        return torch.optim.Adam(
            [
                {"params": backbone_params, "lr": args.lr * 0.5},
                {"params": actor_params, "lr": args.lr},
            ],
            weight_decay=args.weight_decay,
        )

    if args.optimizer == "sgd":
        return torch.optim.SGD(
            model.parameters(),
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
        )

    return torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )


def build_scheduler(args, optimizer):
    if args.scheduler == "cosine":
        return CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    if args.scheduler == "step":
        return StepLR(optimizer, step_size=args.step_size, gamma=args.step_gamma)
    raise ValueError(f"Unsupported scheduler: {args.scheduler}")


def build_criterion(args, class_weights: torch.Tensor, device: torch.device) -> nn.Module:
    if args.loss_type == "ce":
        return nn.CrossEntropyLoss(weight=class_weights).to(device)
    if args.loss_type == "focal":
        return FocalLoss(weight=class_weights).to(device)
    raise ValueError(f"Unsupported loss type: {args.loss_type}")


def top1_accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    preds = logits.argmax(dim=1)
    correct = (preds == targets).sum().item()
    return 100.0 * correct / max(targets.size(0), 1)


def log_train_step(
    logger: logging.Logger,
    epoch: int,
    total_epochs: int,
    step: int,
    total_steps: int,
    batch_time: AverageMeter,
    data_time: AverageMeter,
    loss_meter: AverageMeter,
    acc_meter: AverageMeter,
):
    logger.info(
        f"Epoch [{epoch}/{total_epochs}] "
        f"Step [{step}/{total_steps}] "
        f"Time {batch_time.val:.3f} ({batch_time.avg:.3f}) "
        f"Data {data_time.val:.3f} ({data_time.avg:.3f}) "
        f"Loss {loss_meter.val:.4f} ({loss_meter.avg:.4f}) "
        f"Acc@1 {acc_meter.val:.2f} ({acc_meter.avg:.2f})"
    )


def train_one_epoch(
    train_loader: DataLoader,
    model: nn.Module,
    criterion: nn.Module,
    optimizer,
    device: torch.device,
    epoch: int,
    args,
    logger: logging.Logger,
    writer: Optional[SummaryWriter] = None,
):
    model.train()

    batch_time = AverageMeter("batch_time")
    data_time = AverageMeter("data_time")
    loss_meter = AverageMeter("loss")
    acc_meter = AverageMeter("acc1")

    end = time.time()

    for step, (img_seq, past_actions, targets) in enumerate(train_loader, start=1):
        data_time.update(time.time() - end)

        img_seq = img_seq.to(device, non_blocking=True)
        past_actions = past_actions.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        outputs = model(img_seq, past_actions)
        logits = outputs["policy"]
        loss = criterion(logits, targets)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        acc1 = top1_accuracy(logits, targets)
        loss_meter.update(loss.item(), img_seq.size(0))
        acc_meter.update(acc1, img_seq.size(0))

        batch_time.update(time.time() - end)
        end = time.time()

        if is_main_process() and (step % args.print_freq == 0 or step == len(train_loader)):
            log_train_step(
                logger=logger,
                epoch=epoch,
                total_epochs=args.epochs,
                step=step,
                total_steps=len(train_loader),
                batch_time=batch_time,
                data_time=data_time,
                loss_meter=loss_meter,
                acc_meter=acc_meter,
            )

    loss_meter.synchronize_between_processes(device)
    acc_meter.synchronize_between_processes(device)

    if is_main_process():
        logger.info(f"Train Epoch [{epoch}] Loss {loss_meter.avg:.4f} Acc@1 {acc_meter.avg:.2f}")
        if writer is not None:
            writer.add_scalar("Loss", loss_meter.avg, epoch)
            writer.add_scalar("Acc1", acc_meter.avg, epoch)

    return {
        "loss": loss_meter.avg,
        "acc1": acc_meter.avg,
    }


@torch.no_grad()
def validate_for_correction_flag(
    val_loader: DataLoader,
    model: nn.Module,
    device: torch.device,
    epoch: int,
    args,
    logger: logging.Logger,
    writer: Optional[SummaryWriter] = None,
):
    model.eval()

    total_correct = torch.tensor(0.0, device=device)
    total_samples = torch.tensor(0.0, device=device)

    recall_correct_c0 = torch.tensor(0.0, device=device)
    recall_total_c0 = torch.tensor(0.0, device=device)
    recall_correct_c1 = torch.tensor(0.0, device=device)
    recall_total_c1 = torch.tensor(0.0, device=device)

    inclusive_tp = torch.tensor(0.0, device=device)
    inclusive_fp = torch.tensor(0.0, device=device)
    inclusive_tn = torch.tensor(0.0, device=device)
    inclusive_fn = torch.tensor(0.0, device=device)

    for img_seq, past_actions, targets in val_loader:
        img_seq = img_seq.to(device, non_blocking=True)
        past_actions = past_actions.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        outputs = model(img_seq, past_actions)
        logits = outputs["policy"]
        preds = torch.argmax(logits, dim=1)

        hits = targets.gather(1, preds.view(-1, 1)).squeeze(1)

        total_correct += hits.sum()
        total_samples += targets.size(0)

        recall_total_c0 += targets[:, 0].sum()
        recall_total_c1 += targets[:, 1].sum()
        recall_correct_c0 += (hits * targets[:, 0]).sum()
        recall_correct_c1 += (hits * targets[:, 1]).sum()

        pred_is_1 = preds == 1
        pred_is_0 = preds == 0
        target_allows_1 = targets[:, 1] == 1.0
        target_allows_0 = targets[:, 0] == 1.0
        target_only_1 = (targets[:, 1] == 1.0) & (targets[:, 0] == 0.0)
        target_only_0 = (targets[:, 0] == 1.0) & (targets[:, 1] == 0.0)

        inclusive_tp += (pred_is_1 & target_allows_1).sum()
        inclusive_tn += (pred_is_0 & target_allows_0).sum()
        inclusive_fp += (pred_is_1 & target_only_0).sum()
        inclusive_fn += (pred_is_0 & target_only_1).sum()

    counters = [
        total_correct,
        total_samples,
        recall_correct_c0,
        recall_total_c0,
        recall_correct_c1,
        recall_total_c1,
        inclusive_tp,
        inclusive_fp,
        inclusive_tn,
        inclusive_fn,
    ]
    if dist.is_available() and dist.is_initialized():
        for counter in counters:
            dist.all_reduce(counter, op=dist.ReduceOp.SUM)

    eps = 1e-9
    acc = (total_correct / (total_samples + eps)).item() * 100.0
    recall_c0 = (recall_correct_c0 / (recall_total_c0 + eps)).item() * 100.0
    recall_c1 = (recall_correct_c1 / (recall_total_c1 + eps)).item() * 100.0

    precision = (inclusive_tp / (inclusive_tp + inclusive_fp + eps)).item()
    recall = (inclusive_tp / (inclusive_tp + inclusive_fn + eps)).item()
    f1 = 2.0 * precision * recall / (precision + recall + eps)

    if is_main_process():
        logger.info(f"[VAL] Epoch {epoch}: Overall Acc@1 = {acc:.2f}%")
        logger.info(f"Class 0 Recall (inclusive) = {recall_c0:.2f}% ({int(recall_correct_c0.item())}/{int(recall_total_c0.item())})")
        logger.info(f"Class 1 Recall (inclusive) = {recall_c1:.2f}% ({int(recall_correct_c1.item())}/{int(recall_total_c1.item())})")
        logger.info("-" * 50)
        logger.info("Inclusive metrics")
        logger.info(f"Precision = {precision:.4f}")
        logger.info(f"Recall    = {recall:.4f}")
        logger.info(f"F1-score  = {f1:.4f}")
        logger.info(
            f"TP = {int(inclusive_tp.item())}, "
            f"FP = {int(inclusive_fp.item())}, "
            f"TN = {int(inclusive_tn.item())}, "
            f"FN = {int(inclusive_fn.item())}"
        )
        logger.info("-" * 50)

        if writer is not None:
            writer.add_scalar("Acc_Overall", acc, epoch)
            writer.add_scalar("Recall_C0_Inclusive", recall_c0, epoch)
            writer.add_scalar("Recall_C1_Inclusive", recall_c1, epoch)
            writer.add_scalar("Precision_Inclusive", precision, epoch)
            writer.add_scalar("Recall_Inclusive", recall, epoch)
            writer.add_scalar("F1_Inclusive", f1, epoch)

    return {
        "acc": acc,
        "recall_c0": recall_c0,
        "recall_c1": recall_c1,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def main():
    args = parse_args()
    init_distributed_mode(args)
    device = get_device(args)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    log_dir = Path(args.log_dir) / f"stage{args.stage}" / args.exp_name / timestamp
    logger = create_logger(log_dir, args.rank)
    if is_main_process():
        save_args(args, log_dir)
    seed_everything(args.seed, args.rank)

    if args.batch_size % args.world_size != 0:
        raise ValueError(
            f"Global batch size ({args.batch_size}) must be divisible by world size ({args.world_size})"
        )

    per_process_batch_size = args.batch_size // args.world_size

    train_transform = build_transform(args, is_train=True)
    val_transform = build_transform(args, is_train=False)

    train_dataset = SeqDataset_Corrective_Gate(
        csv_path=args.train_csv,
        transform=train_transform,
        stage=args.stage,
        seq_len=args.seq_len,
        past_k=args.past_k,
        mode="train",
    )
    val_dataset = SeqDataset_Corrective_Gate(
        csv_path=args.val_csv,
        transform=val_transform,
        stage=args.stage,
        seq_len=args.seq_len,
        past_k=args.past_k,
        mode="val",
    )

    class_counts = train_dataset.class_counts
    class_weights_np = compute_class_weights(
        counts=class_counts,
        strategy=args.reweight,
        beta=args.class_balanced_beta,
    )
    class_weights = torch.tensor(class_weights_np, dtype=torch.float32, device=device)

    if is_main_process():
        logger.info(f"Distributed training: {args.distributed}")
        logger.info(f"World size: {args.world_size}")
        logger.info(f"Device: {device}")
        logger.info(f"Train samples: {len(train_dataset)}")
        logger.info(f"Val samples: {len(val_dataset)}")
        logger.info(f"Train class counts: {class_counts.tolist()}")
        logger.info(f"Class weights: {class_weights_np.tolist()}")

    train_sampler = DistributedSampler(train_dataset, shuffle=True, drop_last=True) if args.distributed else None
    val_sampler = DistributedSampler(val_dataset, shuffle=False, drop_last=False) if args.distributed else None

    train_loader = DataLoader(
        train_dataset,
        batch_size=per_process_batch_size,
        sampler=train_sampler,
        shuffle=(train_sampler is None),
        num_workers=args.workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=per_process_batch_size,
        sampler=val_sampler,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )

    raw_model = build_model(args, logger).to(device)
    optimizer = build_optimizer(args, raw_model)
    scheduler = build_scheduler(args, optimizer)
    criterion = build_criterion(args, class_weights, device)

    start_epoch, best_f1 = load_resume_checkpoint(args, raw_model, optimizer, scheduler, logger)

    model = raw_model
    if args.distributed:
        model = DDP(
            raw_model,
            device_ids=[args.local_rank] if torch.cuda.is_available() else None,
            output_device=args.local_rank if torch.cuda.is_available() else None,
            find_unused_parameters=False,
        )

    train_writer = SummaryWriter(log_dir / "train") if is_main_process() else None
    val_writer = SummaryWriter(log_dir / "val") if is_main_process() else None

    if args.evaluate:
        validate_for_correction_flag(
            val_loader=val_loader,
            model=model,
            device=device,
            epoch=start_epoch,
            args=args,
            logger=logger,
            writer=val_writer,
        )
        if train_writer is not None:
            train_writer.close()
        if val_writer is not None:
            val_writer.close()
        cleanup_distributed()
        return

    for epoch in range(start_epoch + 1, args.epochs + 1):
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

        val_stats = None
        if epoch % args.val_every == 0:
            if val_sampler is not None:
                val_sampler.set_epoch(epoch)

            val_stats = validate_for_correction_flag(
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
            state = {
                "epoch": epoch,
                "arch": args.arch,
                "state_dict": unwrap_model(model).state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "best_f1": best_f1,
                "args": vars(args),
                "train_stats": train_stats,
                "val_stats": val_stats,
            }

            save_checkpoint(state, log_dir / "checkpoint_last.pth.tar")

            if epoch % args.save_every == 0:
                save_checkpoint(state, log_dir / f"checkpoint_epoch_{epoch}.pth.tar")

            if val_stats is not None and val_stats["f1"] > best_f1:
                best_f1 = val_stats["f1"]
                state["best_f1"] = best_f1
                save_checkpoint(state, log_dir / "checkpoint_best.pth.tar")
                logger.info(f"Saved new best checkpoint at epoch {epoch} with F1 = {best_f1:.4f}")

    if train_writer is not None:
        train_writer.close()
    if val_writer is not None:
        val_writer.close()

    cleanup_distributed()


if __name__ == "__main__":
    main()