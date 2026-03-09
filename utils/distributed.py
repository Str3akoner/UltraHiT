import os
import random
from typing import Optional

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist


def init_distributed_mode(args):
    args.distributed = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    args.rank = int(os.environ.get("RANK", 0))
    args.world_size = int(os.environ.get("WORLD_SIZE", 1))
    args.local_rank = int(os.environ.get("LOCAL_RANK", 0))

    if args.distributed:
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        if torch.cuda.is_available():
            torch.cuda.set_device(args.local_rank)
        dist.init_process_group(backend=backend, init_method="env://")
        dist.barrier()


def cleanup_distributed():
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def is_main_process() -> bool:
    return not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0


def seed_everything(seed: Optional[int], rank: int):
    if seed is None:
        cudnn.benchmark = True
        return

    seed = seed + rank
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    cudnn.deterministic = True
    cudnn.benchmark = False


def get_device(args) -> torch.device:
    if torch.cuda.is_available():
        return torch.device(f"cuda:{args.local_rank}" if args.distributed else "cuda")
    return torch.device("cpu")