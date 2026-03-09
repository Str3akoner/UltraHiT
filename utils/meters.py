import torch
import torch.distributed as dist


class AverageMeter:
    def __init__(self, name: str):
        self.name = name
        self.reset()

    def reset(self):
        self.val = 0.0
        self.sum = 0.0
        self.count = 0
        self.avg = 0.0

    def update(self, val: float, n: int = 1):
        self.val = float(val)
        self.sum += float(val) * n
        self.count += n
        self.avg = self.sum / max(self.count, 1)

    def synchronize_between_processes(self, device: torch.device):
        if not dist.is_available() or not dist.is_initialized():
            return
        tensor = torch.tensor([self.sum, self.count], dtype=torch.float64, device=device)
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        self.sum = tensor[0].item()
        self.count = int(tensor[1].item())
        self.avg = self.sum / max(self.count, 1)