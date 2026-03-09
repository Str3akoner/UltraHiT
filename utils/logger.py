import json
import logging
from pathlib import Path


def create_logger(log_dir: Path, rank: int) -> logging.Logger:
    logger = logging.getLogger(f"train_logger_rank_{rank}")
    logger.setLevel(logging.INFO)
    logger.propagate = False

    if logger.handlers:
        logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    if rank == 0:
        log_dir.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_dir / "train.log")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)
    else:
        logger.addHandler(logging.NullHandler())

    return logger



def save_args(args, log_dir):
    if getattr(args, "rank", 0) != 0:
        return
    log_dir.mkdir(parents=True, exist_ok=True)
    with open(log_dir / "args.json", "w") as f:
        json.dump(vars(args), f, indent=2)