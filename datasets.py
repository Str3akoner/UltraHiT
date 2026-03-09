import os
import random
import re
from bisect import bisect_left
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset


# -----------------------------------------------------------------------------
# Action mappings
# -----------------------------------------------------------------------------

ACTION_KEYS = ["u", "i", "o", "j", "k", "l", "7", "8", "9", "4", "5", "6", "x"]
ACTION_KEY_TO_INDEX = {key: idx for idx, key in enumerate(ACTION_KEYS)}
ACTION_INDEX_TO_KEY = {idx: key for idx, key in enumerate(ACTION_KEYS)}

# For adaptive corrector:
TARGET_ACTIONS_STAGE1 = {"j", "l", "4", "6", "o", "x"}
TARGET_ACTIONS_STAGE2 = {"j", "l", "i", "k", "4", "6", "5", "8", "7", "u", "o", "x"}

# For corrective gate:
POSITIVE_ACTION_KEYS_STAGE1 = {"j", "l", "4", "6", "o"}
POSITIVE_ACTION_KEYS_STAGE2 = {"j", "l", "i", "k", "4", "6", "5", "8", "u", "o", "x", "7"}

VALID_ACTION_SET = set(ACTION_KEYS)
ACTION_PATTERN = re.compile(r"[uiojkl789456x]", re.IGNORECASE)

_RE_TIMESTEP = re.compile(r"^(\d+)-key", re.IGNORECASE)
_RE_KEY_ACTION = re.compile(r"key_([a-z0-9])", re.IGNORECASE)
_RE_FRAME = re.compile(r"-frame_(\d+)", re.IGNORECASE)


# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------

def _date_folder_from_path(path: str) -> str:
    """
    Extract the recording/date folder from the path.
    If a 'us_image' directory exists in the path, use the previous segment.
    Otherwise, fall back to the parent folder name.
    """
    parts = Path(path).parts
    for i, seg in enumerate(parts):
        if seg == "us_image" and i - 1 >= 0:
            return parts[i - 1]
    return Path(path).parent.name


def _timestep_from_path(path: str) -> Optional[int]:
    """
    Extract timestep from filename prefix.

    Example:
        123-key_j-frame_7.jpg -> 123
    """
    filename = os.path.basename(path)
    match = _RE_TIMESTEP.match(filename)
    if match is None:
        return None
    return int(match.group(1))


def _frame_idx_from_path(path: str) -> int:
    """
    Extract frame index from filename.

    Example:
        ...-frame_7.jpg -> 7
    """
    match = _RE_FRAME.search(Path(path).name)
    if match is None:
        return -1
    return int(match.group(1))


def _single_action_from_key_segment(path: str) -> int:
    """
    Extract the single action from 'key_<x>' in the filename.
    Fall back to 'x' when parsing fails.
    """
    match = _RE_KEY_ACTION.search(Path(path).name)
    if match is None:
        return ACTION_KEY_TO_INDEX["x"]

    action_key = match.group(1).lower()
    return ACTION_KEY_TO_INDEX.get(action_key, ACTION_KEY_TO_INDEX["x"])


def _parse_action_multi(action_str: str) -> List[int]:
    """
    Parse a multi-label action string into a list of unique action indices,
    preserving order.

    Example:
        'ik6' -> [idx(i), idx(k), idx(6)]
    """
    seen = set()
    parsed = []

    for ch in str(action_str):
        if ch in ACTION_KEY_TO_INDEX:
            idx = ACTION_KEY_TO_INDEX[ch]
            if idx not in seen:
                seen.add(idx)
                parsed.append(idx)

    return parsed


def _first_action_idx(action_str: str) -> Optional[int]:
    """
    Return the first valid action index found in the string.
    """
    for ch in str(action_str):
        if ch in ACTION_KEY_TO_INDEX:
            return ACTION_KEY_TO_INDEX[ch]
    return None


# -----------------------------------------------------------------------------
# Dataset for adaptive corrector
# -----------------------------------------------------------------------------

class SeqDataset_Adaptive_Corrector(Dataset):
    """
    Sequential dataset for multi-class correction-action prediction.

    Each sample contains:
      - `seq_len` frames: `past_k` history frames + 1 current frame
      - `past_k` past single-label actions
      - current label:
          - train: scalar class index
          - val: multi-hot mask over all classes

    Filtering rule:
      - train: keep a sample only if its first action is in the target set
      - val: keep a sample if any of its actions is in the target set
    """

    def __init__(
        self,
        csv_path: str,
        transform,
        stage: int = 1,
        seq_len: int = 5,
        past_k: int = 4,
        mode: str = "train",
        crop_box: Tuple[int, int, int, int] = (520, 163, 1144, 729),
        verbose: bool = True,
    ):
        if seq_len != past_k + 1:
            raise ValueError("Expected seq_len == past_k + 1")
        if mode not in ("train", "val"):
            raise ValueError("mode must be either 'train' or 'val'")
        if stage not in (1, 2):
            raise ValueError("Only stage 1 and stage 2 are supported")

        self.csv_path = csv_path
        self.transform = transform
        self.stage = stage
        self.seq_len = seq_len
        self.past_k = past_k
        self.mode = mode
        self.crop_box = crop_box

        self.target_action_keys = (
            set(TARGET_ACTIONS_STAGE1)
            if stage == 1
            else set(TARGET_ACTIONS_STAGE2)
        )
        self.target_action_indices = {
            ACTION_KEY_TO_INDEX[key] for key in self.target_action_keys
        }

        df = pd.read_csv(csv_path)
        if "stage" not in df.columns or "path" not in df.columns or "action_key" not in df.columns:
            raise ValueError("CSV must contain at least: ['stage', 'path', 'action_key']")

        df = df[df["stage"] == stage].reset_index(drop=True)

        self.rows = self._build_rows(df)
        self.by_dt_idx, self.by_dt_paths, self.ts_by_date = self._build_indices(self.rows)

        if self.mode == "val":
            for key in self.by_dt_paths:
                self.by_dt_paths[key].sort(key=_frame_idx_from_path)

        self.samples = self._build_sample_indices()
        self.targets = self._build_targets()

        if self.mode == "train":
            self.class_counts = np.bincount(np.asarray(self.targets, dtype=np.int64), minlength=len(ACTION_KEYS))
        else:
            self.class_counts = None

        if verbose:
            print(
                f"[{self.mode} stage={self.stage}] "
                f"Total rows available for history lookup: {len(self.rows)} | "
                f"Final samples: {len(self.samples)}"
            )

    def __len__(self) -> int:
        return len(self.samples)

    def _build_rows(self, df: pd.DataFrame) -> List[Dict]:
        rows = []

        for _, row in df.iterrows():
            path = str(row["path"])
            date_key = _date_folder_from_path(path)
            timestep = _timestep_from_path(path)

            if timestep is None:
                continue

            action_str = str(row["action_key"])
            first_action = _first_action_idx(action_str)
            multi_action = _parse_action_multi(action_str)

            rows.append(
                {
                    "path": path,
                    "date": date_key,
                    "t": timestep,
                    "a_first": first_action,
                    "a_multi": multi_action,
                }
            )

        return rows

    def _build_indices(
        self,
        rows: List[Dict],
    ) -> Tuple[
        Dict[Tuple[str, int], List[int]],
        Dict[Tuple[str, int], List[str]],
        Dict[str, List[int]],
    ]:
        by_dt_idx = defaultdict(list)
        by_dt_paths = defaultdict(list)
        ts_by_date = defaultdict(set)

        for ridx, row in enumerate(rows):
            key = (row["date"], row["t"])
            by_dt_idx[key].append(ridx)
            by_dt_paths[key].append(row["path"])
            ts_by_date[row["date"]].add(row["t"])

        ts_by_date = {date: sorted(list(ts)) for date, ts in ts_by_date.items()}
        return by_dt_idx, by_dt_paths, ts_by_date

    def _build_sample_indices(self) -> List[int]:
        samples = []

        if self.mode == "train":
            for ridx, row in enumerate(self.rows):
                action_idx = row["a_first"]
                if action_idx is not None and action_idx in self.target_action_indices:
                    samples.append(ridx)
        else:
            for ridx, row in enumerate(self.rows):
                if any(idx in self.target_action_indices for idx in row["a_multi"]):
                    samples.append(ridx)

        return samples

    def _build_targets(self) -> List:
        if self.mode == "train":
            return [
                self.rows[ridx]["a_first"] if self.rows[ridx]["a_first"] is not None else ACTION_KEY_TO_INDEX["x"]
                for ridx in self.samples
            ]

        return [
            self._build_val_mask(self.rows[ridx]["a_multi"])
            for ridx in self.samples
        ]

    def _build_val_mask(self, action_indices: List[int]) -> List[float]:
        mask = np.zeros(len(ACTION_KEYS), dtype=np.float32)

        if len(action_indices) == 0:
            mask[ACTION_KEY_TO_INDEX["x"]] = 1.0
            return mask.tolist()

        for idx in action_indices:
            mask[idx] = 1.0

        return mask.tolist()

    def _load_image(self, path: str) -> torch.Tensor:
        image = Image.open(path).convert("RGB")
        image = image.crop(self.crop_box)
        return self.transform(image)

    def _pick_train_from_timestep(self, date_key: str, timestep: int) -> Optional[Tuple[str, int]]:
        candidates = self.by_dt_idx.get((date_key, timestep), [])
        if not candidates:
            return None

        ridx = random.choice(candidates)
        path = self.rows[ridx]["path"]
        label = _single_action_from_key_segment(path)
        return path, label

    def _pick_val_from_timestep(self, date_key: str, timestep: int) -> Optional[Tuple[str, int]]:
        candidates = self.by_dt_paths.get((date_key, timestep), [])
        if not candidates:
            return None

        path = candidates[-1]
        label = _single_action_from_key_segment(path)
        return path, label

    def _pick_from_timestep(self, date_key: str, timestep: int) -> Optional[Tuple[str, int]]:
        if self.mode == "train":
            return self._pick_train_from_timestep(date_key, timestep)
        return self._pick_val_from_timestep(date_key, timestep)

    def _fallback_prev(self, date_key: str, target_timestep: int) -> Optional[Tuple[str, int]]:
        ts_list = self.ts_by_date.get(date_key, [])
        if not ts_list:
            return None

        pos = bisect_left(ts_list, target_timestep) - 1
        if pos < 0:
            return None

        fallback_timestep = ts_list[pos]
        return self._pick_from_timestep(date_key, fallback_timestep)

    def _placeholder_past_label(self) -> int:
        """
        Use a stage-specific placeholder label when no valid past frame exists.
        This avoids leaking the current target into history.
        """
        if self.stage == 1:
            return ACTION_KEY_TO_INDEX["i"]
        return ACTION_KEY_TO_INDEX["9"]

    def _build_sequence(self, row: Dict) -> Tuple[torch.Tensor, torch.Tensor]:
        date_key = row["date"]
        current_timestep = row["t"]
        current_path = row["path"]

        images = []
        past_labels = []

        for dt in range(self.past_k, 0, -1):
            target_timestep = current_timestep - dt
            selected = self._pick_from_timestep(date_key, target_timestep)

            if selected is None:
                selected = self._fallback_prev(date_key, target_timestep)

            if selected is None:
                selected_path = current_path
                selected_label = self._placeholder_past_label()
            else:
                selected_path, selected_label = selected

            images.append(self._load_image(selected_path))
            past_labels.append(selected_label)

        images.append(self._load_image(current_path))

        img_seq = torch.stack(images, dim=0)
        past_actions = torch.tensor(past_labels, dtype=torch.long)
        return img_seq, past_actions

    def __getitem__(self, idx: int):
        ridx = self.samples[idx]
        row = self.rows[ridx]

        img_seq, past_actions = self._build_sequence(row)

        if self.mode == "train":
            target = torch.tensor(self.targets[idx], dtype=torch.long)
        else:
            target = torch.tensor(self.targets[idx], dtype=torch.float32)

        return img_seq, past_actions, target
    
    



# -----------------------------------------------------------------------------
# Dataset for corrective gate
# -----------------------------------------------------------------------------

class SeqDataset_Corrective_Gate(Dataset):
    """
    Sequence dataset for binary correction-flag training and evaluation.

    Each sample contains:
      - `seq_len` frames: `past_k` history frames + 1 anchor frame
      - `past_k` past single-label actions
      - a binary target

    Train mode:
      - returns a scalar label in {0, 1}

    Validation mode:
      - returns a 2D multi-hot mask:
          [1, 0] -> only class 0 is valid
          [0, 1] -> only class 1 is valid
          [1, 1] -> both classes are valid
    """

    def __init__(
        self,
        csv_path: str,
        transform,
        stage: int = 1,
        seq_len: int = 5,
        past_k: int = 4,
        mode: str = "train",
        crop_box: Tuple[int, int, int, int] = (520, 163, 1144, 729),
        verbose: bool = True,
    ):
        if seq_len != past_k + 1:
            raise ValueError("Expected seq_len == past_k + 1")
        if mode not in ("train", "val"):
            raise ValueError("mode must be either 'train' or 'val'")
        if stage not in (1, 2):
            raise ValueError("Only stage 1 and stage 2 are supported")

        self.csv_path = csv_path
        self.transform = transform
        self.stage = stage
        self.seq_len = seq_len
        self.past_k = past_k
        self.mode = mode
        self.crop_box = crop_box

        self.positive_keys = (
            set(POSITIVE_ACTION_KEYS_STAGE1)
            if stage == 1
            else set(POSITIVE_ACTION_KEYS_STAGE2)
        )

        df = pd.read_csv(csv_path)
        if "stage" not in df.columns or "path" not in df.columns or "action_key" not in df.columns:
            raise ValueError("CSV must contain at least: ['stage', 'path', 'action_key']")

        df = df[df["stage"] == stage].reset_index(drop=True)

        self.rows = self._build_rows(df)
        self.by_dt_idx, self.by_dt_paths, self.ts_by_date = self._build_indices(self.rows)

        if self.mode == "val":
            for key in self.by_dt_paths:
                self.by_dt_paths[key].sort(key=_frame_idx_from_path)

        self.samples = list(range(len(self.rows)))
        self.targets = self._build_targets()

        if self.mode == "train":
            self.class_counts = np.bincount(np.asarray(self.targets, dtype=np.int64), minlength=2)
        else:
            self.class_counts = None

        if verbose:
            print(
                f"[{self.mode} stage={self.stage}] "
                f"Total rows: {len(self.rows)} | Valid anchor samples: {len(self.samples)}"
            )

    def __len__(self) -> int:
        return len(self.samples)

    def _build_rows(self, df: pd.DataFrame) -> List[Dict]:
        rows = []

        for _, row in df.iterrows():
            path = str(row["path"])
            date_key = _date_folder_from_path(path)
            timestep = _timestep_from_path(path)

            if timestep is None:
                continue

            action_str = str(row["action_key"])
            first_action = _first_action_idx(action_str)
            multi_action = _parse_action_multi(action_str)

            rows.append(
                {
                    "path": path,
                    "date": date_key,
                    "t": timestep,
                    "a_first": first_action,
                    "a_multi": multi_action,
                }
            )

        return rows

    def _build_indices(
        self,
        rows: List[Dict],
    ) -> Tuple[
        Dict[Tuple[str, int], List[int]],
        Dict[Tuple[str, int], List[str]],
        Dict[str, List[int]],
    ]:
        by_dt_idx = defaultdict(list)
        by_dt_paths = defaultdict(list)
        ts_by_date = defaultdict(set)

        for ridx, row in enumerate(rows):
            key = (row["date"], row["t"])
            by_dt_idx[key].append(ridx)
            by_dt_paths[key].append(row["path"])
            ts_by_date[row["date"]].add(row["t"])

        ts_by_date = {date: sorted(list(ts)) for date, ts in ts_by_date.items()}
        return by_dt_idx, by_dt_paths, ts_by_date

    def _build_targets(self) -> List:
        if self.mode == "train":
            return [self._build_train_target(self.rows[ridx]["a_first"]) for ridx in self.samples]
        return [self._build_val_target(self.rows[ridx]["a_multi"]) for ridx in self.samples]

    def _build_train_target(self, action_idx: Optional[int]) -> int:
        if action_idx is None:
            action_char = "x"
        else:
            action_char = ACTION_INDEX_TO_KEY[action_idx]
        return int(action_char in self.positive_keys)

    def _build_val_target(self, action_indices: List[int]) -> List[float]:
        if not action_indices:
            return [1.0, 0.0]

        found_positive = False
        found_negative = False

        for action_idx in action_indices:
            action_char = ACTION_INDEX_TO_KEY[action_idx]
            if action_char in self.positive_keys:
                found_positive = True
            else:
                found_negative = True

            if found_positive and found_negative:
                break

        if found_positive and not found_negative:
            return [0.0, 1.0]
        if not found_positive and found_negative:
            return [1.0, 0.0]
        return [1.0, 1.0]

    def _load_image(self, path: str) -> torch.Tensor:
        image = Image.open(path).convert("RGB")
        image = image.crop(self.crop_box)
        return self.transform(image)

    def _pick_train_from_timestep(self, date_key: str, timestep: int) -> Optional[Tuple[str, int]]:
        candidates = self.by_dt_idx.get((date_key, timestep), [])
        if not candidates:
            return None

        ridx = random.choice(candidates)
        path = self.rows[ridx]["path"]
        label = _single_action_from_key_segment(path)
        return path, label

    def _pick_val_from_timestep(self, date_key: str, timestep: int) -> Optional[Tuple[str, int]]:
        candidates = self.by_dt_paths.get((date_key, timestep), [])
        if not candidates:
            return None

        path = candidates[-1]
        label = _single_action_from_key_segment(path)
        return path, label

    def _pick_from_timestep(self, date_key: str, timestep: int) -> Optional[Tuple[str, int]]:
        if self.mode == "train":
            return self._pick_train_from_timestep(date_key, timestep)
        return self._pick_val_from_timestep(date_key, timestep)

    def _fallback_prev(self, date_key: str, target_timestep: int) -> Optional[Tuple[str, int]]:
        ts_list = self.ts_by_date.get(date_key, [])
        if not ts_list:
            return None

        pos = bisect_left(ts_list, target_timestep) - 1
        if pos < 0:
            return None

        fallback_timestep = ts_list[pos]
        return self._pick_from_timestep(date_key, fallback_timestep)

    def _placeholder_past_label(self) -> int:
        """
        Use a stage-specific placeholder label when no valid past frame exists.
        This avoids leaking the current target into the history.
        """
        if self.stage == 1:
            return ACTION_KEY_TO_INDEX["i"]
        return ACTION_KEY_TO_INDEX["9"]

    def _build_sequence(self, row: Dict) -> Tuple[torch.Tensor, torch.Tensor]:
        date_key = row["date"]
        current_timestep = row["t"]
        current_path = row["path"]

        images = []
        past_labels = []

        for dt in range(self.past_k, 0, -1):
            target_timestep = current_timestep - dt
            selected = self._pick_from_timestep(date_key, target_timestep)

            if selected is None:
                selected = self._fallback_prev(date_key, target_timestep)

            if selected is None:
                selected_path = current_path
                selected_label = self._placeholder_past_label()
            else:
                selected_path, selected_label = selected

            images.append(self._load_image(selected_path))
            past_labels.append(selected_label)

        images.append(self._load_image(current_path))

        img_seq = torch.stack(images, dim=0)
        past_actions = torch.tensor(past_labels, dtype=torch.long)
        return img_seq, past_actions

    def __getitem__(self, idx: int):
        ridx = self.samples[idx]
        row = self.rows[ridx]

        img_seq, past_actions = self._build_sequence(row)

        if self.mode == "train":
            target = torch.tensor(self.targets[idx], dtype=torch.long)
        else:
            target = torch.tensor(self.targets[idx], dtype=torch.float32)

        return img_seq, past_actions, target