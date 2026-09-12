"""
Four-class task: left/right imagined fist, both fists, both feet -- baseline
excluded from the answer choices, passed to transformer/run.py via --task task-only
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from leftright import Task
from mlp.train_mlp import CLASS_TO_IDX, CLASSES

TASK_ONLY_CLASSES = [c for c in CLASSES if c != "baseline"]


def filter_no_baseline(X, y, lengths, sids, runs):
    """Narrow the 5-class run sequences to the four task classes.

    Simpler than filter_to_left_right in two ways:

    * No runs are dropped. The lr runs (4/8/12) and ff runs (6/10/14) between
      them already cover all four non-baseline classes, so every run stays.
    * No relabelling is needed either. Baseline sits last in CLASSES, so the
      other four classes already occupy indices 0-3 -- only baseline's own
      chunks (index 4) need to become -1, the same ignore_index used for
      padding. They cost nothing in the loss or the metrics, but are still
      encoded and still visible to the causal context transformer as
      history, exactly like the leftright task keeps its dropped baseline
      chunks. And because left_fist/right_fist keep their original indices
      0/1, RunSequenceDataset's LABEL_MIRROR swap needs no change either.
    """
    baseline_idx = CLASS_TO_IDX["baseline"]
    y = np.where(y == baseline_idx, -1, y)
    print(f"No-baseline filter: kept all {len(runs)} runs, "
          f"{int((y >= 0).sum())} labelled chunks out of {int(lengths.sum())}; "
          f"baseline chunks stay as unlabelled context")
    return X, y, lengths, sids, runs


TASK_ONLY = Task("4-class (baseline excluded)", TASK_ONLY_CLASSES, filter_no_baseline)
