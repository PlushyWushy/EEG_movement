"""
Left vs right imagined fist: the task spec every -leftright variant passes in
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "transformer" / "multiview"))

from train_multiview_transformer import LABEL_MIRROR, RUN_GROUPS
from mlp.train_mlp import CLASS_TO_IDX, CLASSES

LR_CLASSES = ["left_fist", "right_fist"]


class Task:
    """A narrowing of the label problem, handed to a parent script's main().

    Kept deliberately small: a name for the log, the class list everything
    downstream reports against, and a filter applied to the built sequences.
    Parent scripts default to task_spec=None and behave exactly as before, so
    adding this changed none of their existing results."""

    def __init__(self, name, classes, filter_fn):
        self.name = name
        self.classes = classes
        self.filter = filter_fn


def filter_to_left_right(X, y, lengths, sids, runs):
    """Narrow the 5-class run sequences to imagined left vs right fist.

    Worth being explicit that IMAGERY is not a new restriction: this repo only
    ever epochs runs 4/8/12 and 6/10/14, which are the motor-imagery runs.
    The executed-movement runs (3/7/11, 5/9/13) are never loaded at all, so
    every result in this project is already imagined movement.

    What this does change:

    * Runs 6/10/14 are dropped. They are the imagined fists/feet runs and
      contain no left or right chunk whatsoever, so keeping them would add
      compute and no supervision.
    * Baseline chunks inside the kept runs are relabelled -1 rather than
      removed. -1 is already the ignore_index used for padding, so they cost
      nothing in the loss and nothing in the metrics -- but they are still
      encoded, and still visible to the causal context transformer as history.
      Deleting them instead would quietly redefine what "the previous chunk"
      means: the run's real temporal structure would collapse and the context
      attention would be reading a sequence that never happened.
    * Surviving labels are remapped to 0/1 so the head is genuinely binary. A
      5-way head scored on a 2-way problem can spend probability on classes
      that cannot occur, which only ever costs accuracy.
    """
    keep_runs = np.isin(runs, RUN_GROUPS["lr"])
    X, y = X[keep_runs], y[keep_runs]
    lengths, sids, runs = lengths[keep_runs], sids[keep_runs], runs[keep_runs]

    remap = np.full(len(CLASSES), -1, dtype=np.int64)
    for new_idx, name in enumerate(LR_CLASSES):
        remap[CLASS_TO_IDX[name]] = new_idx
    y = np.where(y >= 0, remap[np.clip(y, 0, None)], -1)

    # RunSequenceDataset's hemisphere mirror indexes LABEL_MIRROR with whatever
    # labels it finds, which are now the remapped ones. That stays correct only
    # while left/right occupy indices 0/1 and the mirror swaps them -- which is
    # exactly the augmentation that matters most for this task, so assert it
    # rather than let it rot silently.
    assert LABEL_MIRROR[0] == 1 and LABEL_MIRROR[1] == 0, \
        "mirror augmentation no longer swaps labels 0/1; fix filter_to_left_right"

    print(f"Left/right filter: kept {len(runs)} runs (imagery runs "
          f"{sorted(RUN_GROUPS['lr'])}), {int((y >= 0).sum())} labelled chunks "
          f"out of {int(lengths.sum())}; the rest stay as unlabelled context")
    return X, y, lengths, sids, runs


LEFT_RIGHT = Task("left vs right imagined fist", LR_CLASSES, filter_to_left_right)
