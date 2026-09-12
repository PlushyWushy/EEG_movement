"""
Subject-count split, shared by sv_subject_holdout and the master runner
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "transformer" / "multiview"))

from train_multiview_transformer import RUN_GROUPS


def subject_count_split(sids, runs, n_train, n_val_subjects=0, seed=42):
    """Subject-independent split by COUNT, not fraction.

    n_train subjects go to training and every remaining subject is test --
    whole subjects, so nothing about a test subject was ever seen. Where
    validation comes from is the real choice here, and the two options are
    not equivalent:

    * n_val_subjects = 0 (default): validation is one held-out RUN from each
      training subject, so every remaining subject goes to test. This is the
      literal 'train on N, test on the rest' split, but be aware the val set
      is then subject-DEPENDENT while the test set is subject-independent --
      early stopping is selecting on within-subject generalisation and asked
      to predict across-subject generalisation, which it does badly.
    * n_val_subjects > 0: that many of the remaining subjects become a
      subject-independent validation set, and the rest are test. Fewer test
      subjects, but model selection now tracks the quantity being reported.

    Which run is held out for validation alternates by task group across
    subjects, so validation always contains every class.
    """
    rng = np.random.default_rng(seed)
    subjects = sorted(set(sids.tolist()))
    if n_train >= len(subjects):
        raise SystemExit(
            f"--train-subjects {n_train} leaves nothing to test on: only "
            f"{len(subjects)} subjects are in this cache. Lower it, or raise "
            f"--max-subjects.")
    if n_train + n_val_subjects >= len(subjects):
        raise SystemExit(
            f"--train-subjects {n_train} + --val-subjects {n_val_subjects} "
            f"leaves no test subjects ({len(subjects)} available).")

    order = [int(s) for s in rng.permutation(subjects)]
    train_s = order[:n_train]
    rest = order[n_train:]
    val_s = rest[:n_val_subjects]
    test_s = rest[n_val_subjects:]

    by_subject = {}
    for i, s in enumerate(sids.tolist()):
        by_subject.setdefault(s, []).append(i)

    train, val = [], []
    if n_val_subjects > 0:
        for s in train_s:
            train.extend(by_subject[s])
        for s in val_s:
            val.extend(by_subject[s])
    else:
        groups = list(RUN_GROUPS.values())
        for k, s in enumerate(train_s):
            subj = by_subject[s]
            cand = [i for i in subj if runs[i] in groups[k % len(groups)]]
            pick = int(rng.choice(cand if cand else subj))
            val.append(pick)
            train.extend(i for i in subj if i != pick)

    test = [i for s in test_s for i in by_subject[s]]

    print(f"Subject-count split: {len(train_s)} train / "
          f"{len(val_s) if n_val_subjects else 0} val / {len(test_s)} test subjects "
          f"({len(train)} / {len(val)} / {len(test)} runs)")
    if n_val_subjects == 0:
        print("  validation is one held-out run per TRAINING subject, so all "
              "remaining subjects are test -- see subject_count_split for why "
              "that makes early stopping a biased selector")
    print(f"  test subjects: {sorted(test_s)}")
    return np.array(train), np.array(val), np.array(test), sorted(test_s)


def per_subject_report(preds, trues, chunk_sids, classes):
    """Subject-independent accuracy is an average over very unequal subjects --
    some people's motor imagery simply does not decode. The spread matters as
    much as the mean, so print it rather than hide it behind one number."""
    rows = []
    for s in sorted(set(chunk_sids.tolist())):
        m = chunk_sids == s
        has_baseline = "baseline" in classes
        task = m & (trues != classes.index("baseline")) if has_baseline else m
        rows.append((s, int(m.sum()), (preds[m] == trues[m]).mean(),
                     (preds[task] == trues[task]).mean() if task.any() else float("nan")))
    rows.sort(key=lambda r: r[2])

    print("\nPer-subject test accuracy (worst first):")
    print(f"  {'subject':>8} {'chunks':>7} {'acc':>7} {'task-only':>10}")
    for s, n, acc, task_acc in rows:
        print(f"  S{s:03d}     {n:>7} {acc:>7.3f} {task_acc:>10.3f}")
    accs = np.array([r[2] for r in rows])
    print(f"  mean over subjects={accs.mean():.4f}  sd={accs.std():.4f}  "
          f"min={accs.min():.4f}  max={accs.max():.4f}")


class SubjectCountSplit:
    """A split spec: what a parent script's main() calls instead of its own
    run-holdout / subject-independent branch.

    Carries the two knobs the count-based split needs, which the parent
    scripts have no flags for -- the master runner owns them and hands over a
    configured spec, so none of the six scripts had to grow another argument.
    """

    name = "subject-count"

    def __init__(self, n_train, n_val_subjects=0):
        self.n_train = n_train
        self.n_val_subjects = n_val_subjects
        self.test_subjects = None

    def __call__(self, sids, runs, seed):
        tr, va, te, subs = subject_count_split(
            sids, runs, self.n_train, self.n_val_subjects, seed=seed)
        self.test_subjects = subs
        return tr, va, te
