"""
One entry point for every transformer variant
"""
import argparse
import importlib
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for p in (ROOT, HERE, HERE / "multiview", HERE / "singleview"):
    sys.path.insert(0, str(p))

# Three orthogonal axes -- architecture, what runs on top of the CNN, and the
# label problem -- plus the split. Everything is dispatched to the scripts that
# already exist rather than reimplemented here, so a run through this file and
# a run of the underlying script are the same code path.
MODULES = {
    ("multiview", "ann"):        "train_multiview_transformer",
    ("multiview", "cnntosnn"):   "mt_cnntosnn",
    ("multiview", "spikformer"): "mt_spikformer",
    ("singleview", "ann"):        "mt_singleview",
    ("singleview", "cnntosnn"):   "sv_cnntosnn",
    ("singleview", "spikformer"): "sv_spikformer",
}

MODEL_BLURB = {
    "ann": "conventional network throughout",
    "cnntosnn": "CNNs converted to spiking after training, ANN attention above",
    "spikformer": "converted CNNs frozen, Spikformer attention trained on top",
}

SPLIT_BLURB = {
    "run-holdout": "whole runs held out per subject (subject-dependent)",
    "subject-independent": "whole subjects held out, 15%/15% by fraction",
    "subject-count": "train on --train-subjects subjects, rest is test",
}


def print_matrix():
    print("Everything this runner can launch:\n")
    print(f"  {'--arch':<12} {'--model':<12} script")
    print(f"  {'-' * 12} {'-' * 12} {'-' * 34}")
    for (arch, model), module in MODULES.items():
        folder = "multiview" if arch == "multiview" else "singleview"
        print(f"  {arch:<12} {model:<12} {folder}/{module}.py")
    print("\n  --task full        5-class: left/right fist, both fists, both feet, baseline")
    print("  --task leftright   binary: left vs right imagined fist")
    print("  --task task-only   4-class: left/right fist, both fists, both feet "
          "(baseline excluded from the answer choices, kept as context)")
    print()
    for name, blurb in SPLIT_BLURB.items():
        print(f"  --split {name:<20} {blurb}")
    print("\nAnything this runner does not recognise is passed straight through to")
    print("the underlying script, so its own flags (--epochs, --f1, --timesteps,")
    print("--pretrain-epochs, ...) all work unchanged. To see a given script's")
    print("own flags, add --target-help:")
    print("  python transformer/run.py --arch singleview --model spikformer "
          "--target-help")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Unrecognised flags are forwarded to the selected script. "
               "Use --list to see the matrix, --dry-run to resolve without running.")
    ap.add_argument("--arch", choices=["multiview", "singleview"], default="multiview",
                    help="multiview: five per-region CNNs + fusion attention. "
                         "singleview: every channel through ONE encoder, which is the "
                         "ablation of the multi-view architecture (default: multiview)")
    ap.add_argument("--model", choices=["ann", "cnntosnn", "spikformer"], default="ann",
                    help="ann: no spikes. cnntosnn: post-hoc conversion of the CNNs, "
                         "ANN attention kept. spikformer: converted CNNs frozen and "
                         "spiking attention trained on top (default: ann)")
    ap.add_argument("--task", choices=["full", "leftright", "task-only"], default="full",
                    help="full: 5-class. leftright: left vs right imagined fist. "
                         "task-only: 4-class, baseline excluded from the answer choices. "
                         "Both narrowed tasks keep their excluded chunks as unlabelled "
                         "context rather than dropping them (default: full)")
    ap.add_argument("--split", choices=list(SPLIT_BLURB), default="run-holdout",
                    help="run-holdout / subject-independent are the scripts' own "
                         "--split-mode values. subject-count is the train-on-N-subjects "
                         "split (default: run-holdout)")
    ap.add_argument("--train-subjects", type=int, default=90,
                    help="--split subject-count only: subjects used for training")
    ap.add_argument("--val-subjects", type=int, default=0,
                    help="--split subject-count only: 0 means validation comes from "
                         "held-out RUNS of the training subjects, so every remaining "
                         "subject is test. >0 takes that many of the remainder as a "
                         "subject-independent validation set, which is the sounder "
                         "choice for model selection")
    ap.add_argument("--list", action="store_true", help="Print the matrix and exit")
    ap.add_argument("--target-help", action="store_true",
                    help="Show the SELECTED script's own flags and exit (plain --help "
                         "documents this runner)")
    ap.add_argument("--dry-run", action="store_true",
                    help="Show what would run, including forwarded flags, then exit")
    args, forwarded = ap.parse_known_args()

    if args.list:
        print_matrix()
        return

    if any(a == "--split-mode" or a.startswith("--split-mode=") for a in forwarded):
        raise SystemExit(
            "Use --split rather than --split-mode here: the runner owns the split so "
            "it can also offer subject-count, which the scripts have no flag for.")

    module_name = MODULES[(args.arch, args.model)]
    folder = "multiview" if args.arch == "multiview" else "singleview"

    task_spec = None
    if args.task == "leftright":
        from leftright import LEFT_RIGHT
        task_spec = LEFT_RIGHT
    elif args.task == "task-only":
        from no_baseline import TASK_ONLY
        task_spec = TASK_ONLY

    split_spec = None
    if args.split == "subject-count":
        from splits import SubjectCountSplit
        split_spec = SubjectCountSplit(args.train_subjects, args.val_subjects)
    else:
        # The two original modes are the scripts' own flag, so just pass it on.
        forwarded = forwarded + ["--split-mode", args.split]

    print("=" * 72)
    print(f"  {folder}/{module_name}.py")
    print(f"  arch  : {args.arch}")
    print(f"  model : {args.model:<12} {MODEL_BLURB[args.model]}")
    print(f"  task  : {args.task:<12} "
          f"{task_spec.name if task_spec else '5-class'}")
    print(f"  split : {args.split:<12} {SPLIT_BLURB[args.split]}")
    if split_spec is not None:
        print(f"  {'':<8}train={args.train_subjects} subjects, "
              f"val={args.val_subjects or 'held-out runs of training subjects'}")
    if forwarded:
        print(f"  args  : {' '.join(forwarded)}")
    print("=" * 72)

    if args.dry_run:
        print("(--dry-run: stopping here)")
        return

    if args.target_help:
        forwarded = ["--help"]

    module = importlib.import_module(module_name)
    # The selected script parses these itself, so an unknown flag fails with
    # that script's own usage message rather than something invented here.
    sys.argv = [f"{module_name}.py"] + forwarded
    module.main(task_spec=task_spec, split_spec=split_spec)


if __name__ == "__main__":
    main()
