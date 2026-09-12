"""
Singleview ablation, subject-independent: train on N subjects, test on the rest
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "transformer"))
sys.path.insert(0, str(ROOT / "transformer" / "multiview"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from mt_singleview import SingleViewRunModel, pretrain_single, view_channel_index
from splits import per_subject_report, subject_count_split
from train_multiview_transformer import (
    CACHE_DIR,
    CANONICAL_CHANNELS,
    RUN_GROUPS,
    VIEW_NAMES,
    MultiViewRunModel,
    RunSequenceDataset,
    build_sequences,
    build_smote_neighbors,
    make_pad_mask,
    run_epoch,
)
from mlp.train_mlp import CLASS_TO_IDX, CLASSES


def main(task_spec=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train-subjects", type=int, default=90,
                    help="Subjects used for training. Every subject not used for "
                         "training or validation is test")
    ap.add_argument("--val-subjects", type=int, default=0,
                    help="0 (default): validation is one held-out run per training "
                         "subject, so ALL remaining subjects are test. >0: that many "
                         "of the remaining subjects form a subject-independent "
                         "validation set instead -- recommended, since it makes early "
                         "stopping select on the thing being measured")
    ap.add_argument("--view", choices=["all"] + VIEW_NAMES, default="all",
                    help="'all' (default) ablates the multi-view architecture")
    ap.add_argument("--max-subjects", type=int, default=None)
    ap.add_argument("--no-cache", action="store_true")
    # model
    ap.add_argument("--d-model", type=int, default=128)
    ap.add_argument("--ctx-layers", type=int, default=2)
    ap.add_argument("--ctx-heads", type=int, default=4)
    ap.add_argument("--f1", type=int, default=8)
    ap.add_argument("--depth", type=int, default=2)
    ap.add_argument("--dropout", type=float, default=0.3)
    # context knobs
    ap.add_argument("--context-weight", type=float, default=0.1)
    ap.add_argument("--freeze-context-weight", action="store_true")
    ap.add_argument("--context-len", type=int, default=0)
    # augmentation
    ap.add_argument("--no-smote", action="store_true")
    ap.add_argument("--smote-prob", type=float, default=0.3)
    ap.add_argument("--smote-k", type=int, default=5)
    ap.add_argument("--smote-lambda-max", type=float, default=1.0)
    ap.add_argument("--no-augment", action="store_true")
    ap.add_argument("--mirror-prob", type=float, default=0.5)
    ap.add_argument("--channel-drop", type=float, default=0.1)
    ap.add_argument("--scale-jitter", type=float, default=0.1)
    ap.add_argument("--time-mask", type=float, default=0.1)
    ap.add_argument("--noise-std", type=float, default=0.1)
    # pretraining
    ap.add_argument("--pretrain-epochs", type=int, default=0)
    ap.add_argument("--pretrain-mask-frac", type=float, default=0.15)
    ap.add_argument("--pretrain-lr", type=float, default=1e-3)
    # optimization
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--no-class-weights", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    suffix = "all" if args.max_subjects is None else str(args.max_subjects)
    X, y, lengths, sids, runs = build_sequences(
        CACHE_DIR / f"mv_seq_raw_{suffix}.npz", args.max_subjects,
        use_cache=not args.no_cache)
    # Task hook. A variant script passes a spec that narrows the label problem
    # (see transformer/leftright.py); None keeps the full 5-class task, so this
    # script's own behaviour is unchanged.
    classes = CLASSES
    if task_spec is not None:
        X, y, lengths, sids, runs = task_spec.filter(X, y, lengths, sids, runs)
        classes = task_spec.classes
        print(f"\nTask: {task_spec.name}  ->  classes={classes}")
    print(f"Sequences: X={X.shape}, subjects={len(set(sids.tolist()))}")
    print("Chunk counts:", {c: int((y == i).sum()) for i, c in enumerate(classes)})

    idx = view_channel_index(args.view)
    if args.view == "all":
        print(f"\nABLATION: all {len(idx)} channels through ONE encoder -- no view "
              f"embeddings, no fusion attention.")
    else:
        print(f"\nABLATION: view '{args.view}' only -- {len(idx)} of "
              f"{len(CANONICAL_CHANNELS)} channels.")

    tr_idx, va_idx, te_idx, test_subjects = subject_count_split(
        sids, runs, args.train_subjects, args.val_subjects, seed=args.seed)

    neighbors = None
    if not args.no_smote and args.smote_prob > 0:
        t0 = time.time()
        neighbors = build_smote_neighbors(X, y, lengths, tr_idx, sids, k=args.smote_k)
        print(f"SMOTE: neighbour index built in {time.time()-t0:.0f}s")

    aug = not args.no_augment
    if aug:
        print(f"Augmentation (train only): mirror={args.mirror_prob} "
              f"channel_drop={args.channel_drop} scale_jitter={args.scale_jitter} "
              f"time_mask={args.time_mask} noise_std={args.noise_std}")

    def loader(sel, shuffle, augment=False):
        ds = RunSequenceDataset(
            X, y, lengths, sel,
            neighbors=neighbors if augment else None,
            smote_prob=args.smote_prob if augment else 0.0,
            lam_max=args.smote_lambda_max,
            mirror_prob=args.mirror_prob if (augment and aug) else 0.0,
            channel_drop=args.channel_drop if (augment and aug) else 0.0,
            scale_jitter=args.scale_jitter if (augment and aug) else 0.0,
            time_mask_frac=args.time_mask if (augment and aug) else 0.0,
            noise_std=args.noise_std if (augment and aug) else 0.0,
            seed=args.seed)
        return torch.utils.data.DataLoader(ds, batch_size=args.batch_size,
                                           shuffle=shuffle)

    train_loader, val_loader, test_loader = (loader(tr_idx, True, augment=True),
                                             loader(va_idx, False),
                                             loader(te_idx, False))
    print(f"Chunks: train={int(lengths[tr_idx].sum())} val={int(lengths[va_idx].sum())} "
          f"test={int(lengths[te_idx].sum())}")

    device = torch.device("cuda" if torch.cuda.is_available()
                          else "mps" if torch.backends.mps.is_available()
                          else "cpu")
    print(f"Using device: {device}")

    common = dict(n_classes=len(classes), d_model=args.d_model, ctx_layers=args.ctx_layers,
                  ctx_heads=args.ctx_heads, dropout=args.dropout, max_len=X.shape[1],
                  context_weight=args.context_weight,
                  freeze_context=args.freeze_context_weight,
                  context_len=args.context_len, f1=args.f1, depth=args.depth)
    model = SingleViewRunModel(idx, **common).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    reference = sum(p.numel() for p in MultiViewRunModel(**common).parameters())
    print(f"Model parameters: {n_params:,}")
    print(f"  multi-view parent at these settings: {reference:,} "
          f"({n_params / reference:.2f}x)")

    if args.pretrain_epochs > 0:
        print(f"\nCausal masked-chunk pre-training on {len(tr_idx)} training runs "
              f"(mask_frac={args.pretrain_mask_frac}):")
        pretrain_single(model, train_loader, device, args.pretrain_epochs,
                        args.pretrain_mask_frac, args.pretrain_lr, args.d_model, idx)

    if args.no_class_weights:
        weight = None
    else:
        freq = np.array([max((y[tr_idx] == i).sum(), 1) for i in range(len(classes))],
                        dtype=np.float64)
        weight = torch.tensor((freq.sum() / (len(classes) * freq)),
                              dtype=torch.float32, device=device)
        print("Class weights:", {c: round(float(w), 3) for c, w in zip(classes, weight)})

    criterion = nn.CrossEntropyLoss(weight=weight, ignore_index=-1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)

    best_val_loss, best_state, stale = float("inf"), None, 0
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc = run_epoch(model, train_loader, criterion, optimizer, device, True)
        va_loss, va_acc = run_epoch(model, val_loader, criterion, optimizer, device, False)
        print(f"epoch {epoch:3d}  train_loss={tr_loss:.4f} train_acc={tr_acc:.4f}  "
              f"val_loss={va_loss:.4f} val_acc={va_acc:.4f}  "
              f"gate={model.context_gate.item():.3f}  ({time.time()-t0:.0f}s)", flush=True)
        if va_loss < best_val_loss - 1e-4:
            best_val_loss, stale = va_loss, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
            if stale >= args.patience:
                print(f"Early stopping at epoch {epoch}")
                break
    if best_state is not None:
        model.load_state_dict(best_state)

    test_loss, test_acc = run_epoch(model, test_loader, criterion, optimizer, device, False)
    print(f"\nTest loss={test_loss:.4f}  Test accuracy={test_acc:.4f}  "
          f"({len(test_subjects)} unseen subjects)")
    print(f"Learned context gate: {model.context_gate.item():.4f}")

    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for xb, yb, lb in test_loader:
            xb, lb = xb.to(device), lb.to(device)
            logits = model(xb, make_pad_mask(lb, xb.shape[1], device))
            valid = yb.reshape(-1) != -1
            preds.append(logits.reshape(-1, logits.shape[-1]).argmax(1).cpu()[valid].numpy())
            trues.append(yb.reshape(-1)[valid].numpy())
    preds, trues = np.concatenate(preds), np.concatenate(trues)

    # test_loader is unshuffled, so chunks arrive in te_idx order and this
    # lines up one-to-one with preds/trues. Count LABELLED chunks per run, not
    # lengths[i]: a task spec can leave chunks inside a run unlabelled (-1) as
    # context, and those are dropped from preds/trues by the valid mask.
    chunk_sids = np.concatenate([np.full(int((y[i] >= 0).sum()), sids[i])
                                 for i in te_idx])
    assert len(chunk_sids) == len(trues), "chunk/subject alignment broke"

    print(f"\nClassification report (test set, {len(test_subjects)} unseen subjects):")
    print(classification_report(trues, preds, labels=list(range(len(classes))),
                                target_names=classes, digits=3, zero_division=0))
    task = (trues != classes.index("baseline")) if "baseline" in classes else np.zeros(0, bool)
    if task.any():
        print(f"Task-only accuracy (baseline chunks excluded): "
              f"{(preds[task] == trues[task]).mean():.4f}")
    print("Confusion matrix (rows=true, cols=pred):")
    print(classes)
    print(confusion_matrix(trues, preds, labels=list(range(len(classes)))))

    per_subject_report(preds, trues, chunk_sids, classes)


if __name__ == "__main__":
    main()
