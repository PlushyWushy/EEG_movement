"""
Multiview ablation: one view only
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import classification_report, confusion_matrix

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "transformer" / "multiview"))

# Imported, not copied. The point of an ablation is that the control and the
# treatment share every line except the one under test, and a forked copy
# stops being comparable the moment anyone touches the parent's augmentation
# or splits. train_multiview_transformer.py is importable (the folder is
# hyphenated, the module isn't), so this file changes the chunk encoder and
# nothing else.
from train_multiview_transformer import (
    CACHE_DIR,
    CANONICAL_CHANNELS,
    VIEW_NAMES,
    VIEWS,
    MultiViewRunModel,
    RunSequenceDataset,
    ViewEncoder,
    build_sequences,
    build_smote_neighbors,
    make_pad_mask,
    run_epoch,
    run_holdout_split,
    save_checkpoint,
    subject_holdout_split,
)
from mlp.train_mlp import CLASS_TO_IDX, CLASSES, N_SAMPLES


def view_channel_index(view):
    """Canonical-order channel indices this ablation feeds its one encoder.

    'all' is the ablation of the multi-view ARCHITECTURE: every channel still
    goes in, but through a single encoder with no view embeddings and no
    fusion attention, so the only thing removed is the decomposition itself.
    Naming a single view instead ablates the DATA as well, which answers the
    different question of how much any one scalp region carries on its own --
    useful, but not a like-for-like test of multi-view.
    """
    chans = CANONICAL_CHANNELS if view == "all" else VIEWS[view]
    return np.array([CANONICAL_CHANNELS.index(c) for c in chans], dtype=np.int64)


VIEW_KEY = "single"


class SingleViewChunkEncoder(nn.Module):
    """Drop-in replacement for MultiViewChunkEncoder: one ViewEncoder over one
    channel set, and that is all. No per-view tokens, no learned view
    embeddings, no fusion attention layer -- removing those three is the
    ablation. The encoder itself is the parent's, unmodified, so the
    convolutional stack is held constant.

    The lone encoder still lives in a ModuleDict, keyed VIEW_KEY, purely so
    this duck-types as MultiViewChunkEncoder: the SNN conversion machinery
    iterates chunk_encoder.encoders to hook activations, and a one-entry dict
    lets all of it drive this class unmodified. `encoder_cls` is how the
    conversion scripts swap in their ReLU ViewEncoder without a fork.
    """

    def __init__(self, channel_idx, d_model, dropout=0.3, f1=8, depth=2,
                 encoder_cls=ViewEncoder):
        super().__init__()
        self.register_buffer("idx", torch.as_tensor(channel_idx), persistent=False)
        self.encoders = nn.ModuleDict({
            VIEW_KEY: encoder_cls(len(channel_idx), d_model, f1=f1, depth=depth,
                                  dropout=dropout)})

    def forward(self, x):                 # (N, 64, 640) -> (N, d_model)
        return self.encoders[VIEW_KEY](x.index_select(1, self.idx))


class SingleViewRunModel(MultiViewRunModel):
    """The parent model with its chunk encoder swapped out.

    Subclassed rather than reimplemented so that the positional embedding,
    the causal context transformer, the mask token, the context gate, the
    LayerNorm(local + gate * context) combination and the head are literally
    the parent's code -- forward(), encode_chunks() and apply_context() are
    all inherited untouched. The multi-view encoder super().__init__ builds is
    discarded immediately; that wastes a construction, and buys the guarantee
    that nothing below the chunk encoder drifted."""

    def __init__(self, channel_idx, d_model=128, dropout=0.3, f1=8, depth=2, **kw):
        super().__init__(d_model=d_model, dropout=dropout, f1=f1, depth=depth, **kw)
        self.chunk_encoder = SingleViewChunkEncoder(channel_idx, d_model,
                                                    dropout=dropout, f1=f1, depth=depth)


def load_model(path, device="cpu"):
    """Reconstruct a SingleViewRunModel saved via --save. Returns (model, classes)."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    idx = np.asarray(ckpt["channel_idx"], dtype=np.int64)
    model = SingleViewRunModel(idx, **ckpt["config"]).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, ckpt["classes"]


class SingleReconHead(nn.Module):
    """The pre-training decoder, collapsed from ReconHeads' five to one.

    Same shape per channel as the parent's per-view heads, so the
    reconstruction budget is matched rather than merely similar: the parent
    spends d_model x 160 x C_view on each of five views, which sums to exactly
    the d_model x 160 x 64 spent here when --view is 'all'."""

    def __init__(self, d_model, n_channels):
        super().__init__()
        self.c = n_channels
        self.lin = nn.Linear(d_model, n_channels * 4 * 40)
        self.up = nn.Sequential(
            nn.ConvTranspose1d(n_channels * 4, n_channels * 2, 4, stride=4), nn.ELU(),
            nn.ConvTranspose1d(n_channels * 2, n_channels, 4, stride=4),
        )

    def forward(self, h):                 # (N, d) -> (N, C, 640)
        return self.up(self.lin(h).reshape(-1, self.c * 4, 40))


def pretrain_single(model, loader, device, epochs, mask_frac, lr, d_model, idx):
    """train_multiview_transformer.pretrain with the per-view loop removed.

    Identical objective otherwise: replace a fraction of chunk embeddings with
    the mask token, reconstruct their raw signal from PRECEDING chunks only,
    never mask position 0. The mask is redrawn every batch, so this is the
    re-masking schedule rather than a fixed one."""
    recon = SingleReconHead(d_model, len(idx)).to(device)
    params = list(model.chunk_encoder.parameters()) + list(model.context.parameters()) \
        + [model.mask_token] + list(model.pos.parameters()) + list(recon.parameters())
    opt = torch.optim.Adam(params, lr=lr)
    idx = torch.as_tensor(idx, device=device)

    for ep in range(1, epochs + 1):
        model.train()
        total, n = 0.0, 0
        for xb, yb, lb in loader:
            xb, lb = xb.to(device), lb.to(device)
            B, T = xb.shape[:2]
            pad_mask = make_pad_mask(lb, T, device)
            maskable = (~pad_mask).clone()
            maskable[:, 0] = False
            sel = (torch.rand(B, T, device=device) < mask_frac) & maskable
            if not sel.any():
                continue

            local = model.encode_chunks(xb)
            h = model.apply_context(local, pad_mask, mask_positions=sel)
            target = xb.index_select(2, idx)[sel]
            loss = F.mse_loss(recon(h[sel]), target)

            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            total += loss.item() * int(sel.sum())
            n += int(sel.sum())
        print(f"  pretrain epoch {ep:3d}  recon_mse={total / max(n, 1):.4f}")
    return model


def main(task_spec=None, split_spec=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--view", choices=["all"] + VIEW_NAMES, default="all",
                    help="'all' (default): every channel through ONE encoder, which "
                         "ablates the multi-view architecture while holding the input "
                         "data fixed. A view name instead keeps one scalp region and "
                         "discards the other channels, which ablates the data too")
    ap.add_argument("--max-subjects", type=int, default=None)
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--split-mode", choices=["run-holdout", "subject-independent"],
                    default="run-holdout")
    # model
    ap.add_argument("--d-model", type=int, default=128)
    ap.add_argument("--ctx-layers", type=int, default=2)
    ap.add_argument("--ctx-heads", type=int, default=4)
    ap.add_argument("--f1", type=int, default=8, help="ViewEncoder temporal filters")
    ap.add_argument("--depth", type=int, default=2, help="ViewEncoder depth multiplier")
    ap.add_argument("--dropout", type=float, default=0.3)
    # context knobs
    ap.add_argument("--context-weight", type=float, default=0.1)
    ap.add_argument("--freeze-context-weight", action="store_true")
    ap.add_argument("--context-len", type=int, default=0)
    # SMOTE
    ap.add_argument("--no-smote", action="store_true")
    ap.add_argument("--smote-prob", type=float, default=0.3)
    ap.add_argument("--smote-k", type=int, default=5)
    ap.add_argument("--smote-lambda-max", type=float, default=1.0)
    # signal-level augmentation
    ap.add_argument("--no-augment", action="store_true")
    ap.add_argument("--mirror-prob", type=float, default=0.5,
                    help="Reflect a run across the midline, swapping left_fist and "
                         "right_fist. Note this stays meaningful under --view all but "
                         "is close to useless for a single lateral view, whose "
                         "mirror image lives in channels that view no longer sees")
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
    ap.add_argument("--batch-size", type=int, default=8, help="runs per batch")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--no-class-weights", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--save", nargs="?", const="", default=None, metavar="PATH",
                    help="Save the trained model to PATH after training (bare --save "
                         "picks checkpoints/<script>_<timestamp>.pt). Bundles the "
                         "constructor config and channel indices alongside the "
                         "weights so load_model() can reconstruct a working model.")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # The cache holds all 64 channels regardless of --view (the selection
    # happens inside the model), so this is the parent's cache file, shared.
    suffix = "all" if args.max_subjects is None else str(args.max_subjects)
    cache_path = CACHE_DIR / f"mv_seq_raw_{suffix}.npz"
    X, y, lengths, sids, runs = build_sequences(cache_path, args.max_subjects,
                                                use_cache=not args.no_cache)
    # Task hook. A variant script passes a spec that narrows the label problem
    # (see transformer/leftright.py); None keeps the full 5-class task, so this
    # script's own behaviour is unchanged.
    classes = CLASSES
    if task_spec is not None:
        X, y, lengths, sids, runs = task_spec.filter(X, y, lengths, sids, runs)
        classes = task_spec.classes
        print(f"\nTask: {task_spec.name}  ->  classes={classes}")
    print(f"Sequences: X={X.shape} (n_runs, T_max, channels, times), "
          f"subjects={len(set(sids.tolist()))}")
    print("Chunk counts:", {c: int((y == i).sum()) for i, c in enumerate(classes)})

    idx = view_channel_index(args.view)
    if args.view == "all":
        print(f"\nall {len(idx)} channels through ONE encoder -- no view "
              f"embeddings, no fusion attention. Same input as the parent, so the "
              f"only thing removed is the multi-view decomposition.")
    else:
        print(f"\nABLATION: view '{args.view}' only -- {len(idx)} of "
              f"{len(CANONICAL_CHANNELS)} channels ({', '.join(VIEWS[args.view])}). "
              f"The other {len(CANONICAL_CHANNELS) - len(idx)} channels are discarded, "
              f"so this ablates the data as well as the architecture.")

    # Split hook, same idea as the task hook: a variant or the master runner
    # can hand over its own splitter (see transformer/splits.py). None keeps
    # this script's own --split-mode behaviour.
    if split_spec is not None:
        tr_idx, va_idx, te_idx = split_spec(sids, runs, args.seed)
    elif args.split_mode == "run-holdout":
        tr_idx, va_idx, te_idx = run_holdout_split(sids, runs, seed=args.seed)
    else:
        tr_idx, va_idx, te_idx = subject_holdout_split(sids, seed=args.seed)

    neighbors = None
    if not args.no_smote and args.smote_prob > 0:
        t0 = time.time()
        neighbors = build_smote_neighbors(X, y, lengths, tr_idx, sids, k=args.smote_k)
        print(f"SMOTE: neighbour index built in {time.time()-t0:.0f}s "
              f"(prob={args.smote_prob}, lambda~U(0,{args.smote_lambda_max}))")

    aug = not args.no_augment
    if aug:
        print(f"Augmentation (train only): mirror={args.mirror_prob} "
              f"channel_drop={args.channel_drop} scale_jitter={args.scale_jitter} "
              f"time_mask={args.time_mask} noise_std={args.noise_std}")
        if args.view in ("sensorimotor_left", "sensorimotor_right") and args.mirror_prob > 0:
            print("  NOTE: mirroring a single lateral view swaps its labels but "
                  "reflects its signal into channels the encoder cannot see. "
                  "Consider --mirror-prob 0 for this view.")

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

    save_path = None
    if args.save is not None:
        save_path = args.save or str(ROOT / "checkpoints" /
            f"{Path(__file__).stem}_{time.strftime('%Y%m%d-%H%M%S')}.pt")

    # The capacity gap is the obvious confound in this ablation -- one encoder
    # and no fusion layer is a smaller model, so a drop could be either the
    # missing decomposition or the missing parameters. State the gap rather
    # than let it lurk; --f1 / --depth are there to close it.
    n_params = sum(p.numel() for p in model.parameters())
    reference = sum(p.numel() for p in MultiViewRunModel(**common).parameters())
    print(f"Model parameters: {n_params:,}")
    print(f"  multi-view parent at these settings: {reference:,} "
          f"({n_params / reference:.2f}x) -- raise --f1/--depth to match capacity")

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
        tr_loss, tr_acc = run_epoch(model, train_loader, criterion, optimizer, device, True)
        va_loss, va_acc = run_epoch(model, val_loader, criterion, optimizer, device, False)
        print(f"epoch {epoch:3d}  train_loss={tr_loss:.4f} train_acc={tr_acc:.4f}  "
              f"val_loss={va_loss:.4f} val_acc={va_acc:.4f}  "
              f"gate={model.context_gate.item():.3f}")
        if va_loss < best_val_loss - 1e-4:
            best_val_loss, stale = va_loss, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
            if stale >= args.patience:
                print(f"Early stopping at epoch {epoch} "
                      f"(no val improvement for {args.patience} epochs)")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    test_loss, test_acc = run_epoch(model, test_loader, criterion, optimizer, device, False)
    print(f"\nTest loss={test_loss:.4f}  Test accuracy={test_acc:.4f}")
    print(f"Learned context gate: {model.context_gate.item():.4f}  "
          f"(initialised at {args.context_weight})")

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

    print(f"\nClassification report (test set, view={args.view}):")
    print(classification_report(trues, preds, labels=list(range(len(classes))),
                                target_names=classes, digits=3, zero_division=0))
    task = (trues != classes.index("baseline")) if "baseline" in classes else np.zeros(0, bool)
    if task.any():
        print(f"Task-only accuracy (baseline chunks excluded): "
              f"{(preds[task] == trues[task]).mean():.4f}")
    print("Confusion matrix (rows=true, cols=pred):")
    print(classes)
    print(confusion_matrix(trues, preds, labels=list(range(len(classes)))))

    if save_path:
        save_checkpoint(save_path, model_state=model.state_dict(), config=common,
                        classes=classes, channel_idx=idx.tolist())


if __name__ == "__main__":
    main()
