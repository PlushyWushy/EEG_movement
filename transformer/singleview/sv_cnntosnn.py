"""
SNN singleview
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
sys.path.insert(0, str(ROOT / "transformer" / "multiview"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

# Same policy as mt_singleview: import the multiview version rather than fork
# it, so the conversion procedure under test is literally the same code. Note
# ViewEncoder here is mt_cnntosnn's ReLU one, not the parent's ELU one -- that
# swap is what makes the encoder convertible at all.
from mt_cnntosnn import (
    HybridSNNModel,
    MultiViewRunModel as ReluMultiViewRunModel,
    SpikingViewEncoder,
    ViewEncoder as ReluViewEncoder,
    calibrate_thresholds,
    evaluate_hybrid,
)
from mt_singleview import VIEW_KEY, SingleViewChunkEncoder, pretrain_single, view_channel_index
from train_multiview_transformer import (
    CACHE_DIR,
    CANONICAL_CHANNELS,
    VIEW_NAMES,
    RunSequenceDataset,
    build_sequences,
    build_smote_neighbors,
    make_pad_mask,
    run_epoch,
    run_holdout_split,
    save_checkpoint,
    subject_holdout_split,
)
from mlp.train_mlp import CLASS_TO_IDX, CLASSES


class SingleViewRunModel(ReluMultiViewRunModel):
    """mt_singleview.SingleViewRunModel, rebased onto the ReLU model so the
    lone encoder is convertible. Everything above the chunk encoder is
    mt_cnntosnn's, inherited untouched."""

    def __init__(self, channel_idx, d_model=128, dropout=0.3, f1=8, depth=2, **kw):
        super().__init__(d_model=d_model, dropout=dropout, f1=f1, depth=depth, **kw)
        self.chunk_encoder = SingleViewChunkEncoder(
            channel_idx, d_model, dropout=dropout, f1=f1, depth=depth,
            encoder_cls=ReluViewEncoder)


class SingleSpikingChunkEncoder:
    """The multiview SpikingChunkEncoder with the parts that only exist to
    serve five views taken out: no per-view loop, no view embeddings, no
    fusion attention. One spiking encoder, and its rate-decoded output IS the
    chunk token. Same return contract, so HybridSNNModel.forward and
    evaluate_hybrid drive it unchanged."""

    def __init__(self, chunk_encoder, thresholds):
        self.encoder = SpikingViewEncoder(chunk_encoder.encoders[VIEW_KEY],
                                          thresholds[VIEW_KEY])
        self.idx = chunk_encoder.idx

    @torch.no_grad()
    def forward(self, x, timesteps, checkpoints):
        """x: (N, 64, 640) -> ({t: (N, d_model)}, spike rates)."""
        emb_by_t, rates = self.encoder.run(x.index_select(1, self.idx),
                                           timesteps, checkpoints)
        return emb_by_t, {f"{VIEW_KEY}.{a}": r for a, r in rates.items()}


class SingleHybridSNNModel(HybridSNNModel):
    """HybridSNNModel with the single-view spiking front end. forward() is
    inherited verbatim -- the padded-chunk handling, the chunk sub-batching
    and the ANN context stack above are all the multiview code."""

    def __init__(self, model, thresholds, chunk_batch=64):
        self.model = model
        self.chunk_encoder = SingleSpikingChunkEncoder(model.chunk_encoder, thresholds)
        self.chunk_batch = chunk_batch


def load_model(path, device="cpu"):
    """Reconstruct the trained ANN saved via --save. Returns (model, thresholds,
    classes) -- thresholds is None if the checkpoint was saved with --no-snn
    before calibration ran. Wrap the result in SingleHybridSNNModel(model,
    thresholds) to run it as the converted spiking network."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    idx = np.asarray(ckpt["channel_idx"], dtype=np.int64)
    model = SingleViewRunModel(idx, **ckpt["config"]).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, ckpt.get("thresholds"), ckpt["classes"]


def main(task_spec=None, split_spec=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--view", choices=["all"] + VIEW_NAMES, default="all",
                    help="'all' (default) ablates the multi-view architecture; a view "
                         "name keeps only that region's channels")
    ap.add_argument("--max-subjects", type=int, default=None)
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--split-mode", choices=["run-holdout", "subject-independent"],
                    default="run-holdout")
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
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--no-class-weights", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    # conversion
    ap.add_argument("--no-snn", action="store_true")
    ap.add_argument("--timesteps", type=int, default=128)
    ap.add_argument("--timestep-checkpoints", type=str, default="8,16,32,64,128")
    ap.add_argument("--calib-batches", type=int, default=10)
    ap.add_argument("--calib-percentile", type=float, default=99.9)
    ap.add_argument("--snn-chunk-batch", type=int, default=64)
    ap.add_argument("--snn-limit-batches", type=int, default=0)
    ap.add_argument("--save", nargs="?", const="", default=None, metavar="PATH",
                    help="Save the trained ANN to PATH (bare --save picks "
                         "checkpoints/<script>_<timestamp>.pt). Saved once after ANN "
                         "training (or, if --no-snn wasn't passed, again at the end "
                         "with calibrated thresholds included) -- see load_model().")
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

    idx = view_channel_index(args.view)
    print(f"\nSingle-view SNN conversion: view='{args.view}', {len(idx)} of "
          f"{len(CANONICAL_CHANNELS)} channels through one encoder "
          f"(no fusion attention to preserve, so the only ANN attention left "
          f"above the spikes is the causal context transformer).")

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
        print(f"SMOTE: neighbour index built in {time.time()-t0:.0f}s")

    aug = not args.no_augment

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

    model_config = dict(
        n_classes=len(classes), d_model=args.d_model, ctx_layers=args.ctx_layers,
        ctx_heads=args.ctx_heads, dropout=args.dropout, max_len=X.shape[1],
        context_weight=args.context_weight, freeze_context=args.freeze_context_weight,
        context_len=args.context_len, f1=args.f1, depth=args.depth)
    model = SingleViewRunModel(idx, **model_config).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    save_path = None
    if args.save is not None:
        save_path = args.save or str(ROOT / "checkpoints" /
            f"{Path(__file__).stem}_{time.strftime('%Y%m%d-%H%M%S')}.pt")

    def save_now(**extra):
        save_checkpoint(save_path, model_state=model.state_dict(), config=model_config,
                        classes=classes, view=args.view, channel_idx=idx.tolist(), **extra)

    if args.pretrain_epochs > 0:
        print(f"\nCausal masked-chunk pre-training (mask_frac={args.pretrain_mask_frac}):")
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
                print(f"Early stopping at epoch {epoch}")
                break
    if best_state is not None:
        model.load_state_dict(best_state)

    test_loss, test_acc = run_epoch(model, test_loader, criterion, optimizer, device, False)
    print(f"\n[ANN] Test loss={test_loss:.4f}  Test accuracy={test_acc:.4f}")

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
    print("\n[ANN] Classification report (test set):")
    print(classification_report(trues, preds, labels=list(range(len(classes))),
                                target_names=classes, digits=3, zero_division=0))
    task = (trues != classes.index("baseline")) if "baseline" in classes else np.zeros(0, bool)
    if task.any():
        print(f"[ANN] Task-only accuracy (baseline chunks excluded): "
              f"{(preds[task] == trues[task]).mean():.4f}")

    if args.no_snn:
        if save_path:
            save_now()
        return

    # ---------------------------------------------------------------------
    # Convert the one view CNN and re-run the test set.
    # ---------------------------------------------------------------------
    print(f"\nConverting 1 view CNN to a spiking network.")
    print(f"Calibrating IF thresholds on {args.calib_batches} unaugmented "
          f"training batches (percentile={args.calib_percentile})...")
    thresholds = calibrate_thresholds(model, loader(tr_idx, True), device,
                                      args.calib_batches, args.calib_percentile)
    for key, th in thresholds.items():
        print(f"  {key:>19}: act1={th['act1']:.4f}  act2={th['act2']:.4f}")

    checkpoints = sorted({min(int(t), args.timesteps)
                          for t in args.timestep_checkpoints.split(",")}
                         | {args.timesteps})
    if args.snn_limit_batches:
        print(f"NOTE: SNN evaluated on the first {args.snn_limit_batches} test "
              f"batches only")
    print(f"\nSimulating the hybrid (spiking view CNN -> ANN context attention) "
          f"up to T={args.timesteps}:")
    hybrid = SingleHybridSNNModel(model, thresholds, chunk_batch=args.snn_chunk_batch)
    acc_by_t, snn_trues, snn_preds, ann_preds, spike_rates = evaluate_hybrid(
        hybrid, test_loader, device, args.timesteps, checkpoints,
        limit_batches=args.snn_limit_batches)

    t_max = max(checkpoints)
    print(f"\n[SNN, T={t_max}] Classification report (test set):")
    print(classification_report(snn_trues, snn_preds,
                                labels=list(range(len(classes))),
                                target_names=classes, digits=3, zero_division=0))
    snn_task = (snn_trues != classes.index("baseline")) if "baseline" in classes else np.zeros(0, bool)
    if snn_task.any():
        print(f"[SNN, T={t_max}] Task-only accuracy (baseline chunks excluded): "
              f"{(snn_preds[snn_task] == snn_trues[snn_task]).mean():.4f}")
    print(f"[SNN, T={t_max}] Confusion matrix (rows=true, cols=pred):")
    print(classes)
    print(confusion_matrix(snn_trues, snn_preds, labels=list(range(len(classes)))))

    print("\n=== ANN vs hybrid SNN (same test chunks) ===")
    print(f"ANN view CNN:             test_acc={(ann_preds == snn_trues).mean():.4f}")
    for t in checkpoints:
        print(f"Spiking view CNN (T={t:>4}): test_acc={acc_by_t[t]:.4f}")
    print(f"Prediction agreement with the ANN at T={t_max}: "
          f"{(snn_preds == ann_preds).mean():.4f}  "
          f"(1.0 would be a lossless conversion)")
    print(f"\n[SNN] Mean spike rate per layer at T={args.timesteps}:")
    for key in sorted(spike_rates):
        print(f"  {key:>26}: {spike_rates[key]:.4f}")

    if save_path:
        save_now(thresholds=thresholds)


if __name__ == "__main__":
    main()
