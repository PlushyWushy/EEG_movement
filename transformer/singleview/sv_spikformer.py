"""
SNN singleview, spiking attention
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix
from snntorch import surrogate
from snntorch import utils as snn_utils

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "transformer" / "multiview"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

# Everything spiking comes from mt_spikformer, unforked. Staying on that one
# module (rather than mixing in mt_cnntosnn) keeps a single class hierarchy:
# the model below subclasses its ReLU MultiViewRunModel, so its SpikingViewEncoder,
# its calibrations and its SpikformerBlock all apply without adaptation.
from mt_spikformer import (
    HybridSNNModel,
    MultiViewRunModel as ReluMultiViewRunModel,
    SpikeMonitor,
    SpikformerBlock,
    SpikingViewEncoder,
    ViewEncoder as ReluViewEncoder,
    calibrate_ssa_scales,
    calibrate_thresholds,
    calibrate_token_scales,
    evaluate_hybrid,
    lif,
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
    """The phase-1 ANN: one convertible encoder, no fusion attention."""

    def __init__(self, channel_idx, d_model=128, dropout=0.3, f1=8, depth=2, **kw):
        super().__init__(d_model=d_model, dropout=dropout, f1=f1, depth=depth, **kw)
        self.chunk_encoder = SingleViewChunkEncoder(
            channel_idx, d_model, dropout=dropout, f1=f1, depth=depth,
            encoder_cls=ReluViewEncoder)


class SingleSpikingChunkEncoder:
    """Rate-decoding front end for the hybrid baseline: one spiking encoder,
    its time-averaged projection IS the chunk token."""

    def __init__(self, chunk_encoder, thresholds):
        self.encoder = SpikingViewEncoder(chunk_encoder.encoders[VIEW_KEY],
                                          thresholds[VIEW_KEY])
        self.idx = chunk_encoder.idx

    @torch.no_grad()
    def forward(self, x, timesteps, checkpoints):
        emb_by_t, rates = self.encoder.run(x.index_select(1, self.idx),
                                           timesteps, checkpoints)
        return emb_by_t, {f"{VIEW_KEY}.{a}": r for a, r in rates.items()}


class SingleHybridSNNModel(HybridSNNModel):
    """Converted CNN + ANN context attention. forward() inherited verbatim."""

    def __init__(self, model, thresholds, chunk_batch=64):
        self.model = model
        self.chunk_encoder = SingleSpikingChunkEncoder(model.chunk_encoder, thresholds)
        self.chunk_batch = chunk_batch


class SingleSpikingFrontEnd:
    """The one converted encoder, stepped a timestep at a time and stopping at
    the projection. Not an nn.Module on purpose: its weights never enter the
    spiking model's .parameters(), so the phase-2 optimizer cannot reach the
    CNN even by accident, and it runs under no_grad throughout."""

    def __init__(self, chunk_encoder, thresholds, token_scales):
        self.encoder = SpikingViewEncoder(chunk_encoder.encoders[VIEW_KEY],
                                          thresholds[VIEW_KEY])
        self.idx = chunk_encoder.idx
        self.scale = token_scales[VIEW_KEY]

    @torch.no_grad()
    def start(self, x):
        return self.encoder.start(x.index_select(1, self.idx))

    @torch.no_grad()
    def step(self, state, record=False):
        cur, spk1, spk2 = self.encoder.step(state)
        rates = {}
        if record:
            rates[f"{VIEW_KEY}.act1"] = spk1.mean()
            rates[f"{VIEW_KEY}.act2"] = spk2.mean()
        return cur / self.scale, rates


class SingleSpikformerRunModel(nn.Module):
    """Frozen spiking CNN -> spiking causal context attention -> readout.

    Two modules from the multiview version are simply absent here, and both
    for the same reason: with one view there is nothing to fuse.

    * The fusion SpikformerBlock is gone. It attended over 5 view tokens; one
      token has nothing to attend over, so the chunk representation goes
      straight from the projection to the context paths.
    * The token LIF is gone with it. In the multiview model that neuron
      existed to turn the projection current into the SPIKES the fusion
      attention needed as input. Nothing downstream needs spikes at that
      point any more -- local_lif and ctx_lif already take a current, exactly
      as they did there (they were fed the mean of the fused view spikes) --
      so keeping it would just be a redundant nonlinearity.

    What survives from view_emb is `token_bias`: the trainable per-feature
    bias that decides where the projection current sits in the input neurons'
    firing range. Everything else -- the position-free local path, the
    positioned context path, the gate, the mean-over-time readout -- matches
    the multiview model line for line.
    """

    def __init__(self, ann_model, thresholds, token_scales, timesteps,
                 n_classes=len(CLASSES), heads=4, ctx_layers=2, dropout=0.3,
                 max_len=64, beta=0.9, scale=0.125, attn_threshold=0.5,
                 spike_grad=None):
        super().__init__()
        spike_grad = spike_grad or surrogate.fast_sigmoid()
        self.front = SingleSpikingFrontEnd(ann_model.chunk_encoder, thresholds,
                                           token_scales)
        self.timesteps = timesteps
        self.context_len = ann_model.context_len
        d = ann_model.pos.embedding_dim

        self.token_bias = nn.Parameter(torch.zeros(d))
        nn.init.normal_(self.token_bias, std=0.02)

        self.pos = nn.Embedding(max_len, d)
        self.pos.weight.data.copy_(ann_model.pos.weight.detach())
        self.local_lif = lif(beta, spike_grad)
        self.ctx_lif = lif(beta, spike_grad)
        self.context = nn.ModuleList([
            SpikformerBlock(d, heads, beta, spike_grad, scale, attn_threshold)
            for _ in range(ctx_layers)])

        self.context_gate = nn.Parameter(ann_model.context_gate.detach().clone())
        self.norm = nn.LayerNorm(d)
        self.norm.load_state_dict(ann_model.norm.state_dict())
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(d, n_classes))
        self.head.load_state_dict(ann_model.head.state_dict())

        self.record_spikes = False
        self.spike_rates = {}

    def attn_mask(self, T, device, pad_mask):
        """True = disallowed. Causal, optionally window-limited, combined with
        the padding mask on the key axis and broadcast over heads."""
        i = torch.arange(T, device=device).unsqueeze(1)
        j = torch.arange(T, device=device).unsqueeze(0)
        causal = j > i
        if self.context_len > 0:
            causal = causal | ((i - j) >= self.context_len)
        return causal[None, None] | pad_mask[:, None, None, :]

    def forward(self, x, pad_mask):
        B, T = x.shape[:2]
        d = self.token_bias.shape[0]
        snn_utils.reset(self)

        keep = (~pad_mask).reshape(-1)
        valid = (~pad_mask).unsqueeze(-1).float()
        chunks = x.reshape(B * T, x.shape[2], x.shape[3])[keep]
        state = self.front.start(chunks)
        mask = self.attn_mask(T, x.device, pad_mask)
        pos = self.pos(torch.arange(T, device=x.device)).unsqueeze(0)

        feat, rate_sums = 0.0, {}
        for _ in range(self.timesteps):
            tok_cur, rates = self.front.step(state, self.record_spikes)

            grid = torch.zeros(B * T, d, device=x.device, dtype=tok_cur.dtype)
            grid[keep] = tok_cur + self.token_bias
            grid = grid.reshape(B, T, d)

            local = self.local_lif(grid)                   # position-free
            h = self.ctx_lif(grid + pos)                   # context path only
            for block in self.context:
                h = block(h, mask) * valid
            feat = feat + local + self.context_gate * h

            if self.record_spikes:
                for k, v in rates.items():
                    rate_sums[k] = rate_sums.get(k, 0.0) + v

        if self.record_spikes:
            self.spike_rates = {k: float(v / self.timesteps)
                                for k, v in rate_sums.items()}
        return self.head(self.norm(feat / self.timesteps))


def load_model(path, device="cpu"):
    """Reconstruct the trained spiking model saved via --save. Rebuilds the
    ANN first (its weights are what the frozen, folded, IF-converted front
    end is deterministically derived from -- SingleSpikingFrontEnd is not an
    nn.Module, so its weights are never in spiking_state at all), then
    restores the trained spiking attention weights on top. Returns
    (spiking_model, classes)."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    idx = np.asarray(ckpt["channel_idx"], dtype=np.int64)
    ann_model = SingleViewRunModel(idx, **ckpt["ann_config"]).to(device)
    ann_model.load_state_dict(ckpt["ann_state"])
    spiking = SingleSpikformerRunModel(
        ann_model, ckpt["thresholds"], ckpt["token_scales"],
        **ckpt["spiking_config"]).to(device)
    spiking.load_state_dict(ckpt["spiking_state"])
    spiking.eval()
    return spiking, ckpt["classes"]


def main(task_spec=None, split_spec=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--view", choices=["all"] + VIEW_NAMES, default="all")
    ap.add_argument("--max-subjects", type=int, default=None)
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--split-mode", choices=["run-holdout", "subject-independent"],
                    default="run-holdout")
    ap.add_argument("--d-model", type=int, default=128)
    ap.add_argument("--ctx-layers", type=int, default=2)
    ap.add_argument("--ctx-heads", type=int, default=4)
    ap.add_argument("--f1", type=int, default=8)
    ap.add_argument("--depth", type=int, default=2)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--context-weight", type=float, default=0.1)
    ap.add_argument("--freeze-context-weight", action="store_true")
    ap.add_argument("--context-len", type=int, default=0)
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
    ap.add_argument("--pretrain-epochs", type=int, default=0)
    ap.add_argument("--pretrain-mask-frac", type=float, default=0.15)
    ap.add_argument("--pretrain-lr", type=float, default=1e-3)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--no-class-weights", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    # conversion + phase 2
    ap.add_argument("--timesteps", type=int, default=32,
                    help="Shared by the converted CNN and the spiking transformer; "
                         "phase 2 backpropagates through every timestep")
    ap.add_argument("--timestep-checkpoints", type=str, default="4,8,16,32")
    ap.add_argument("--calib-batches", type=int, default=10)
    ap.add_argument("--calib-percentile", type=float, default=99.9)
    ap.add_argument("--snn-chunk-batch", type=int, default=64)
    ap.add_argument("--snn-limit-batches", type=int, default=0)
    ap.add_argument("--no-hybrid-eval", action="store_true")
    ap.add_argument("--snn-epochs", type=int, default=None,
                    help="Surrogate-gradient epochs (default: same budget as --epochs)")
    ap.add_argument("--snn-lr", type=float, default=1e-3)
    ap.add_argument("--ssa-heads", type=int, default=4)
    ap.add_argument("--ssa-scale", type=float, default=0.125)
    ap.add_argument("--ssa-target", type=float, default=1.0)
    ap.add_argument("--attn-threshold", type=float, default=0.5)
    ap.add_argument("--lif-beta", type=float, default=0.9)
    ap.add_argument("--token-percentile", type=float, default=99.0)
    ap.add_argument("--save", nargs="?", const="", default=None, metavar="PATH",
                    help="Save the trained spiking model to PATH at the end (bare "
                         "--save picks checkpoints/<script>_<timestamp>.pt). Bundles "
                         "the ANN's weights/config, the calibrated thresholds and "
                         "token scales, and the spiking attention's weights/config "
                         "together, since the frozen front end isn't an nn.Module and "
                         "would otherwise be invisible to state_dict() -- see "
                         "load_model().")
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
    print(f"\nSingle-view Spikformer: view='{args.view}', {len(idx)} of "
          f"{len(CANONICAL_CHANNELS)} channels. With one view there is no fusion "
          f"stage, so the spiking attention here is the context transformer alone.")

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

    ann_config = dict(
        n_classes=len(classes), d_model=args.d_model, ctx_layers=args.ctx_layers,
        ctx_heads=args.ctx_heads, dropout=args.dropout, max_len=X.shape[1],
        context_weight=args.context_weight, freeze_context=args.freeze_context_weight,
        context_len=args.context_len, f1=args.f1, depth=args.depth)
    model = SingleViewRunModel(idx, **ann_config).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    save_path = None
    if args.save is not None:
        save_path = args.save or str(ROOT / "checkpoints" /
            f"{Path(__file__).stem}_{time.strftime('%Y%m%d-%H%M%S')}.pt")

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

    # ------------------------------------------------------------------
    # Convert, then retrain the spiking attention on the frozen CNN.
    # ------------------------------------------------------------------
    print(f"\nConverting the view CNN and calibrating...")
    thresholds = calibrate_thresholds(model, loader(tr_idx, True), device,
                                      args.calib_batches, args.calib_percentile)
    for key, th in thresholds.items():
        print(f"  {key:>19}: act1={th['act1']:.4f}  act2={th['act2']:.4f}")

    checkpoints = sorted({min(int(t), args.timesteps)
                          for t in args.timestep_checkpoints.split(",")}
                         | {args.timesteps})
    if not args.no_hybrid_eval:
        print(f"\nBaseline: converted CNN + ANN context attention, up to "
              f"T={args.timesteps}:")
        hybrid = SingleHybridSNNModel(model, thresholds,
                                      chunk_batch=args.snn_chunk_batch)
        acc_by_t, h_trues, h_preds, h_ann, _ = evaluate_hybrid(
            hybrid, test_loader, device, args.timesteps, checkpoints,
            limit_batches=args.snn_limit_batches)
        print(f"  ANN attention:           test_acc={(h_ann == h_trues).mean():.4f}")
        for t in checkpoints:
            print(f"  hybrid (T={t:>4}):          test_acc={acc_by_t[t]:.4f}")
        print(f"  agreement with the ANN at T={max(checkpoints)}: "
              f"{(h_preds == h_ann).mean():.4f}")

    token_scales = calibrate_token_scales(model, loader(tr_idx, True), device,
                                          args.calib_batches, args.token_percentile)
    print("\nToken gain (percentile of the frozen projection's output):",
          {k: round(v, 3) for k, v in token_scales.items()})

    spiking_config = dict(
        timesteps=args.timesteps, n_classes=len(classes), heads=args.ssa_heads,
        ctx_layers=args.ctx_layers, dropout=args.dropout, max_len=X.shape[1],
        beta=args.lif_beta, scale=args.ssa_scale, attn_threshold=args.attn_threshold)
    spiking = SingleSpikformerRunModel(
        model, thresholds, token_scales, **spiking_config).to(device)

    frozen = sum(p.numel() for p in model.chunk_encoder.encoders.parameters())
    trainable = sum(p.numel() for p in spiking.parameters() if p.requires_grad)
    print(f"Spiking model: {trainable:,} trainable attention parameters, "
          f"{frozen:,} frozen CNN parameters")

    ssa_scales, ssa_rates = calibrate_ssa_scales(spiking, train_loader, device,
                                                 target=args.ssa_target)
    print("SSA scales, and the firing rate each produces at initialisation:")
    for name, s in ssa_scales.items():
        rate = ssa_rates.get(f"{name}.attn_lif", float("nan"))
        print(f"  {name:>22}: scale={s:<10.4f} attn_lif fires {rate:.4f}")

    cnn_snapshot = {n: q.detach().clone()
                    for n, q in model.chunk_encoder.encoders.named_parameters()}

    snn_epochs = args.epochs if args.snn_epochs is None else args.snn_epochs
    print(f"\nTraining spiking attention with surrogate gradients "
          f"(beta={args.lif_beta}, T={args.timesteps}) for {snn_epochs} epochs:")
    snn_opt = torch.optim.AdamW(spiking.parameters(), lr=args.snn_lr,
                                weight_decay=args.weight_decay)
    best_val, best_snn_state, stale = float("inf"), None, 0
    for epoch in range(1, snn_epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc = run_epoch(spiking, train_loader, criterion, snn_opt, device, True)
        va_loss, va_acc = run_epoch(spiking, val_loader, criterion, snn_opt, device, False)
        print(f"snn epoch {epoch:3d}  train_loss={tr_loss:.4f} train_acc={tr_acc:.4f}  "
              f"val_loss={va_loss:.4f} val_acc={va_acc:.4f}  "
              f"gate={spiking.context_gate.item():.3f}  ({time.time() - t0:.0f}s)",
              flush=True)
        if va_loss < best_val - 1e-4:
            best_val, stale = va_loss, 0
            best_snn_state = {k: v.clone() for k, v in spiking.state_dict().items()}
        else:
            stale += 1
            if stale >= args.patience:
                print(f"Early stopping at epoch {epoch}")
                break
    if best_snn_state is not None:
        spiking.load_state_dict(best_snn_state)

    drift = max((cnn_snapshot[n] - q).abs().max().item()
                for n, q in model.chunk_encoder.encoders.named_parameters())
    print(f"Frozen-CNN check: largest weight change across phase 2 = {drift:.2e}")

    snn_loss, snn_acc = run_epoch(spiking, test_loader, criterion, snn_opt, device, False)
    print(f"\n[Spiking] Test loss={snn_loss:.4f}  Test accuracy={snn_acc:.4f}")

    spiking.eval()
    spiking.record_spikes = True
    sp_preds, sp_trues = [], []
    with SpikeMonitor(spiking) as monitor, torch.no_grad():
        for xb, yb, lb in test_loader:
            xb, lb = xb.to(device), lb.to(device)
            logits = spiking(xb, make_pad_mask(lb, xb.shape[1], device))
            valid = yb.reshape(-1) != -1
            sp_preds.append(
                logits.reshape(-1, logits.shape[-1]).argmax(1).cpu()[valid].numpy())
            sp_trues.append(yb.reshape(-1)[valid].numpy())
        lif_rates = monitor.rates()
    sp_preds, sp_trues = np.concatenate(sp_preds), np.concatenate(sp_trues)

    print("\n[Spiking] Classification report (test set):")
    print(classification_report(sp_trues, sp_preds, labels=list(range(len(classes))),
                                target_names=classes, digits=3, zero_division=0))
    sp_task = (sp_trues != classes.index("baseline")) if "baseline" in classes else np.zeros(0, bool)
    if sp_task.any():
        print(f"[Spiking] Task-only accuracy (baseline chunks excluded): "
              f"{(sp_preds[sp_task] == sp_trues[sp_task]).mean():.4f}")
    print("[Spiking] Confusion matrix (rows=true, cols=pred):")
    print(classes)
    print(confusion_matrix(sp_trues, sp_preds, labels=list(range(len(classes)))))

    print(f"\n=== ANN attention vs spiking attention (T={args.timesteps}) ===")
    print(f"ANN throughout:                       test_acc={test_acc:.4f}  "
          f"({args.epochs} epoch budget)")
    print(f"Spiking CNN + Spikformer attention:   test_acc={snn_acc:.4f}  "
          f"(+{snn_epochs} more on a frozen CNN)")
    print("These budgets are not equal -- phase 2 trains on top of phase 1, and "
          "freezing the CNN to retrain the head is itself a regulariser.")
    print(f"\n[Spiking] Mean spike rate per layer at T={args.timesteps}:")
    for key, rate in {**spiking.spike_rates, **lif_rates}.items():
        print(f"  {key:>28}: {rate:.4f}")

    if save_path:
        save_checkpoint(save_path, ann_state=model.state_dict(), ann_config=ann_config,
                        thresholds=thresholds, token_scales=token_scales,
                        spiking_state=spiking.state_dict(), spiking_config=spiking_config,
                        classes=classes, view=args.view, channel_idx=idx.tolist())


if __name__ == "__main__":
    main()
