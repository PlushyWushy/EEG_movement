
"""
EEGNet, raw/no-preprocessing ablation (see train_eegnet-raw.py), converted
to an SNN via the standard rate-based ANN->SNN procedure.

Unlike the CNN-GRU line, EEGNet needs no architecture surgery to become
convertible -- it's already fully feedforward (temporal conv -> depthwise
spatial conv -> separable conv -> dense, no recurrence anywhere), exactly
the kind of network Akida's CNN2SNN and spikingjelly's ann2snn target.
The one incompatibility is EEGNet's ELU activations: rate coding needs an
activation whose output maps onto a non-negative firing rate, which ELU
(negative for negative inputs) doesn't provide, unlike ReLU.
FeedforwardEEGNet is EEGNet with elu1/elu2 swapped for relu1/relu2 and is
otherwise identical -- same depthwise/separable convs, same BatchNorm
placement, same bias=False convs, same max-norm weight constraint.

Trained with the same recipe as train_eegnet-raw.py (RandAugment + SMOTE
+ max-norm), then converted post-hoc exactly as in
train_cnn-raw-ann2snn.py: BatchNorm folded into its preceding conv, each
ReLU replaced by a subtractive-reset, zero-floored integrate-and-fire
neuron calibrated from that layer's activation on a calibration set,
weights copied unchanged, final Linear left as a non-spiking accumulator
(see that file's docstring for why the zero floor and the non-spiking
readout are both necessary, and for the latency caveat on deep/sparse
converted networks -- it applies here too, though EEGNet's every-conv-
has-BN, no-bare-bias design should condition better than CNN-GRU's did).

conv1->bn1->depthwise->bn2 (and sep_depthwise->sep_pointwise->bn3) are
purely linear runs -- no activation sits between them in the ANN -- so
they're composed directly with no spiking in between, exactly matching
the ANN's own computational graph: spikes only appear where the ANN
itself had a nonlinearity (relu1, relu2).
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from mlp.train_mlp import (
    CLASSES,
    build_dataset_raw,
    subject_dependent_split,
    subject_independent_split,
    smote_augment,
)

CACHE_PATH = Path(__file__).parent.parent / ".cache" / "eegnet_epochs_raw.npz"

EEGNET_CHANNELS = [
    "Fc3", "Fc1", "Fcz", "Fc2", "Fc4",
    "C3", "C1", "Cz", "C2", "C4",
    "Cp3", "Cp1", "Cpz", "Cp2", "Cp4",
]

RELU_LAYERS = ["relu1", "relu2"]


class RandAugmentEEG:
    """RandAugment (Cubuk et al., 2019) adapted to EEG time series.
    Duplicated from train_eegnet-raw.py -- hyphenated filenames aren't
    importable as modules, so variant scripts in this repo each carry
    their own copy rather than share one."""

    def __init__(self, n_ops=2, magnitude=0.5, seed=None):
        self.n_ops = n_ops
        self.magnitude = magnitude
        self.rng = np.random.default_rng(seed)
        self.ops = [
            self._jitter, self._scale, self._time_shift,
            self._time_mask, self._channel_dropout, self._magnitude_warp,
        ]

    def __call__(self, x):
        chosen = self.rng.choice(len(self.ops), size=min(self.n_ops, len(self.ops)),
                                  replace=False)
        for idx in chosen:
            x = self.ops[idx](x)
        return x

    def _jitter(self, x):
        sigma = 0.05 + 0.25 * self.magnitude
        return (x + self.rng.normal(0, sigma, size=x.shape)).astype(x.dtype)

    def _scale(self, x):
        factor = 1.0 + self.magnitude * self.rng.uniform(-0.3, 0.3)
        return (x * factor).astype(x.dtype)

    def _time_shift(self, x):
        max_shift = int(x.shape[1] * 0.1 * self.magnitude) + 1
        shift = self.rng.integers(-max_shift, max_shift + 1)
        return np.roll(x, shift, axis=1)

    def _time_mask(self, x):
        t = x.shape[1]
        mask_len = int(t * 0.2 * self.magnitude) + 1
        start = self.rng.integers(0, max(1, t - mask_len))
        x = x.copy()
        x[:, start:start + mask_len] = 0.0
        return x

    def _channel_dropout(self, x):
        n_drop = max(1, int(x.shape[0] * 0.2 * self.magnitude))
        idx = self.rng.choice(x.shape[0], size=n_drop, replace=False)
        x = x.copy()
        x[idx, :] = 0.0
        return x

    def _magnitude_warp(self, x):
        t = x.shape[1]
        n_knots = 4
        knot_vals = 1.0 + self.magnitude * self.rng.uniform(-0.2, 0.2, size=n_knots)
        knot_pos = np.linspace(0, t - 1, n_knots)
        envelope = np.interp(np.arange(t), knot_pos, knot_vals)
        return (x * envelope[np.newaxis, :]).astype(x.dtype)


class AugmentedEEGDataset(torch.utils.data.Dataset):
    def __init__(self, X, y, augment=None):
        self.X = X
        self.y = y
        self.augment = augment

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        x = self.X[idx]
        if self.augment is not None:
            x = self.augment(x)
        return torch.from_numpy(np.ascontiguousarray(x)).float(), \
            torch.tensor(self.y[idx], dtype=torch.long)


class FeedforwardEEGNet(nn.Module):
    """EEGNet (train_eegnet-raw.py's EEGNet) with elu1/elu2 replaced by
    relu1/relu2 -- the only change needed to make it convertible via the
    standard rate-based ANN->SNN procedure (see module docstring)."""

    def __init__(self, n_classes, chans, samples,
                 dropout_rate=0.5, kernel_length=64,
                 F1=8, D=2, F2=None, norm_rate=0.25,
                 dropout_type="Dropout"):
        super().__init__()
        F2 = F2 or F1 * D
        if dropout_type == "SpatialDropout2D":
            DropoutCls = nn.Dropout2d
        elif dropout_type == "Dropout":
            DropoutCls = nn.Dropout
        else:
            raise ValueError("dropout_type must be 'SpatialDropout2D' or 'Dropout'")

        self.norm_rate = norm_rate

        self.conv1 = nn.Conv2d(1, F1, kernel_size=(1, kernel_length),
                                padding="same", bias=False)
        self.bn1 = nn.BatchNorm2d(F1)
        self.depthwise = nn.Conv2d(F1, F1 * D, kernel_size=(chans, 1),
                                    groups=F1, bias=False)
        self.bn2 = nn.BatchNorm2d(F1 * D)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.AvgPool2d((1, 4))
        self.drop1 = DropoutCls(dropout_rate)

        self.sep_depthwise = nn.Conv2d(F1 * D, F1 * D, kernel_size=(1, 16),
                                        padding="same", groups=F1 * D, bias=False)
        self.sep_pointwise = nn.Conv2d(F1 * D, F2, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(F2)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.AvgPool2d((1, 8))
        self.drop2 = DropoutCls(dropout_rate)

        flat_samples = samples // 4 // 8
        self.classifier = nn.Linear(F2 * flat_samples, n_classes)

    def forward(self, x):
        x = x.unsqueeze(1)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.depthwise(x)
        x = self.bn2(x)
        x = self.relu1(x)
        x = self.pool1(x)
        x = self.drop1(x)

        x = self.sep_depthwise(x)
        x = self.sep_pointwise(x)
        x = self.bn3(x)
        x = self.relu2(x)
        x = self.pool2(x)
        x = self.drop2(x)

        x = x.flatten(1)
        return self.classifier(x)

    @torch.no_grad()
    def apply_max_norm(self):
        for weight, max_val in (
            (self.depthwise.weight, 1.0),
            (self.classifier.weight, self.norm_rate),
        ):
            norms = weight.view(weight.size(0), -1).norm(dim=1, keepdim=True)
            desired = norms.clamp(max=max_val)
            scale = (desired / (norms + 1e-8)).view(-1, *([1] * (weight.dim() - 1)))
            weight.mul_(scale)


def fold_conv_bn(conv, bn):
    """Fuse Conv2d + BatchNorm2d into a single equivalent Conv2d (conv's
    bias, if any, folded in too -- EEGNet's convs all use bias=False, so
    the fused bias here comes entirely from BatchNorm's learned shift)."""
    fused = nn.Conv2d(conv.in_channels, conv.out_channels, conv.kernel_size,
                       stride=conv.stride, padding=conv.padding,
                       dilation=conv.dilation, groups=conv.groups, bias=True)
    scale = bn.weight / torch.sqrt(bn.running_var + bn.eps)
    conv_bias = conv.bias if conv.bias is not None \
        else torch.zeros(conv.out_channels, device=conv.weight.device)
    with torch.no_grad():
        fused.weight.copy_(conv.weight * scale.view(-1, 1, 1, 1))
        fused.bias.copy_((conv_bias - bn.running_mean) * scale + bn.bias)
    fused.eval()
    for p in fused.parameters():
        p.requires_grad_(False)
    return fused


@torch.no_grad()
def calibrate_thresholds(model, calib_loader, device, n_batches, percentile):
    """Standard ANN->SNN threshold-balancing calibration (Diehl et al. 2015 /
    Rueckauer et al. 2017) -- see train_cnn-raw-ann2snn.py for the full
    rationale. Percentile rather than raw max, so a single outlier
    activation doesn't inflate the threshold and starve that layer's IF
    neuron of spikes."""
    model.eval()
    activations = {}

    def make_hook(name):
        def hook(module, inp, out):
            activations[name] = out.detach()
        return hook

    handles = [getattr(model, name).register_forward_hook(make_hook(name))
               for name in RELU_LAYERS]
    thresholds = {name: 0.0 for name in RELU_LAYERS}
    try:
        for i, (xb, _) in enumerate(calib_loader):
            if i >= n_batches:
                break
            model(xb.to(device))
            for name in RELU_LAYERS:
                act = activations[name].flatten().float()
                q = torch.quantile(act, percentile / 100.0).item()
                thresholds[name] = max(thresholds[name], q)
    finally:
        for h in handles:
            h.remove()
    return thresholds


class ConvertedSNN:
    """
    Inference-only rate-based conversion of a trained FeedforwardEEGNet:
    identical weights (BatchNorm folded into its preceding conv), each
    ReLU replaced by a subtractive-reset, zero-floored integrate-and-fire
    neuron using the calibrated thresholds. Nothing here is trained --
    this only runs forward passes.
    """

    def __init__(self, model, thresholds, device):
        self.conv1_fused = fold_conv_bn(model.conv1, model.bn1).to(device)
        self.depthwise_fused = fold_conv_bn(model.depthwise, model.bn2).to(device)
        self.sep_depthwise = model.sep_depthwise
        self.sep_pointwise_fused = fold_conv_bn(model.sep_pointwise, model.bn3).to(device)
        self.pool1 = model.pool1
        self.pool2 = model.pool2
        self.classifier = model.classifier
        self.thresholds = thresholds

    @staticmethod
    def _if_step(cur, mem, threshold):
        mem = mem + cur
        # Zero floor: prevents a mostly-negative input current (here, a
        # BatchNorm-derived shift dominating when upstream spikes are
        # sparse) from dragging an unbounded, leak-free membrane
        # permanently negative -- see train_cnn-raw-ann2snn.py's docstring
        # for the failure mode this avoids.
        mem = mem.clamp(min=0.0)
        spk = (mem >= threshold).float()
        mem = mem - spk * threshold
        return spk, mem

    @torch.no_grad()
    def run(self, x, timesteps, checkpoints):
        x = x.unsqueeze(1)
        # conv1->bn1->depthwise->bn2 has no activation in between in the
        # ANN, and x is constant across timesteps, so this linear chain is
        # computed once rather than redundantly inside the per-timestep loop.
        cur1_const = self.depthwise_fused(self.conv1_fused(x))

        mem1 = mem2 = None
        out_acc = torch.zeros(x.shape[0], self.classifier.out_features, device=x.device)
        spike_sums = {name: 0.0 for name in RELU_LAYERS}
        checkpoint_outputs = {}

        for t in range(1, timesteps + 1):
            mem1 = torch.zeros_like(cur1_const) if mem1 is None else mem1
            spk1, mem1 = self._if_step(cur1_const, mem1, self.thresholds["relu1"])
            spk1 = self.pool1(spk1)

            cur2 = self.sep_pointwise_fused(self.sep_depthwise(spk1))
            mem2 = torch.zeros_like(cur2) if mem2 is None else mem2
            spk2, mem2 = self._if_step(cur2, mem2, self.thresholds["relu2"])
            spk2 = self.pool2(spk2)

            out_acc = out_acc + self.classifier(spk2.flatten(1))

            for name, spk in zip(RELU_LAYERS, [spk1, spk2]):
                spike_sums[name] += spk.mean().item()

            if t in checkpoints:
                checkpoint_outputs[t] = (out_acc / t).clone()

        spike_rates = {name: s / timesteps for name, s in spike_sums.items()}
        return checkpoint_outputs, spike_rates


def train_model(model, train_loader, val_loader, device, args):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    def run_epoch_with_constraint(loader, train):
        model.train(train)
        total_loss, correct, n = 0.0, 0, 0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            if train:
                optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            if train:
                loss.backward()
                optimizer.step()
                model.apply_max_norm()
            total_loss += loss.item() * len(yb)
            correct += (logits.argmax(1) == yb).sum().item()
            n += len(yb)
        return total_loss / n, correct / n

    best_val_loss = float("inf")
    best_state = None
    epochs_without_improvement = 0

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = run_epoch_with_constraint(train_loader, train=True)
        val_loss, val_acc = run_epoch_with_constraint(val_loader, train=False)
        print(f"epoch {epoch:3d}  train_loss={train_loss:.4f} train_acc={train_acc:.4f}  "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}")

        if val_loss < best_val_loss - 1e-4:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.patience:
                print(f"Early stopping at epoch {epoch} "
                      f"(no val improvement for {args.patience} epochs)")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def evaluate_ann(model, test_loader, device):
    criterion = nn.CrossEntropyLoss()
    model.eval()
    all_preds, all_true = [], []
    total_loss, n = 0.0, 0
    with torch.no_grad():
        for xb, yb in test_loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            loss = criterion(logits, yb)
            total_loss += loss.item() * len(yb)
            n += len(yb)
            all_preds.append(logits.argmax(1).cpu().numpy())
            all_true.append(yb.cpu().numpy())
    all_preds = np.concatenate(all_preds)
    all_true = np.concatenate(all_true)
    print(f"\n[ANN] Test loss={total_loss / n:.4f}  Test accuracy={(all_preds == all_true).mean():.4f}")
    print("[ANN] Classification report (test set):")
    print(classification_report(all_true, all_preds, target_names=CLASSES, digits=3))
    return (all_preds == all_true).mean()


def evaluate_converted_snn(converted, test_loader, device, timesteps, checkpoints):
    preds_by_t = {t: [] for t in checkpoints}
    all_true = []
    spike_rate_sums = {name: 0.0 for name in RELU_LAYERS}
    n_batches = 0
    for xb, yb in test_loader:
        xb = xb.to(device)
        checkpoint_outputs, spike_rates = converted.run(xb, timesteps, checkpoints)
        for t, out in checkpoint_outputs.items():
            preds_by_t[t].append(out.argmax(1).cpu().numpy())
        all_true.append(yb.numpy())
        for name, r in spike_rates.items():
            spike_rate_sums[name] += r
        n_batches += 1

    all_true = np.concatenate(all_true)
    acc_by_t = {}
    for t in checkpoints:
        preds = np.concatenate(preds_by_t[t])
        acc_by_t[t] = (preds == all_true).mean()

    spike_rates = {name: s / n_batches for name, s in spike_rate_sums.items()}

    t_max = max(checkpoints)
    preds_max = np.concatenate(preds_by_t[t_max])
    print(f"\n[SNN, T={t_max}] Classification report (test set):")
    print(classification_report(all_true, preds_max, target_names=CLASSES, digits=3))
    print(f"[SNN, T={t_max}] Confusion matrix (rows=true, cols=pred):")
    print(CLASSES)
    print(confusion_matrix(all_true, preds_max))

    return acc_by_t, spike_rates


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--max-subjects", type=int, default=None,
                     help="Limit number of subjects processed (default: all available)")
    ap.add_argument("--no-cache", action="store_true",
                     help="Ignore/overwrite the cached processed-epoch array")
    ap.add_argument("--f1", type=int, default=8, help="Temporal filters (paper default: 8)")
    ap.add_argument("--d", type=int, default=2, help="Spatial filters per temporal filter (paper default: 2)")
    ap.add_argument("--f2", type=int, default=None, help="Pointwise filters (default: F1*D)")
    ap.add_argument("--kernel-length", type=int, default=80,
                     help="Temporal kernel length (paper rule: half the sampling rate; "
                          "160Hz here -> 80, vs. their 128Hz -> 64)")
    ap.add_argument("--dropout", type=float, default=0.5,
                     help="Paper: 0.5 within-subject, 0.25 cross-subject")
    ap.add_argument("--dropout-type", choices=["Dropout", "SpatialDropout2D"], default="Dropout")
    ap.add_argument("--norm-rate", type=float, default=0.25)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--patience", type=int, default=200)
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--test-frac", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--split-mode", choices=["subject-dependent", "subject-independent"],
                     default="subject-dependent")
    ap.add_argument("--no-smote", action="store_true")
    ap.add_argument("--smote-k", type=int, default=5)
    ap.add_argument("--no-augment", action="store_true",
                     help="Disable RandAugment-style training-time data augmentation")
    ap.add_argument("--randaugment-n", type=int, default=2)
    ap.add_argument("--randaugment-m", type=float, default=0.5)
    ap.add_argument("--timesteps", type=int, default=512,
                     help="Max SNN simulation length (rate-coding timesteps)")
    ap.add_argument("--timestep-checkpoints", type=str, default="16,32,64,128,256,512",
                     help="Comma-separated T values to report accuracy at (capped at --timesteps)")
    ap.add_argument("--calib-batches", type=int, default=20,
                     help="Number of (unaugmented) training batches used for threshold calibration")
    ap.add_argument("--calib-percentile", type=float, default=99.9,
                     help="Percentile of calibration-set activations used as each IF "
                          "neuron's threshold (robust max, Rueckauer et al. 2017)")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    X, y, groups = build_dataset_raw(CACHE_PATH, EEGNET_CHANNELS, expand_pairs=None,
                                      max_subjects=args.max_subjects,
                                      use_cache=not args.no_cache)
    print(f"Dataset: X={X.shape}, classes={CLASSES}, "
          f"subjects={len(set(groups.tolist()))}")
    print("Class counts:", {c: int((y == i).sum()) for i, c in enumerate(CLASSES)})

    if args.split_mode == "subject-dependent":
        train_mask, val_mask, test_mask = subject_dependent_split(
            y, groups, val_frac=args.val_frac, test_frac=args.test_frac, seed=args.seed
        )
    else:
        train_mask, val_mask, test_mask = subject_independent_split(
            groups, val_frac=args.val_frac, test_frac=args.test_frac, seed=args.seed
        )
    X_train, y_train = X[train_mask], y[train_mask]
    X_val, y_val = X[val_mask], y[val_mask]
    X_test, y_test = X[test_mask], y[test_mask]
    print(f"Epochs: train={len(y_train)} val={len(y_val)} test={len(y_test)}")

    if not args.no_smote:
        X_train, y_train = smote_augment(X_train, y_train, k=args.smote_k, seed=args.seed)
        counts = {c: int((y_train == i).sum()) for i, c in enumerate(CLASSES)}
        print(f"After SMOTE: train={len(y_train)} epochs, class counts={counts}")

    device = torch.device("cuda" if torch.cuda.is_available()
                           else "mps" if torch.backends.mps.is_available()
                           else "cpu")
    print(f"Using device: {device}")

    def make_loader(Xa, ya, shuffle, augment=None):
        if augment is not None:
            ds = AugmentedEEGDataset(Xa, ya, augment=augment)
        else:
            ds = torch.utils.data.TensorDataset(
                torch.from_numpy(Xa).float(), torch.from_numpy(ya).long()
            )
        return torch.utils.data.DataLoader(ds, batch_size=args.batch_size, shuffle=shuffle)

    augmenter = None
    if not args.no_augment:
        augmenter = RandAugmentEEG(n_ops=args.randaugment_n, magnitude=args.randaugment_m,
                                    seed=args.seed)
        print(f"RandAugment: n_ops={args.randaugment_n} magnitude={args.randaugment_m} "
              f"(train split only)")

    train_loader = make_loader(X_train, y_train, shuffle=True, augment=augmenter)
    val_loader = make_loader(X_val, y_val, shuffle=False)
    test_loader = make_loader(X_test, y_test, shuffle=False)
    # Calibration wants clean, representative activations -- not RandAugment's
    # jittered/masked/scaled ones -- so it gets its own unaugmented loader.
    calib_loader = make_loader(X_train, y_train, shuffle=True)

    n_channels, n_times = X_train.shape[1], X_train.shape[2]
    model = FeedforwardEEGNet(len(CLASSES), n_channels, n_times,
                               dropout_rate=args.dropout, kernel_length=args.kernel_length,
                               F1=args.f1, D=args.d, F2=args.f2, norm_rate=args.norm_rate,
                               dropout_type=args.dropout_type).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    model = train_model(model, train_loader, val_loader, device, args)
    ann_test_acc = evaluate_ann(model, test_loader, device)

    print(f"\nCalibrating IF thresholds from {args.calib_batches} training batches "
          f"(percentile={args.calib_percentile})...")
    thresholds = calibrate_thresholds(model, calib_loader, device,
                                       args.calib_batches, args.calib_percentile)
    print("Calibrated thresholds:", {k: round(v, 4) for k, v in thresholds.items()})

    checkpoints = sorted({min(int(t), args.timesteps)
                           for t in args.timestep_checkpoints.split(",")} | {args.timesteps})
    converted = ConvertedSNN(model, thresholds, device)
    acc_by_t, spike_rates = evaluate_converted_snn(converted, test_loader, device,
                                                    args.timesteps, checkpoints)

    print("\n=== Benchmark summary ===")
    print(f"ANN:                 test_acc={ann_test_acc:.4f}  params={n_params:,}")
    for t in checkpoints:
        print(f"SNN (T={t:>4}):        test_acc={acc_by_t[t]:.4f}")
    print("\n[SNN] Mean spike rate per layer at T={} (fraction of neuron-timesteps that "
          "fired, test set) -- lower is sparser/cheaper on spiking hardware:".format(args.timesteps))
    for name, rate in spike_rates.items():
        print(f"  {name:>10}: {rate:.4f}")


if __name__ == "__main__":
    main()
