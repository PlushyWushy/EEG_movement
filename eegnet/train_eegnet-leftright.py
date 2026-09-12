
"""EEGNet restricted to left_fist vs right_fist only (binary classification).

Same model, same RandAugment/SMOTE/training loop, and same cleaning
pipeline as train_eegnet.py -- the only change is filtering the built
5-class dataset down to just these two classes (remapped to labels 0/1)
right after loading, via the shared filter_to_classes() utility.
"""

import argparse
import sys

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix

import mne
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from mlp.train_mlp import (
    DATA_ROOT,
    MONTAGE,
    RUN_LABELS,
    CLASS_TO_IDX,
    EPOCH_TMIN,
    EPOCH_TMAX,
    N_SAMPLES,
    remove_ica_artifacts,
    subject_dirs,
    filter_to_classes,
    subject_dependent_split,
    subject_independent_split,
    smote_augment,
)

mne.set_log_level("ERROR")

CACHE_PATH = Path(__file__).parent.parent / ".cache" / "eegnet_epochs.npz"

LR_CLASSES = ["left_fist", "right_fist"]

EEGNET_CHANNELS = [
    "Fc3", "Fc1", "Fcz", "Fc2", "Fc4",
    "C3", "C1", "Cz", "C2", "C4",
    "Cp3", "Cp1", "Cpz", "Cp2", "Cp4",
]


def load_subject_epochs_eegnet(sid: int, subj_dir: Path):
    Xs, ys = [], []
    for run, label_map in RUN_LABELS.items():
        edf_path = subj_dir / f"S{sid:03d}R{run:02d}.edf"
        if not edf_path.exists():
            continue
        raw = mne.io.read_raw_edf(edf_path, preload=True, verbose="ERROR")
        raw.rename_channels({ch: ch.rstrip(".") for ch in raw.ch_names})
        raw.set_montage(MONTAGE, on_missing="ignore")
        raw.notch_filter(50.0, verbose="ERROR")
        raw = remove_ica_artifacts(raw)
        raw.pick(EEGNET_CHANNELS)
        raw.filter(4.0, 40.0, fir_design="firwin", verbose="ERROR")

        events, event_id = mne.events_from_annotations(raw, verbose="ERROR")
        wanted = {k: v for k, v in event_id.items() if k in label_map}
        if not wanted:
            continue
        epochs = mne.Epochs(
            raw, events, event_id=wanted,
            tmin=EPOCH_TMIN, tmax=EPOCH_TMAX,
            baseline=None, preload=True, verbose="ERROR",
        )
        data = epochs.get_data(copy=True)[:, :, :N_SAMPLES].astype(np.float32)
        codes = epochs.events[:, 2]
        code_to_desc = {v: k for k, v in wanted.items()}
        labels = np.array(
            [CLASS_TO_IDX[label_map[code_to_desc[c]]] for c in codes],
            dtype=np.int64,
        )
        Xs.append(data)
        ys.append(labels)

    if not Xs:
        return None, None
    X = np.concatenate(Xs, axis=0)
    y = np.concatenate(ys, axis=0)

    mean = X.mean(axis=(0, 2), keepdims=True)
    std = X.std(axis=(0, 2), keepdims=True) + 1e-8
    X = (X - mean) / std
    return X, y


def build_dataset_eegnet(max_subjects=None, use_cache=True):
    if use_cache and CACHE_PATH.exists():
        print(f"Loading cached epochs from {CACHE_PATH}")
        d = np.load(CACHE_PATH)
        return d["X"], d["y"], d["groups"]

    subs = subject_dirs(DATA_ROOT, max_subjects)
    print(f"Processing {len(subs)} subjects from {DATA_ROOT}")
    Xs, ys, groups = [], [], []
    t0 = time.time()
    for i, (sid, subj_dir) in enumerate(subs, 1):
        X, y = load_subject_epochs_eegnet(sid, subj_dir)
        if X is None:
            print(f"  [{i}/{len(subs)}] S{sid:03d}: no usable runs, skipped")
            continue
        Xs.append(X)
        ys.append(y)
        groups.append(np.full(len(y), sid, dtype=np.int64))
        print(f"  [{i}/{len(subs)}] S{sid:03d}: {len(y)} epochs "
              f"({time.time()-t0:.0f}s elapsed)")

    X = np.concatenate(Xs, axis=0)
    y = np.concatenate(ys, axis=0)
    groups = np.concatenate(groups, axis=0)

    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(CACHE_PATH, X=X, y=y, groups=groups)
    print(f"Cached processed epochs to {CACHE_PATH}")
    return X, y, groups


class RandAugmentEEG:
    """RandAugment (Cubuk et al., 2019) adapted to EEG time series.
    Same as train_eegnet.py -- see that file for the full rationale."""

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


class EEGNet(nn.Module):
    """Direct port of EEGModels.py's `EEGNet` (Keras/TF) to PyTorch.
    Same architecture as train_eegnet.py -- see that file for details."""

    def __init__(self, n_classes, chans, samples,
                 dropout_rate=0.5, kernel_length=64,
                 F1=8, D=2, F2=None, d_model=None, norm_rate=0.25,
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
        self.elu1 = nn.ELU()
        self.pool1 = nn.AvgPool2d((1, 4))
        self.drop1 = DropoutCls(dropout_rate)

        self.sep_depthwise = nn.Conv2d(F1 * D, F1 * D, kernel_size=(1, 16),
                                        padding="same", groups=F1 * D, bias=False)
        self.sep_pointwise = nn.Conv2d(F1 * D, F2, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(F2)
        self.elu2 = nn.ELU()
        self.pool2 = nn.AvgPool2d((1, 8))
        self.drop2 = DropoutCls(dropout_rate)

        flat_samples = samples // 4 // 8
        flat_features = F2 * flat_samples
        # Optional bottleneck between the conv stack and the classifier,
        # mirroring ViewEncoder's chunk-token projection (transformer/*/
        # train_multiview_transformer.py) -- see train_eegnet.py for why.
        self.proj = nn.Linear(flat_features, d_model) if d_model else None
        self.classifier = nn.Linear(d_model if d_model else flat_features, n_classes)

    def forward(self, x):
        x = x.unsqueeze(1)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.depthwise(x)
        x = self.bn2(x)
        x = self.elu1(x)
        x = self.pool1(x)
        x = self.drop1(x)

        x = self.sep_depthwise(x)
        x = self.sep_pointwise(x)
        x = self.bn3(x)
        x = self.elu2(x)
        x = self.pool2(x)
        x = self.drop2(x)

        x = x.flatten(1)
        if self.proj is not None:
            x = self.proj(x)
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
    ap.add_argument("--d-model", type=int, default=128,
                     help="Bottleneck Linear(flat_features -> d_model) inserted before the "
                          "classifier, mirroring ViewEncoder's chunk-token projection in the "
                          "transformer scripts -- the lever for matching EEGNet's parameter count "
                          "against theirs. 0 disables it: classify directly off the flattened "
                          "conv features, the original paper's design.")
    ap.add_argument("--kernel-length", type=int, default=80,
                     help="Temporal kernel length (paper rule: half the sampling rate; "
                          "160Hz here -> 80, vs. their 128Hz -> 64)")
    ap.add_argument("--dropout", type=float, default=0.5,
                     help="Paper: 0.5 within-subject, 0.25 cross-subject")
    ap.add_argument("--dropout-type", choices=["Dropout", "SpatialDropout2D"], default="Dropout",
                     help="Reference code recommends plain Dropout for oscillatory "
                          "paradigms like motor imagery -- SpatialDropout2D hurt their SMR results")
    ap.add_argument("--norm-rate", type=float, default=0.25)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--test-frac", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--split-mode", choices=["subject-dependent", "subject-independent"],
                     default="subject-dependent")
    ap.add_argument("--no-smote", action="store_true")
    ap.add_argument("--smote-k", type=int, default=5)
    ap.add_argument("--no-augment", action="store_true",
                     help="Disable RandAugment-style training-time data augmentation")
    ap.add_argument("--randaugment-n", type=int, default=2,
                     help="Number of augmentation ops applied per sample (RandAugment default: 2)")
    ap.add_argument("--randaugment-m", type=float, default=0.5,
                     help="Shared augmentation magnitude in [0,1] (RandAugment 'M')")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    X, y, groups = build_dataset_eegnet(max_subjects=args.max_subjects,
                                         use_cache=not args.no_cache)
    X, y, groups = filter_to_classes(X, y, groups, LR_CLASSES)
    print(f"Dataset (left_fist vs right_fist only): X={X.shape}, classes={LR_CLASSES}, "
          f"subjects={len(set(groups.tolist()))}")
    print("Class counts:", {c: int((y == i).sum()) for i, c in enumerate(LR_CLASSES)})

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
        counts = {c: int((y_train == i).sum()) for i, c in enumerate(LR_CLASSES)}
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

    n_channels, n_times = X_train.shape[1], X_train.shape[2]
    model = EEGNet(len(LR_CLASSES), n_channels, n_times,
                   dropout_rate=args.dropout, kernel_length=args.kernel_length,
                   F1=args.f1, D=args.d, F2=args.f2, d_model=args.d_model or None,
                   norm_rate=args.norm_rate, dropout_type=args.dropout_type).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

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

    test_loss, test_acc = run_epoch_with_constraint(test_loader, train=False)
    print(f"\nTest loss={test_loss:.4f}  Test accuracy={test_acc:.4f}")

    model.eval()
    all_preds, all_true = [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            preds = model(xb.to(device)).argmax(1).cpu().numpy()
            all_preds.append(preds)
            all_true.append(yb.numpy())
    all_preds = np.concatenate(all_preds)
    all_true = np.concatenate(all_true)

    print("\nClassification report (test set):")
    print(classification_report(all_true, all_preds, target_names=LR_CLASSES, digits=3))
    print("Confusion matrix (rows=true, cols=pred):")
    print(LR_CLASSES)
    print(confusion_matrix(all_true, all_preds))


if __name__ == "__main__":
    main()
