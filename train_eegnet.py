
"""EEGnet: (~50% test, major overfitting though)"""

import argparse
import time
from pathlib import Path

import mne
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix

from train_mlp import (
    CLASSES,
    DATA_ROOT,
    MONTAGE,
    RUN_LABELS,
    CLASS_TO_IDX,
    EPOCH_TMIN,
    EPOCH_TMAX,
    N_SAMPLES,
    remove_ica_artifacts,
    subject_dirs,
    subject_dependent_split,
    subject_independent_split,
    smote_augment,
    run_epoch,
)

mne.set_log_level("ERROR")

CACHE_PATH = Path(__file__).parent / ".cache" / "eegnet_epochs.npz"

# Wider motor-strip montage (not channel pairs) -- EEGNet's spatial
# filter needs multiple simultaneous channels to do anything meaningful.
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

    # Per-subject z-score normalization, same rationale as train_mlp.py.
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


class EEGNet(nn.Module):
    """Direct port of EEGModels.py's `EEGNet` (Keras/TF) to PyTorch.

    Input is (batch, chans, samples); internally reshaped to
    (batch, 1, chans, samples) to match the reference's NHWC-style
    (Chans, Samples, 1) convention, ported to PyTorch's NCHW.
    """

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

        # Block 1: temporal conv -> depthwise (spatial) conv
        self.conv1 = nn.Conv2d(1, F1, kernel_size=(1, kernel_length),
                                padding="same", bias=False)
        self.bn1 = nn.BatchNorm2d(F1)
        self.depthwise = nn.Conv2d(F1, F1 * D, kernel_size=(chans, 1),
                                    groups=F1, bias=False)
        self.bn2 = nn.BatchNorm2d(F1 * D)
        self.elu1 = nn.ELU()
        self.pool1 = nn.AvgPool2d((1, 4))
        self.drop1 = DropoutCls(dropout_rate)

        # Block 2: separable conv (depthwise + pointwise)
        self.sep_depthwise = nn.Conv2d(F1 * D, F1 * D, kernel_size=(1, 16),
                                        padding="same", groups=F1 * D, bias=False)
        self.sep_pointwise = nn.Conv2d(F1 * D, F2, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(F2)
        self.elu2 = nn.ELU()
        self.pool2 = nn.AvgPool2d((1, 8))
        self.drop2 = DropoutCls(dropout_rate)

        flat_samples = samples // 4 // 8
        self.classifier = nn.Linear(F2 * flat_samples, n_classes)

    def forward(self, x):
        x = x.unsqueeze(1)  # (batch, 1, chans, samples)

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
        return self.classifier(x)

    @torch.no_grad()
    def apply_max_norm(self):
        """Keras' max_norm weight constraints, applied as a hard clip
        after each optimizer step (Keras applies these as constraints
        baked into the layer; PyTorch has no equivalent, so this must
        be called manually post-step). Clips per-output-filter L2 norm:
        depthwise conv to 1.0 (Table 2), final Dense to norm_rate=0.25."""
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
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    X, y, groups = build_dataset_eegnet(max_subjects=args.max_subjects,
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

    def make_loader(Xa, ya, shuffle):
        ds = torch.utils.data.TensorDataset(
            torch.from_numpy(Xa).float(), torch.from_numpy(ya).long()
        )
        return torch.utils.data.DataLoader(ds, batch_size=args.batch_size, shuffle=shuffle)

    train_loader = make_loader(X_train, y_train, shuffle=True)
    val_loader = make_loader(X_val, y_val, shuffle=False)
    test_loader = make_loader(X_test, y_test, shuffle=False)

    n_channels, n_times = X_train.shape[1], X_train.shape[2]
    model = EEGNet(len(CLASSES), n_channels, n_times,
                   dropout_rate=args.dropout, kernel_length=args.kernel_length,
                   F1=args.f1, D=args.d, F2=args.f2, norm_rate=args.norm_rate,
                   dropout_type=args.dropout_type).to(device)
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
    print(classification_report(all_true, all_preds, target_names=CLASSES, digits=3))
    print("Confusion matrix (rows=true, cols=pred):")
    print(CLASSES)
    print(confusion_matrix(all_true, all_preds))


if __name__ == "__main__":
    main()
