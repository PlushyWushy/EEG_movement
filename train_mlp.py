
"""
MLP                       
"""

import argparse
import time
from pathlib import Path

import mne
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.neighbors import NearestNeighbors

mne.set_log_level("ERROR")

DATA_ROOT = Path(__file__).parent / "physionet.org" / "files" / "eegmmidb" / "1.0.0"
CACHE_PATH = Path(__file__).parent / ".cache" / "mi_epochs.npz"

EXCLUDED_SUBJECTS = {38, 88, 89, 92, 100, 104}


RUN_LABELS = {
    4: {"T0": "baseline", "T1": "left_fist", "T2": "right_fist"},
    8: {"T0": "baseline", "T1": "left_fist", "T2": "right_fist"},
    12: {"T0": "baseline", "T1": "left_fist", "T2": "right_fist"},
    6: {"T0": "baseline", "T1": "both_fists", "T2": "both_feet"},
    10: {"T0": "baseline", "T1": "both_fists", "T2": "both_feet"},
    14: {"T0": "baseline", "T1": "both_fists", "T2": "both_feet"},
}
CLASSES = ["left_fist", "right_fist", "both_fists", "both_feet", "baseline"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}

# CNN-GRU paper's best sensorimotor area (SMA E): 6 symmetric electrode
# pairs, each fed as an INDEPENDENT 2-channel sample carrying the trial's
# label -- not all channels together. One real trial becomes 6 training
# samples (an implicit ~6x data multiplier), and every filter is forced
# to learn a hemisphere-symmetric local pattern rather than an arbitrary
# combination across many electrodes at once.
CHANNEL_PAIRS = [
    ("Fc1", "Fc2"), ("Fc3", "Fc4"),
    ("C3", "C4"), ("C1", "C2"),
    ("Cp1", "Cp2"), ("Cp3", "Cp4"),
]
PAIR_CHANNELS = sorted({ch for pair in CHANNEL_PAIRS for ch in pair})

EPOCH_TMIN, EPOCH_TMAX = 0.0, 4.0
N_SAMPLES = 640

MONTAGE = mne.channels.make_standard_montage("standard_1005")


def remove_ica_artifacts(raw, n_components=15, seed=42):
    """Fit ICA on a 1-45 Hz copy of the full 64-channel signal, flag
    components correlated with eye blinks or matching muscle-artifact
    criteria, and remove them -- the CNN-GRU paper's rationale: ocular
    and muscular artifacts spectrally overlap the EEG band and survive
    plain bandpass filtering, so ICA is needed on top of it. This dataset
    has no dedicated EOG channel, so Fp1 (frontal) is used as a proxy for
    eye-blink correlation, matching common practice when none exists."""
    raw_for_ica = raw.copy().filter(1.0, 45.0, fir_design="firwin", verbose="ERROR")
    ica = mne.preprocessing.ICA(n_components=n_components, method="fastica",
                                 random_state=seed, max_iter="auto")
    ica.fit(raw_for_ica, verbose="ERROR")

    eog_idx, _ = ica.find_bads_eog(raw_for_ica, ch_name="Fp1", verbose="ERROR")
    muscle_idx, _ = ica.find_bads_muscle(raw_for_ica, verbose="ERROR")
    ica.exclude = list(set(eog_idx) | set(muscle_idx))

    ica.apply(raw, verbose="ERROR")
    return raw


def subject_dirs(root: Path, max_subjects=None):
    subs = []
    for p in sorted(root.glob("S*")):
        if not p.is_dir():
            continue
        sid = int(p.name[1:])
        if sid in EXCLUDED_SUBJECTS:
            continue
        subs.append((sid, p))
    if max_subjects:
        subs = subs[:max_subjects]
    return subs


def load_subject_epochs(sid: int, subj_dir: Path):
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
        raw.pick(PAIR_CHANNELS)
        raw.filter(8.0, 30.0, method="iir",
                   iir_params=dict(order=5, ftype="butter"), verbose="ERROR")

        events, event_id = mne.events_from_annotations(raw, verbose="ERROR")
        wanted = {k: v for k, v in event_id.items() if k in label_map}
        if not wanted:
            continue
        epochs = mne.Epochs(
            raw, events, event_id=wanted,
            tmin=EPOCH_TMIN, tmax=EPOCH_TMAX,
            baseline=None, preload=True, verbose="ERROR",
        )
        codes = epochs.events[:, 2]
        code_to_desc = {v: k for k, v in wanted.items()}
        labels_full = np.array(
            [CLASS_TO_IDX[label_map[code_to_desc[c]]] for c in codes],
            dtype=np.int64,
        )
        data_full = epochs.get_data(copy=True)[:, :, :N_SAMPLES].astype(np.float32)

        for ch_a, ch_b in CHANNEL_PAIRS:
            idx_a = epochs.ch_names.index(ch_a)
            idx_b = epochs.ch_names.index(ch_b)
            Xs.append(data_full[:, [idx_a, idx_b], :])
            ys.append(labels_full)

    if not Xs:
        return None, None
    X = np.concatenate(Xs, axis=0)
    y = np.concatenate(ys, axis=0)

    # Per-subject z-score normalization (paper: "normalizing within
    # participants" to remove subject-specific amplitude/baseline
    # differences before pooling across subjects). This only uses this
    # subject's own signal scale -- available at deployment time before
    # any classification -- so it isn't a train/test leakage concern.
    mean = X.mean(axis=(0, 2), keepdims=True)
    std = X.std(axis=(0, 2), keepdims=True) + 1e-8
    X = (X - mean) / std
    return X, y


def build_dataset(max_subjects=None, use_cache=True):
    if use_cache and CACHE_PATH.exists():
        print(f"Loading cached epochs from {CACHE_PATH}")
        d = np.load(CACHE_PATH)
        return d["X"], d["y"], d["groups"]

    subs = subject_dirs(DATA_ROOT, max_subjects)
    print(f"Processing {len(subs)} subjects from {DATA_ROOT}")
    Xs, ys, groups = [], [], []
    t0 = time.time()
    for i, (sid, subj_dir) in enumerate(subs, 1):
        X, y = load_subject_epochs(sid, subj_dir)
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


def subject_independent_split(groups, val_frac=0.15, test_frac=0.15, seed=42):

    unique_subjects = sorted(set(groups.tolist()))
    rng = np.random.default_rng(seed)
    rng.shuffle(unique_subjects)

    n = len(unique_subjects)
    n_test = max(1, round(n * test_frac))
    n_val = max(1, round(n * val_frac))

    test_subjects = set(unique_subjects[:n_test])
    val_subjects = set(unique_subjects[n_test:n_test + n_val])
    train_subjects = set(unique_subjects[n_test + n_val:])

    train_mask = np.isin(groups, list(train_subjects))
    val_mask = np.isin(groups, list(val_subjects))
    test_mask = np.isin(groups, list(test_subjects))

    print(f"Subject-independent split: {len(train_subjects)} train / "
          f"{len(val_subjects)} val / {len(test_subjects)} test subjects "
          f"(disjoint -- every subject appears in exactly one split)")
    return train_mask, val_mask, test_mask


def subject_dependent_split(y, groups, val_frac=0.15, test_frac=0.15, seed=42):

    rng = np.random.default_rng(seed)
    n = len(y)
    train_mask = np.zeros(n, dtype=bool)
    val_mask = np.zeros(n, dtype=bool)
    test_mask = np.zeros(n, dtype=bool)

    for sid in sorted(set(groups.tolist())):
        subj_idx = np.where(groups == sid)[0]
        for c in np.unique(y[subj_idx]):
            class_idx = subj_idx[y[subj_idx] == c].copy()
            rng.shuffle(class_idx)
            n_c = len(class_idx)
            n_test = max(1, round(n_c * test_frac))
            n_val = max(1, round(n_c * val_frac))
            test_mask[class_idx[:n_test]] = True
            val_mask[class_idx[n_test:n_test + n_val]] = True
            train_mask[class_idx[n_test + n_val:]] = True

    n_subjects = len(set(groups.tolist()))
    print(f"Subject-dependent split: every one of {n_subjects} subjects "
          f"contributes trials to train/val/test (stratified by class)")
    return train_mask, val_mask, test_mask


def smote_augment(X, y, k=5, seed=42):
    """SMOTE oversampling, same algorithm as the CNN-GRU paper (Sec 3.3.3):
    for each class below the majority count, repeatedly pick a real
    sample x_i, one of its k nearest same-class neighbors x_j, and
    synthesize x_i + lambda*(x_j - x_i) with lambda ~ Uniform(0,1),
    until every class matches the majority count. Neighbor search and
    interpolation happen on flattened (n_channels*n_times) vectors, then
    results are reshaped back to (n_channels, n_times). Call this on the
    TRAINING split only, after normalization -- val/test must stay real."""
    rng = np.random.default_rng(seed)
    n, c, t = X.shape
    X_flat = X.reshape(n, c * t)
    class_counts = {cls: int((y == cls).sum()) for cls in np.unique(y)}
    majority_count = max(class_counts.values())

    synth_X, synth_y = [], []
    for cls, count in class_counts.items():
        n_needed = majority_count - count
        if n_needed <= 0:
            continue
        cls_idx = np.where(y == cls)[0]
        cls_X = X_flat[cls_idx]
        n_neighbors = min(k + 1, len(cls_idx))  # +1 to drop self at distance 0
        nn_model = NearestNeighbors(n_neighbors=n_neighbors).fit(cls_X)
        _, neighbor_idx = nn_model.kneighbors(cls_X)

        for _ in range(n_needed):
            i = rng.integers(len(cls_idx))
            candidates = neighbor_idx[i][1:]
            j = candidates[rng.integers(len(candidates))]
            lam = rng.uniform(0.0, 1.0)
            synth_X.append(cls_X[i] + lam * (cls_X[j] - cls_X[i]))
            synth_y.append(cls)

    if not synth_X:
        print(f"SMOTE: classes already balanced ({class_counts}), nothing generated")
        return X, y

    synth_X = np.stack(synth_X).reshape(-1, c, t).astype(X.dtype)
    synth_y = np.array(synth_y, dtype=y.dtype)
    print(f"SMOTE: class counts before {class_counts} -> "
          f"generated {len(synth_y)} synthetic samples -> "
          f"target {majority_count} per class")
    return np.concatenate([X, synth_X], axis=0), np.concatenate([y, synth_y], axis=0)


class MLP(nn.Module):


    def __init__(self, input_dim, hidden_dim, num_classes, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x):
        return self.net(x)


def run_epoch(model, loader, criterion, optimizer, device, train):
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
        total_loss += loss.item() * len(yb)
        correct += (logits.argmax(1) == yb).sum().item()
        n += len(yb)
    return total_loss / n, correct / n


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--max-subjects", type=int, default=None,
                     help="Limit number of subjects processed (default: all available)")
    ap.add_argument("--no-cache", action="store_true",
                     help="Ignore/overwrite the cached processed-epoch array")
    ap.add_argument("--hidden-dim", type=int, default=256)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--patience", type=int, default=10,
                     help="Early-stopping patience (epochs without val-loss improvement)")
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--test-frac", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--split-mode", choices=["subject-dependent", "subject-independent"],
                     default="subject-dependent",
                     help="'subject-dependent' (default): every subject contributes to "
                          "train/val/test, matching the reviewed papers' protocol. "
                          "'subject-independent': whole subjects held out for val/test, "
                          "testing generalization to unseen users.")
    ap.add_argument("--no-smote", action="store_true",
                     help="Disable SMOTE oversampling of minority classes in the training set")
    ap.add_argument("--smote-k", type=int, default=5,
                     help="Nearest neighbors used by SMOTE (CNN-GRU paper default: 5)")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    X, y, groups = build_dataset(max_subjects=args.max_subjects,
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

    # Normalization is done per-subject at load time (see load_subject_epochs),
    # matching the paper's "normalizing within participants" -- no global
    # train-set normalization step needed here.

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

    input_dim = X_train.shape[1] * X_train.shape[2]
    model = MLP(input_dim, args.hidden_dim, len(CLASSES), args.dropout).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val_loss = float("inf")
    best_state = None
    epochs_without_improvement = 0

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, optimizer, device, train=False)
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

    test_loss, test_acc = run_epoch(model, test_loader, criterion, optimizer, device, train=False)
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
