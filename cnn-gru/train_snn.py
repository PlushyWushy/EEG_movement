
"""

"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import snntorch as snn
import snntorch.functional as SF
from snntorch import surrogate
from sklearn.metrics import classification_report, confusion_matrix

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from mlp.train_mlp import (
    CLASSES,
    N_SAMPLES,
    CHANNEL_PAIRS,
    PAIR_CHANNELS,
    build_dataset_raw,
    smote_augment,
    subject_dependent_split,
    subject_independent_split,
)

CACHE_PATH = Path(__file__).parent.parent / ".cache" / "mi_epochs_raw.npz"


class ConvSNN(nn.Module):
    """Conv1d + LIF stack mirroring train_cnn-gru-raw.py's CNN1D feature
    extractor (same kernel sizes/order), with the GRU+FC head replaced by
    a global-average-pool -> FC -> spiking readout, so the only recurrence
    in the whole network is each LIF neuron's own membrane leak."""

    def __init__(self, n_channels, num_classes,
                 n_filters1=32, n_filters2=32,
                 kernel_size1=20, kernel_size2=6,
                 fc_dim=64, beta=0.9, dropout=0.5, num_steps=25):
        super().__init__()
        self.num_steps = num_steps
        spike_grad = surrogate.fast_sigmoid()

        self.conv1 = nn.Conv1d(n_channels, n_filters1, kernel_size1, padding="same")
        self.bn1 = nn.BatchNorm1d(n_filters1)
        self.lif1 = snn.Leaky(beta=beta, spike_grad=spike_grad)

        self.conv2 = nn.Conv1d(n_filters1, n_filters2, kernel_size1, padding="valid")
        self.bn2 = nn.BatchNorm1d(n_filters2)
        self.lif2 = snn.Leaky(beta=beta, spike_grad=spike_grad)
        self.drop1 = nn.Dropout1d(dropout)

        self.conv3 = nn.Conv1d(n_filters2, n_filters2, kernel_size2, padding="valid")
        self.bn3 = nn.BatchNorm1d(n_filters2)
        self.lif3 = snn.Leaky(beta=beta, spike_grad=spike_grad)
        self.pool1 = nn.AvgPool1d(2)

        self.conv4 = nn.Conv1d(n_filters2, n_filters2, kernel_size2, padding="valid")
        self.bn4 = nn.BatchNorm1d(n_filters2)
        self.lif4 = snn.Leaky(beta=beta, spike_grad=spike_grad)
        self.drop2 = nn.Dropout1d(dropout)

        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(n_filters2, fc_dim)
        self.bn5 = nn.BatchNorm1d(fc_dim)
        self.lif5 = snn.Leaky(beta=beta, spike_grad=spike_grad)
        self.drop3 = nn.Dropout(dropout)

        self.fc2 = nn.Linear(fc_dim, num_classes)
        # Non-spiking leaky-integrator readout (reset_mechanism="none"):
        # classification uses the continuous membrane potential, not a
        # spike count. A spiking readout collapsed to zero output spikes
        # during training (cross entropy over a binary, mostly-zero spike
        # count gives near-flat gradients once every class stops firing --
        # a known SNN failure mode), so the readout follows the standard
        # SpyTorch/snnTorch fix of reading out membrane potential instead
        # (see snntorch.functional.ce_max_membrane_loss).
        self.lif_out = snn.Leaky(beta=beta, spike_grad=spike_grad, reset_mechanism="none")

    def forward(self, x):
        # x: (batch, n_channels, n_times), presented unchanged at every step.
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        mem3 = self.lif3.init_leaky()
        mem4 = self.lif4.init_leaky()
        mem5 = self.lif5.init_leaky()
        mem_out = self.lif_out.init_leaky()

        mem_rec = []
        spk5_rec = []
        for _ in range(self.num_steps):
            cur1 = self.bn1(self.conv1(x))
            spk1, mem1 = self.lif1(cur1, mem1)

            cur2 = self.bn2(self.conv2(spk1))
            spk2, mem2 = self.lif2(cur2, mem2)
            spk2 = self.drop1(spk2)

            cur3 = self.bn3(self.conv3(spk2))
            spk3, mem3 = self.lif3(cur3, mem3)
            spk3 = self.pool1(spk3)

            cur4 = self.bn4(self.conv4(spk3))
            spk4, mem4 = self.lif4(cur4, mem4)
            spk4 = self.drop2(spk4)

            pooled = self.global_pool(spk4).flatten(1)
            cur5 = self.bn5(self.fc1(pooled))
            spk5, mem5 = self.lif5(cur5, mem5)
            spk5 = self.drop3(spk5)
            spk5_rec.append(spk5)

            cur_out = self.fc2(spk5)
            _, mem_out = self.lif_out(cur_out, mem_out)
            mem_rec.append(mem_out)

        # (num_steps, batch, num_classes) membrane trace for the readout,
        # (num_steps, batch, fc_dim) spike trace from the last spiking
        # layer, kept only to report a genuine spike rate (sparsity is
        # the whole point of a spiking classifier).
        return torch.stack(mem_rec, dim=0), torch.stack(spk5_rec, dim=0)


def run_epoch(model, loader, loss_fn, optimizer, device, train):
    model.train(train)
    total_loss, correct, n = 0.0, 0.0, 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        if train:
            optimizer.zero_grad()
        mem_rec, _ = model(xb)
        loss = loss_fn(mem_rec, yb)
        if train:
            loss.backward()
            optimizer.step()
        total_loss += loss.item() * len(yb)
        preds = mem_rec.max(0).values.argmax(1)
        correct += (preds == yb).sum().item()
        n += len(yb)
    return total_loss / n, correct / n


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--max-subjects", type=int, default=None,
                     help="Limit number of subjects processed (default: all available)")
    ap.add_argument("--no-cache", action="store_true",
                     help="Ignore/overwrite the cached processed-epoch array")
    ap.add_argument("--n-filters1", type=int, default=32)
    ap.add_argument("--n-filters2", type=int, default=32)
    ap.add_argument("--kernel-size1", type=int, default=20)
    ap.add_argument("--kernel-size2", type=int, default=6)
    ap.add_argument("--fc-dim", type=int, default=64)
    ap.add_argument("--dropout", type=float, default=0.5)
    ap.add_argument("--beta", type=float, default=0.9,
                     help="LIF membrane decay rate (snntorch default-ish; higher = longer memory)")
    ap.add_argument("--num-steps", type=int, default=25,
                     help="Simulation timesteps per (static) input epoch")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--patience", type=int, default=15,
                     help="Early-stopping patience (epochs without val-loss improvement)")
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--test-frac", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--split-mode", choices=["subject-dependent", "subject-independent"],
                     default="subject-dependent")
    ap.add_argument("--no-smote", action="store_true",
                     help="Disable SMOTE oversampling of minority classes in the training set")
    ap.add_argument("--smote-k", type=int, default=5)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    X, y, groups = build_dataset_raw(CACHE_PATH, PAIR_CHANNELS, expand_pairs=CHANNEL_PAIRS,
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

    def make_loader(Xa, ya, shuffle):
        ds = torch.utils.data.TensorDataset(
            torch.from_numpy(Xa).float(), torch.from_numpy(ya).long()
        )
        return torch.utils.data.DataLoader(ds, batch_size=args.batch_size, shuffle=shuffle)

    train_loader = make_loader(X_train, y_train, shuffle=True)
    val_loader = make_loader(X_val, y_val, shuffle=False)
    test_loader = make_loader(X_test, y_test, shuffle=False)

    n_channels, n_times = X_train.shape[1], X_train.shape[2]
    assert n_times == N_SAMPLES
    model = ConvSNN(n_channels, len(CLASSES),
                     n_filters1=args.n_filters1, n_filters2=args.n_filters2,
                     kernel_size1=args.kernel_size1, kernel_size2=args.kernel_size2,
                     fc_dim=args.fc_dim, beta=args.beta, dropout=args.dropout,
                     num_steps=args.num_steps).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}  (num_steps={args.num_steps})")

    loss_fn = SF.ce_max_membrane_loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val_loss = float("inf")
    best_state = None
    epochs_without_improvement = 0

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss, train_acc = run_epoch(model, train_loader, loss_fn, optimizer, device, train=True)
        val_loss, val_acc = run_epoch(model, val_loader, loss_fn, optimizer, device, train=False)
        print(f"epoch {epoch:3d}  train_loss={train_loss:.4f} train_acc={train_acc:.4f}  "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}  ({time.time()-t0:.0f}s)",
              flush=True)

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

    test_loss, test_acc = run_epoch(model, test_loader, loss_fn, optimizer, device, train=False)
    print(f"\nTest loss={test_loss:.4f}  Test accuracy={test_acc:.4f}")

    model.eval()
    all_preds, all_true, spike_rate_sum, n_batches = [], [], 0.0, 0
    with torch.no_grad():
        for xb, yb in test_loader:
            mem_rec, spk5_rec = model(xb.to(device))
            preds = mem_rec.max(0).values.argmax(1).cpu().numpy()
            all_preds.append(preds)
            all_true.append(yb.numpy())
            spike_rate_sum += spk5_rec.mean().item()
            n_batches += 1
    all_preds = np.concatenate(all_preds)
    all_true = np.concatenate(all_true)

    print("\nClassification report (test set):")
    print(classification_report(all_true, all_preds, target_names=CLASSES, digits=3))
    print("Confusion matrix (rows=true, cols=pred):")
    print(CLASSES)
    print(confusion_matrix(all_true, all_preds))
    print(f"\nMean spike rate, last spiking layer (test set, fraction of "
          f"neuron-timesteps that fired -- the sparsity relevant for "
          f"neuromorphic-hardware power draw): {spike_rate_sum / n_batches:.4f}")


if __name__ == "__main__":
    main()
